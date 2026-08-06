"""Auto-fix pipeline for `:diagnose`. A deliberately separate code path from
agent_maker/generator.py's additive-only self-extension pipeline — that one
is locked to never edit an existing agent's files (create-only lifecycle);
this one exists specifically to, triggered by a human running `:diagnose`
in the REPL, never automatically on its own.

Reuses generator.py's low-level machinery (write-confinement guard/hook,
Bedrock-vs-Anthropic provider options, the ResultMessage upstream-failure
signal) since none of that is specific to "building a brand-new agent" —
only the system prompt and the starting point (an existing directory with
existing code, not an empty one) differ.

Test-gated the same way generation is: sandbox_runner.run_tests(agent_dir)
must pass — including every pre-existing test, not just a new one for this
bug — before a fix is considered landed. On repeated failure (or an
upstream API failure with no other provider to fall back to), the agent's
original files are restored byte-for-byte via a snapshot taken before any
edit — never left half-fixed or broken.
"""

import asyncio
from pathlib import Path

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    HookMatcher,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
)

from agent_maker import sandbox_runner
from agent_maker.generator import (
    MAX_TURNS,
    REPO_ROOT,
    _bedrock_options,
    _confinement_guard,
    _confinement_hook,
    _is_upstream_api_failure,
)
from orchestrator.agent_loader import discover
from orchestrator.llm import bedrock_configured

FIX_MODEL = "claude-opus-4-8"
MAX_FIX_RETRIES = 2

_FIX_SYSTEM_PROMPT = """\
You are fixing a specific, diagnosed bug in an EXISTING, already-registered \
agent inside a self-extending multi-agent system. You work only inside this \
agent's own directory ({agent_dir}) and must not touch anything outside it \
(same write-confinement as this system's own agent generator).

## The locked State schema (read-only contract)

```python
{state_schema_source}
```

Every agent may READ the full state but may only WRITE an append to \
"messages" and agent_scratchpad[AGENT_ID]. This is already enforced by the \
existing code — do not weaken it.

## The diagnosed bug

{bug_report}

## Your task

1. Read agent.py and test_agent.py (and any sibling module in this \
directory) FIRST, to understand the existing implementation and test \
patterns before changing anything.
2. Make the SMALLEST correct fix for the diagnosed bug. This is a bug fix, \
not a rewrite — do not refactor, rename, or restructure anything unrelated \
to it.
3. Add a new regression test in test_agent.py that reproduces this exact \
bug scenario: it must fail against the old (unfixed) behavior and pass \
against your fix. Use the same recording-fake-LLM / mocking pattern already \
used by the existing tests in this file.
4. Every pre-existing test in test_agent.py must still pass. If a test's \
own assumption was itself wrong (rare — check carefully before concluding \
this), fix the test, but never weaken an assertion just to make it pass.
5. Run nothing yourself. The sandbox runs test_agent.py after you finish \
and will send you the failure output if anything fails, the same as \
generation.

Keep the diff as small as the bug allows. When done, state DONE and list \
the files you changed.
"""


class FixSession:
    """One conversation with the coding harness, scoped to an EXISTING
    agent's directory instead of a fresh one — see module docstring for why
    this needs its own class rather than reusing generator.GenerationSession
    (which assumes an AgentSpec and an empty directory)."""

    def __init__(self, agent_dir: Path, bug_report: str, progress=print, use_bedrock: bool = False):
        self.agent_dir = agent_dir
        self.progress = progress
        self.last_result: ResultMessage | None = None
        model, env = (_bedrock_options() if use_bedrock else (FIX_MODEL, {}))
        state_schema_source = (REPO_ROOT / "orchestrator" / "state.py").read_text()
        system_prompt = _FIX_SYSTEM_PROMPT.format(
            agent_dir=str(agent_dir),
            state_schema_source=state_schema_source,
            bug_report=bug_report,
        )
        options = ClaudeAgentOptions(
            cwd=str(agent_dir),
            system_prompt=system_prompt,
            allowed_tools=["Read", "Write", "Edit"],
            disallowed_tools=["Bash", "WebSearch", "WebFetch", "Task", "Glob", "Grep"],
            permission_mode="acceptEdits",
            can_use_tool=_confinement_guard(agent_dir),
            hooks={"PreToolUse": [HookMatcher(matcher=None, hooks=[_confinement_hook(agent_dir)])]},
            model=model,
            max_turns=MAX_TURNS,
            env=env,
        )
        self._client = ClaudeSDKClient(options=options)

    async def __aenter__(self):
        await self._client.connect()
        return self

    async def __aexit__(self, *exc):
        await self._client.disconnect()

    def upstream_api_failed(self) -> bool:
        return _is_upstream_api_failure(self.last_result)

    async def start(self) -> None:
        await self._send("Fix the bug described in your instructions now.")

    async def refine(self, failure_output: str) -> None:
        await self._send(
            "The sandbox ran `pytest test_agent.py` and it failed:\n\n"
            f"{failure_output}\n\n"
            "Fix the implementation (or the new/existing test, if it "
            "asserts something wrong) while keeping every constraint. Do "
            "not weaken tests just to make them pass."
        )

    async def _send(self, prompt: str) -> None:
        await self._client.query(prompt)
        async for message in self._client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, ToolUseBlock):
                        target = block.input.get("file_path") or block.input.get("path") or ""
                        self.progress(f"  [fixer] {block.name} {target}")
                    elif isinstance(block, TextBlock) and block.text.strip():
                        self.progress(f"  [fixer] {block.text.strip().splitlines()[0][:100]}")
            elif isinstance(message, ResultMessage):
                self.last_result = message


def _snapshot(agent_dir: Path) -> dict[str, str]:
    return {
        str(p.relative_to(agent_dir)): p.read_text()
        for p in agent_dir.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    }


def _restore(agent_dir: Path, snapshot: dict[str, str]) -> None:
    """Full restore: removes any file the fix session created that wasn't
    in the snapshot, then rewrites every original file's exact original
    content. Never leaves a failed attempt's partial edits behind."""
    for p in list(agent_dir.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts and str(p.relative_to(agent_dir)) not in snapshot:
            p.unlink()
    for rel, content in snapshot.items():
        (agent_dir / rel).write_text(content)


async def _attempt_fix(agent_dir: Path, bug_report: str, progress, use_bedrock: bool) -> tuple[bool, str, bool]:
    """Returns (passed, output, upstream_api_failed)."""
    provider = "Amazon Bedrock" if use_bedrock else "the direct Anthropic API"
    async with FixSession(agent_dir, bug_report, progress, use_bedrock=use_bedrock) as session:
        progress(f"[fixer] attempting a fix via {provider}...")
        await session.start()
        output = ""
        for attempt in range(MAX_FIX_RETRIES + 1):
            if session.upstream_api_failed():
                return False, "the fix harness's model call failed upstream", True

            progress(f"[fixer] sandbox test run {attempt + 1}/{MAX_FIX_RETRIES + 1}...")
            passed, output = sandbox_runner.run_tests(agent_dir)
            if passed:
                progress("[fixer] tests passed in sandbox.")
                return True, output, False
            if attempt < MAX_FIX_RETRIES:
                progress("[fixer] checks failed — feeding output back for a fix...")
                await session.refine(output)
        return False, output, False


def apply_fix(user_id: str, agent_id: str, bug_report: str, progress=print) -> bool:
    """Attempts to fix agent_id's code in place for the given bug report.
    Returns True if a fix landed and its (and every pre-existing) test
    passes; False, with the agent's original files fully restored,
    otherwise. Bedrock-first when configured (matching agent_maker/maker.py
    generation), falling back to the direct Anthropic API only if Bedrock's
    own call fails upstream."""
    handle = discover(user_id).get(agent_id)
    if handle is None:
        progress(f"[fixer] no such agent for user {user_id!r}: {agent_id}")
        return False

    agent_dir = handle.directory
    snapshot = _snapshot(agent_dir)

    prefer_bedrock = bedrock_configured()
    passed, output, upstream_failed = asyncio.run(
        _attempt_fix(agent_dir, bug_report, progress, use_bedrock=prefer_bedrock)
    )
    if not passed and upstream_failed and prefer_bedrock:
        progress(
            "[fixer] Bedrock's own model call failed upstream before any fix "
            "was made — retrying via the direct Anthropic API..."
        )
        _restore(agent_dir, snapshot)
        passed, output, _ = asyncio.run(_attempt_fix(agent_dir, bug_report, progress, use_bedrock=False))

    if not passed:
        progress(f"[fixer] could not produce a passing fix for {agent_id} — restoring original files.")
        _restore(agent_dir, snapshot)
        return False

    progress(f"[fixer] fix applied and verified for {agent_id}.")
    return True
