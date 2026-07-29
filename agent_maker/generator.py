"""Drives the agentic coding harness (Claude Agent SDK) to generate an agent.

Containment is mechanical, not just prompted:
  - cwd is pinned to the new agent's directory
  - only Read/Write/Edit tools are allowed (no Bash, no web)
  - a can_use_tool guard denies any file operation whose path resolves
    outside the agent directory

The system prompt template lives in prompts/generator_system_prompt.md
(git-versioned infrastructure); the worked examples are read from the real
Phase 0 agent files at assembly time so they can never drift.
"""

from pathlib import Path

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    HookMatcher,
    PermissionResultAllow,
    PermissionResultDeny,
    TextBlock,
    ToolUseBlock,
)

from agent_maker.planner import AgentSpec
from orchestrator.mcp_bindings import KNOWN_SERVICES

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_PATH = Path(__file__).parent / "prompts" / "generator_system_prompt.md"
SHARED_AGENTS = REPO_ROOT / "agents" / "_shared"

GENERATOR_MODEL = "claude-opus-4-8"
MAX_TURNS = 50

_PATH_KEYS = ("file_path", "path", "notebook_path")

# Ground truth for what a bound service actually hands run(): the SDK's own
# tool objects, not a hallucinated interface. See mcp_bindings.get_tools()
# and orchestrator/supervisor.py::_credential_gate — this is exactly what
# `tools` is at runtime.
_CREDENTIALED_TOOLS_SECTION = '''\
## Credentialed tools contract

This agent requires credentials. When the connection is approved, the
supervisor calls `run(state, system_prompt, llm, tools=bound_tools)` where
`bound_tools` is a `list` of LangChain `StructuredTool` objects — NOT an
object with one method per capability. Each tool has:

  - `.name: str` — the tool's name
  - `.invoke(dict) -> str` — call it with a dict of its arguments

Look a tool up by name and call `.invoke()`; do not guess a method name or
attribute-access the list. When `tools` is `None` (not yet connected — this
is what the sandbox and your tests always see, since there is no real
credential there), reply that the capability is not connected and touch
nothing else.

The bound service for this agent is `{service_name}`, exposing exactly these
tools (call these names and only these names):

{tool_catalog}

Worked pattern for agent.py:

```python
def run(state, system_prompt, llm, tools=None):
    if not tools:
        return {{
            "messages": [AIMessage(content="{service_name} is not connected yet.")],
            "agent_scratchpad": {{AGENT_ID: {{"connected": False}}}},
        }}
    tool = next(t for t in tools if t.name == "{example_tool_name}")
    result = tool.invoke({example_tool_args})
    ...
```

Worked pattern for test_agent.py — fake tools the same shape as real ones,
never a bare list, dict, or custom object with method-per-capability:

```python
class FakeTool:
    def __init__(self, name, result):
        self.name = name
        self.result = result
        self.calls = []

    def invoke(self, args):
        self.calls.append(args)
        return self.result

def test_uses_bound_tool_by_name():
    tool = FakeTool("{example_tool_name}", {example_tool_fake_result!r})
    update = run(make_state(), "system prompt", FakeLLM(), tools=[tool])
    assert tool.calls, "must call the tool via .invoke(), not guess a method"

def test_not_connected_when_tools_is_none():
    update = run(make_state(), "system prompt", FakeLLM(), tools=None)
    assert update["agent_scratchpad"][AGENT_ID]["connected"] is False
```
'''


def _credentialed_tools_section(spec: AgentSpec) -> str:
    if not (spec.requires_credentials and spec.mcp_service in KNOWN_SERVICES):
        return (
            "This agent does not require credentials — it must not accept a "
            "`tools` parameter or reach for any external service.\n"
        )
    entry = KNOWN_SERVICES[spec.mcp_service]
    catalog = "\n".join(
        f"  - `{t['name']}({', '.join(t['args'])})` — {t['description']}"
        for t in entry["tools"]
    )
    example_tool = entry["tools"][0]
    example_args = {name: "..." for name in example_tool["args"]}
    return _CREDENTIALED_TOOLS_SECTION.format(
        service_name=spec.mcp_service,
        tool_catalog=catalog,
        example_tool_name=example_tool["name"],
        example_tool_args=example_args,
        example_tool_fake_result="fake result",
    )


def build_system_prompt(spec: AgentSpec, agent_dir: Path, triggering_request: str) -> str:
    # Everything above the first "---" is maintainer documentation, not prompt.
    template = TEMPLATE_PATH.read_text().split("\n---\n", 1)[1]

    def example(agent: str, filename: str) -> str:
        return (SHARED_AGENTS / agent / filename).read_text()

    replacements = {
        "{state_schema_source}": (REPO_ROOT / "orchestrator" / "state.py").read_text(),
        "{general_qa_agent_py}": example("general_qa", "agent.py"),
        "{general_qa_manifest}": example("general_qa", "manifest.json"),
        "{general_qa_test_py}": example("general_qa", "test_agent.py"),
        "{summarizer_agent_py}": example("summarizer", "agent.py"),
        "{summarizer_manifest}": example("summarizer", "manifest.json"),
        "{summarizer_test_py}": example("summarizer", "test_agent.py"),
        "{agent_dir}": str(agent_dir),
        "{spec_json}": spec.model_dump_json(indent=2),
        "{triggering_request}": triggering_request,
        "{credentialed_tools_section}": _credentialed_tools_section(spec),
    }
    prompt = template
    for placeholder, value in replacements.items():
        prompt = prompt.replace(placeholder, value)
    return prompt


def _resolve_violation(agent_dir: Path, tool_name: str, tool_input: dict) -> str | None:
    for key in _PATH_KEYS:
        raw = tool_input.get(key)
        if raw is None:
            continue
        resolved = (agent_dir / raw).resolve() if not Path(raw).is_absolute() else Path(raw).resolve()
        if not resolved.is_relative_to(agent_dir):
            return (
                f"{tool_name} on {resolved} is outside the agent directory "
                f"{agent_dir}; you may only touch files inside it."
            )
    return None


def _confinement_guard(agent_dir: Path):
    """can_use_tool fallback path. NOTE: when a tool is named in allowed_tools
    as a whole-tool allow (as ours are, for Read/Write/Edit), the SDK
    auto-approves it before this callback ever runs (CanUseToolShadowedWarning)
    — so this alone does NOT enforce containment. The real enforcement is the
    PreToolUse hook below; this is kept as defense in depth for any tool call
    that does reach the callback (e.g. if allowed_tools is narrowed later)."""
    agent_dir = agent_dir.resolve()

    async def guard(tool_name: str, tool_input: dict, _context):
        violation = _resolve_violation(agent_dir, tool_name, tool_input)
        if violation:
            return PermissionResultDeny(message=violation)
        return PermissionResultAllow()

    return guard


def _confinement_hook(agent_dir: Path):
    """PreToolUse hook: this is what actually gates Read/Write/Edit, since
    allowed_tools shadows can_use_tool for whole-tool allows."""
    agent_dir = agent_dir.resolve()

    async def hook(input_data: dict, _tool_use_id, _context) -> dict:
        violation = _resolve_violation(agent_dir, input_data["tool_name"], input_data["tool_input"])
        if violation:
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": violation,
                }
            }
        return {}

    return hook


class GenerationSession:
    """One conversation with the coding harness: generate, then refine on
    sandbox failures without losing context."""

    def __init__(self, spec: AgentSpec, agent_dir: Path, triggering_request: str, progress=print):
        self.agent_dir = agent_dir
        self.progress = progress
        options = ClaudeAgentOptions(
            cwd=str(agent_dir),
            system_prompt=build_system_prompt(spec, agent_dir, triggering_request),
            allowed_tools=["Read", "Write", "Edit"],
            disallowed_tools=["Bash", "WebSearch", "WebFetch", "Task", "Glob", "Grep"],
            permission_mode="acceptEdits",
            can_use_tool=_confinement_guard(agent_dir),
            hooks={"PreToolUse": [HookMatcher(matcher=None, hooks=[_confinement_hook(agent_dir)])]},
            model=GENERATOR_MODEL,
            max_turns=MAX_TURNS,
        )
        self._client = ClaudeSDKClient(options=options)

    async def __aenter__(self):
        await self._client.connect()
        return self

    async def __aexit__(self, *exc):
        await self._client.disconnect()

    async def generate(self) -> None:
        await self._send("Generate the agent now, following your instructions exactly.")

    async def refine(self, failure_output: str) -> None:
        await self._send(
            "The sandbox ran `pytest test_agent.py` and it failed:\n\n"
            f"{failure_output}\n\n"
            "Fix the implementation (or the tests, if they assert something "
            "wrong) while keeping every constraint. Do not weaken the tests "
            "just to make them pass."
        )

    async def _send(self, prompt: str) -> None:
        await self._client.query(prompt)
        async for message in self._client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, ToolUseBlock):
                        target = next(
                            (block.input.get(k) for k in _PATH_KEYS if block.input.get(k)),
                            "",
                        )
                        self.progress(f"  [generator] {block.name} {target}")
                    elif isinstance(block, TextBlock) and block.text.strip():
                        first_line = block.text.strip().splitlines()[0]
                        self.progress(f"  [generator] {first_line[:100]}")
