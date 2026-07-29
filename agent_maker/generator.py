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
    PermissionResultAllow,
    PermissionResultDeny,
    TextBlock,
    ToolUseBlock,
)

from agent_maker.planner import AgentSpec

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_PATH = Path(__file__).parent / "prompts" / "generator_system_prompt.md"
SHARED_AGENTS = REPO_ROOT / "agents" / "_shared"

GENERATOR_MODEL = "claude-opus-4-8"
MAX_TURNS = 50

_PATH_KEYS = ("file_path", "path", "notebook_path")


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
    }
    prompt = template
    for placeholder, value in replacements.items():
        prompt = prompt.replace(placeholder, value)
    return prompt


def _confinement_guard(agent_dir: Path):
    agent_dir = agent_dir.resolve()

    async def guard(tool_name: str, tool_input: dict, _context):
        for key in _PATH_KEYS:
            raw = tool_input.get(key)
            if raw is None:
                continue
            resolved = (agent_dir / raw).resolve() if not Path(raw).is_absolute() else Path(raw).resolve()
            if not resolved.is_relative_to(agent_dir):
                return PermissionResultDeny(
                    message=f"{tool_name} on {resolved} is outside the agent directory "
                    f"{agent_dir}; you may only touch files inside it."
                )
        return PermissionResultAllow()

    return guard


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
