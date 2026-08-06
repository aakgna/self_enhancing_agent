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

import csv
import inspect
import os
from pathlib import Path

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    HookMatcher,
    PermissionResultAllow,
    PermissionResultDeny,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
)

from agent_maker.planner import AgentSpec
from orchestrator.mcp_bindings import KNOWN_SERVICES
from tools.shared.ehr_csv import store as ehr_store

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_PATH = Path(__file__).parent / "prompts" / "generator_system_prompt.md"
SHARED_AGENTS = REPO_ROOT / "agents" / "_shared"

GENERATOR_MODEL = "claude-opus-4-8"
MAX_TURNS = 50

_PATH_KEYS = ("file_path", "path", "notebook_path")


def _bedrock_options() -> tuple[str, dict[str, str]]:
    """(model, env) for routing the coding harness's own subprocess through
    Amazon Bedrock instead of the direct Anthropic API. CLAUDE_CODE_USE_BEDROCK
    is the Claude Code CLI's own documented Bedrock switch (the Claude Agent
    SDK just spawns that CLI); AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY are
    picked up from the inherited environment by boto3's normal credential
    chain, same as orchestrator/llm.py's Bedrock leg — nothing extra to pass
    for those. Read at call time, not module import time, so this stays
    monkeypatch-able in tests the same way orchestrator/llm.py's env reads are."""
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
    return os.environ["BEDROCK_MODEL_ID"], {"CLAUDE_CODE_USE_BEDROCK": "1", "AWS_REGION": region}


def _is_upstream_api_failure(result: ResultMessage | None) -> bool:
    """True when the most recently completed turn's underlying model API call
    itself failed (exhausted credits, rate limit, outage — the Claude Agent
    SDK's equivalent of orchestrator/llm.py's anthropic.AnthropicError catch).
    This can't be caught as a Python exception: the CLI subprocess absorbs
    the failure and reports it structurally via ResultMessage.is_error +
    api_error_status instead of raising, so a generation attempt that never
    wrote any files looks identical to a genuine code bug unless this is
    checked explicitly. api_error_status is only ever set alongside
    is_error=True (see claude_agent_sdk.types.ResultMessage), so checking
    both is belt-and-suspenders, not redundant."""
    return bool(result and result.is_error and result.api_error_status is not None)

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


# Ground truth for the local "hospital DB": introspected from the real
# store.py functions/source and the real CSV headers/first row at assembly
# time, the same never-drift principle as the worked examples above.
# Regression grounding for a live incident: two generated agents
# (appointment_lookup, appointment_scheduler) each invented their own
# incompatible synthetic patient/appointment/slot data instead of using this
# store — because (a) nothing in this prompt told them the store existed at
# all, and (b) the generation sandbox couldn't import it anyway (fixed
# separately in sandbox_runner.py's PYTHONPATH). This section is always
# included, unconditionally, unlike the credentialed-tools section above —
# every spec should see it, since the model (not this code) decides whether
# a given capability happens to touch the EHR domain.
_EHR_STORE_SECTION = '''\
## The local hospital "DB" — use it, never invent your own data

For ANY capability touching patients, appointments, providers, insurance, or
scheduling: this repo has a real local data store — plain CSV files acting
as the hospital's system of record. Use it. NEVER invent your own hardcoded
or synthetic patient/appointment/slot/provider data, and NEVER build a fake
in-memory dataset "for now" — that has happened before and produced agents
that could never find a real patient or actually book anything.

This is local computation (plain file I/O, no network, no external
service): for any spec that only uses this data, `requires_credentials`
must be `false` and `mcp_service` must be `null` — it is NOT a credentialed
tool and does not use the `tools=` parameter/contract above.

### First choice: the existing store module

Module: `{module}` — `from tools.shared.ehr_csv import store`

Functions already available — prefer these whenever one fits:

{functions}

CSV schemas backing those functions (column names, with one real sample row
each so you know the actual formats — YYYY-MM-DD dates, HH:MM 24-hour times,
etc.):

{schemas}

book_slot()/cancel_appointment() write straight back to the CSV files
immediately on every call — this *is* the database, not a cache in front of
one, so a booking made in one turn is visible to every later lookup,
including by a completely different agent, since all agents share these
same files.

Worked pattern for agent.py:

```python
from tools.shared.ehr_csv import store

def run(state, system_prompt, llm):
    patient = store.find_patient(first_name, last_name, dob)
    if patient is None:
        ...
    slots = store.list_available_slots(provider_id=None, date=target_date)
    ...
```

### If none of the above fits: write your own Create/Read/Update function

You may NOT edit store.py itself (additive-only generation: you may only
create files inside your own agent directory). If the capability genuinely
needs a new operation on this data — a new field to look up, a new kind of
record to create or update — implement it as a plain Python module inside
your own agent directory, operating directly on the same CSV files at
`{data_dir}`, using EXACTLY this read/write pattern (copied verbatim from
store.py's own implementation, so it stays consistent with every other
piece of code — including store.py itself — touching these files). Import
`_DATA_DIR` from store.py rather than recomputing the path yourself, so
there is exactly one place that path is ever defined:

```python
import csv
from tools.shared.ehr_csv.store import _DATA_DIR

{rw_pattern}
```

Rules for any such function you write:
  - **Read fresh, every call — never cache a table across calls.** Something
    else (store.py, or another agent's own CRUD function) may have written
    to the same file since you last read it; a cached snapshot goes stale
    silently. Always `_table(...)` at the top of the function, act on the
    result, and `_write_table(...)` before returning.
  - **Create and Update only — never delete a row.** To retire a record
    (cancel, deactivate, etc.), flip a status/flag field on the existing
    row (see cancel_appointment above: `status="cancelled"`, not a removed
    row) and write the table back. The row stays in the record.
  - Match the exact column names shown in the CSV schemas above — a
    mismatched fieldname either crashes csv.DictWriter or silently drops
    data, neither of which is acceptable here.

Worked pattern for test_agent.py — do NOT mock this data; it is plain local
file I/O with real (if fictional) sample data, so call it directly and
assert against known sample rows shown above:

```python
from tools.shared.ehr_csv import store

def test_finds_a_real_sample_patient():
    patient = store.find_patient("Sarah", "Johnson", "1985-03-12")
    assert patient is not None
    assert patient["patient_id"] == "P001"
```
'''

_EHR_STORE_MODULE = "tools.shared.ehr_csv.store"
_EHR_PUBLIC_FUNCTIONS = (
    "find_patient",
    "find_appointment_today",
    "get_insurance",
    "get_provider",
    "list_available_slots",
    "book_slot",
    "cancel_appointment",
)
_EHR_CSV_FILES = ("patients.csv", "appointments.csv", "insurance.csv", "providers.csv", "slots.csv")
_EHR_RW_PATTERN_FUNCTIONS = ("_table", "_write_table")


def _ehr_store_section() -> str:
    function_docs = []
    for name in _EHR_PUBLIC_FUNCTIONS:
        fn = getattr(ehr_store, name)
        summary = (inspect.getdoc(fn) or "").splitlines()[0] if inspect.getdoc(fn) else ""
        function_docs.append(f"  - `{name}{inspect.signature(fn)}` — {summary}")

    schema_lines = []
    for filename in _EHR_CSV_FILES:
        with (ehr_store._DATA_DIR / filename).open(newline="") as f:
            rows = list(csv.reader(f))
        columns, sample_row = rows[0], (rows[1] if len(rows) > 1 else [])
        schema_lines.append(f"  - `{filename}`: {', '.join(columns)}")
        if sample_row:
            schema_lines.append(f"    e.g. {dict(zip(columns, sample_row))}")

    rw_pattern = "\n\n".join(inspect.getsource(getattr(ehr_store, name)) for name in _EHR_RW_PATTERN_FUNCTIONS)

    return _EHR_STORE_SECTION.format(
        module=_EHR_STORE_MODULE,
        data_dir=str(ehr_store._DATA_DIR),
        functions="\n".join(function_docs),
        schemas="\n".join(schema_lines),
        rw_pattern=rw_pattern,
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
        "{checkin_agent_py}": example("checkin", "agent.py"),
        "{checkin_manifest}": example("checkin", "manifest.json"),
        "{checkin_test_py}": example("checkin", "test_agent.py"),
        "{agent_dir}": str(agent_dir),
        "{spec_json}": spec.model_dump_json(indent=2),
        "{triggering_request}": triggering_request,
        "{credentialed_tools_section}": _credentialed_tools_section(spec),
        "{ehr_store_section}": _ehr_store_section(),
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
    sandbox failures without losing context.

    use_bedrock=True routes this session's subprocess through Amazon Bedrock
    instead of the direct Anthropic API (see _bedrock_options) — a completely
    separate mechanism from orchestrator/llm.py's fallback chain, since this
    harness talks to the Claude Agent SDK, not a LangChain chat model."""

    def __init__(
        self,
        spec: AgentSpec,
        agent_dir: Path,
        triggering_request: str,
        progress=print,
        use_bedrock: bool = False,
    ):
        self.agent_dir = agent_dir
        self.progress = progress
        self.last_result: ResultMessage | None = None
        model, env = (_bedrock_options() if use_bedrock else (GENERATOR_MODEL, {}))
        options = ClaudeAgentOptions(
            cwd=str(agent_dir),
            system_prompt=build_system_prompt(spec, agent_dir, triggering_request),
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
            elif isinstance(message, ResultMessage):
                # receive_response() always ends on a ResultMessage (see the
                # SDK's own docs on the method) — this is the structural
                # signal upstream_api_failed() reads, not a text guess.
                self.last_result = message
