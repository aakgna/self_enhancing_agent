"""Offline tests for the Agent Maker's deterministic parts: prompt assembly,
write-confinement guard, registrar normalization, quarantine, and the
credentialed-tools contract (both the prompt grounding and the system-
authored contract check)."""

import asyncio
import json

import pytest

from agent_maker import contract_check, maker, registrar
from agent_maker.generator import _confinement_guard, _confinement_hook, build_system_prompt
from agent_maker.planner import AgentSpec
from claude_agent_sdk import PermissionResultAllow, PermissionResultDeny
from db import prompt_store
from db.conn import get_conn


def setup_module():
    prompt_store.init_db()


@pytest.fixture()
def spec():
    return AgentSpec(
        proposed_agent_name="pig_latin_converter",
        purpose="Converts English text to pig latin deterministically.",
        example_requests=["convert this text to pig latin"],
    )


def test_prompt_assembly_fills_every_placeholder(spec, tmp_path):
    prompt = build_system_prompt(spec, tmp_path, "convert this text to pig latin")
    assert "{" not in prompt.replace("{braces}", "") or "state_schema_source" not in prompt
    # The real contract pieces are embedded:
    assert "class AgentState(TypedDict)" in prompt
    assert 'AGENT_ID = "general_qa"' in prompt
    assert 'AGENT_ID = "summarizer"' in prompt
    assert "pig_latin_converter" in prompt
    assert str(tmp_path) in prompt
    assert "convert this text to pig latin" in prompt
    # Maintainer header must not leak into the model prompt.
    assert "versioned in git, not in Postgres" not in prompt
    # No credentials in this spec: the generator must be told explicitly not
    # to accept a tools param, not left to guess — and no real service's
    # tool names should leak in (those only belong to a credentialed spec).
    assert "does not require credentials" in prompt
    assert "list_unread" not in prompt
    assert "search_email" not in prompt


def test_prompt_assembly_grounds_the_real_tools_contract(tmp_path):
    """This is the regression test for the bug: a live-generated agent called
    `tools.list_recent_emails(count)` against a `list`, because the prompt
    never said what `tools` actually is. It must now state the exact
    StructuredTool interface and the real tool names for the bound service —
    not leave the generator to invent an interface."""
    spec = AgentSpec(
        proposed_agent_name="recent_email_summarizer",
        purpose="Summarizes recent emails.",
        requires_credentials=True,
        mcp_service="mock_email",
        example_requests=["check my email"],
    )
    prompt = build_system_prompt(spec, tmp_path, "check my gmail")

    assert "StructuredTool" in prompt
    assert ".invoke(dict) -> str" in prompt
    # The real tool names from the catalog, not invented ones:
    assert "list_unread" in prompt
    assert "search_email" in prompt
    # Explicit warning against exactly the failure mode that happened live:
    assert "not attribute-access the list" in prompt or "do not guess a method name" in prompt


def test_confinement_guard_denies_outside_writes(tmp_path):
    guard = _confinement_guard(tmp_path)

    async def check():
        inside = await guard("Write", {"file_path": str(tmp_path / "agent.py")}, None)
        relative = await guard("Write", {"file_path": "test_agent.py"}, None)
        outside = await guard("Write", {"file_path": "/etc/hosts"}, None)
        traversal = await guard("Edit", {"file_path": "../../orchestrator/state.py"}, None)
        return inside, relative, outside, traversal

    inside, relative, outside, traversal = asyncio.run(check())
    assert isinstance(inside, PermissionResultAllow)
    assert isinstance(relative, PermissionResultAllow)
    assert isinstance(outside, PermissionResultDeny)
    assert isinstance(traversal, PermissionResultDeny)


def test_confinement_hook_is_the_real_enforcement_path(tmp_path):
    """allowed_tools=[...] auto-approves Read/Write/Edit before can_use_tool
    runs (CanUseToolShadowedWarning) — this PreToolUse hook is what actually
    gates those calls. Must produce the exact hookSpecificOutput shape the
    SDK expects for a deny/allow decision."""
    hook = _confinement_hook(tmp_path)

    async def check():
        allowed = await hook(
            {"tool_name": "Write", "tool_input": {"file_path": "agent.py"}}, "id1", None
        )
        denied = await hook(
            {"tool_name": "Write", "tool_input": {"file_path": "/etc/hosts"}}, "id2", None
        )
        return allowed, denied

    allowed, denied = asyncio.run(check())
    assert allowed == {}
    assert denied["hookSpecificOutput"]["hookEventName"] == "PreToolUse"
    assert denied["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "/etc/hosts" in denied["hookSpecificOutput"]["permissionDecisionReason"]


def test_registrar_normalizes_and_requires_files(spec, tmp_path, monkeypatch):
    upserted = {}
    monkeypatch.setattr(registrar.registry, "upsert", lambda m: upserted.setdefault("m", m))

    with pytest.raises(registrar.GenerationIncomplete):
        registrar.register(spec, tmp_path, "user_x", "convert this")

    (tmp_path / "agent.py").write_text("AGENT_ID='x'\n")
    (tmp_path / "test_agent.py").write_text("def test_a(): pass\n")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "id": "WRONG_ID",
                "user_id": "WRONG_USER",
                "type": "tool",
                "name": "Pig Latin",
                "description": "Converts text to pig latin. Not for translation.",
                "example_requests": ["make this pig latin"],
                "version": 99,
            }
        )
    )
    manifest = registrar.register(spec, tmp_path, "user_x", "convert this")

    # Authoritative fields are system-owned, whatever the generator wrote.
    assert manifest.id == tmp_path.name
    assert manifest.user_id == "user_x"
    assert manifest.type == "agent"
    assert manifest.version == 1
    # Triggering request guaranteed present for retrieval.
    assert "convert this" in manifest.example_requests
    assert upserted["m"].id == manifest.id
    # Normalized manifest written back to disk.
    on_disk = json.loads((tmp_path / "manifest.json").read_text())
    assert on_disk["id"] == tmp_path.name


CORRECT_INTERFACE_AGENT = '''
from langchain_core.messages import AIMessage

AGENT_ID = "recent_email_summarizer"
AGENT_VERSION = 1
DEFAULT_PROMPT = "irrelevant"


def run(state, system_prompt, llm, tools=None):
    if not tools:
        return {
            "messages": [AIMessage(content="not connected")],
            "agent_scratchpad": {AGENT_ID: {"connected": False}},
        }
    tool = next(t for t in tools if t.name == "list_unread")
    result = tool.invoke({"limit": 5})
    return {
        "messages": [AIMessage(content=result)],
        "agent_scratchpad": {AGENT_ID: {"connected": True}},
    }
'''

# The exact bug that shipped: treats `tools` as an object with a method
# per capability instead of a list of StructuredTool objects.
WRONG_INTERFACE_AGENT = '''
from langchain_core.messages import AIMessage

AGENT_ID = "recent_email_summarizer"
AGENT_VERSION = 1
DEFAULT_PROMPT = "irrelevant"


def run(state, system_prompt, llm, tools=None):
    if not tools:
        return {
            "messages": [AIMessage(content="not connected")],
            "agent_scratchpad": {AGENT_ID: {"connected": False}},
        }
    fetched = tools.list_unread(5)  # AttributeError: 'list' has no such attribute
    return {
        "messages": [AIMessage(content=str(fetched))],
        "agent_scratchpad": {AGENT_ID: {"connected": True}},
    }
'''


def test_contract_check_catches_the_shipped_bug(tmp_path):
    (tmp_path / "agent.py").write_text(WRONG_INTERFACE_AGENT)
    passed, output = contract_check.run_contract_check(tmp_path, "mock_email", "check my email")
    assert not passed
    assert "list" in output and "attribute" in output


def test_contract_check_passes_the_real_interface(tmp_path):
    (tmp_path / "agent.py").write_text(CORRECT_INTERFACE_AGENT)
    passed, output = contract_check.run_contract_check(tmp_path, "mock_email", "check my email")
    assert passed, output


def test_contract_check_applies_only_to_known_credentialed_services():
    assert contract_check.applies("mock_email") is True
    assert contract_check.applies(None) is False
    assert contract_check.applies("some_unknown_service") is False


def test_quarantine_hides_dir_from_loader(tmp_path):
    agent_dir = tmp_path / "bad_agent"
    agent_dir.mkdir()
    (agent_dir / "agent.py").write_text("x = 1\n")
    maker._quarantine(agent_dir)
    assert not agent_dir.exists()
    moved = list((tmp_path / ".failed").iterdir())
    assert len(moved) == 1 and moved[0].name.startswith("bad_agent-")


def test_failure_logging(spec):
    registrar.log_failure("user_y", spec, "convert this", "tests failed after retries", "boom")
    row = get_conn().execute(
        "SELECT agent_id, reason FROM generation_failures WHERE user_id = 'user_y' ORDER BY id DESC LIMIT 1"
    ).fetchone()
    assert row == ("pig_latin_converter", "tests failed after retries")
    get_conn().execute("DELETE FROM generation_failures WHERE user_id = 'user_y'")
