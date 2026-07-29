"""The write rule is enforced mechanically, not just by convention."""

import json
import shutil
import uuid

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from db import prompt_store
from db.conn import get_conn
from orchestrator import supervisor
from orchestrator.agent_loader import AGENTS_ROOT
from orchestrator.supervisor import ScratchpadViolation, validate_update


def test_clean_update_passes():
    validate_update(
        "agent_a",
        {"messages": [AIMessage(content="ok")], "agent_scratchpad": {"agent_a": {"k": 1}}},
    )


def test_foreign_scratchpad_key_rejected():
    with pytest.raises(ScratchpadViolation):
        validate_update("agent_a", {"agent_scratchpad": {"agent_b": {"k": 1}}})


def test_disallowed_state_key_rejected():
    with pytest.raises(ScratchpadViolation):
        validate_update("agent_a", {"routing_history": []})


def test_non_message_in_messages_rejected():
    with pytest.raises(ScratchpadViolation):
        validate_update("agent_a", {"messages": ["raw string"]})


AGENT_PY = '''
from langchain_core.messages import AIMessage
AGENT_ID = "isolated_agent"
AGENT_VERSION = 1
DEFAULT_PROMPT = "seed prompt for isolated_agent"

def run(state, system_prompt, llm):
    return {"messages": [AIMessage(content=system_prompt)], "agent_scratchpad": {}}
'''


def _plant_agent(user_id: str) -> None:
    agent_dir = AGENTS_ROOT / user_id / "isolated_agent"
    agent_dir.mkdir(parents=True)
    (agent_dir / "agent.py").write_text(AGENT_PY)
    (agent_dir / "manifest.json").write_text(
        json.dumps(
            {
                "id": "isolated_agent",
                "user_id": user_id,
                "type": "agent",
                "name": "Isolated",
                "description": "test fixture",
                "example_requests": ["hi"],
                "version": 1,
            }
        )
    )


def test_generated_agent_prompts_are_scoped_per_owning_user(monkeypatch):
    """Two different users generating an agent with the same id must not
    collide into one global prompt — that would violate per-user isolation.
    Regression test for execute_node hardcoding prompt_store.SHARED_USER."""
    monkeypatch.setattr(supervisor, "agent_llm", lambda: object())  # fixture agent never calls it
    user_a, user_b = f"_iso_a_{uuid.uuid4().hex[:6]}", f"_iso_b_{uuid.uuid4().hex[:6]}"
    prompt_store.init_db()
    _plant_agent(user_a)
    _plant_agent(user_b)
    try:
        state_a = {
            "messages": [HumanMessage(content="hi")],
            "user_id": user_a,
            "active_agent": "isolated_agent",
            "agent_scratchpad": {},
            "routing_history": [],
        }
        state_b = {**state_a, "user_id": user_b}

        supervisor.execute_node(state_a)
        # A different owning user's same-named agent must seed its OWN
        # version 1, not read user_a's.
        version_a = get_conn().execute(
            "SELECT current_version FROM agent_prompt_current WHERE agent_id = 'isolated_agent' AND user_id = %s",
            (user_a,),
        ).fetchone()
        version_b_before = get_conn().execute(
            "SELECT 1 FROM agent_prompt_current WHERE agent_id = 'isolated_agent' AND user_id = %s",
            (user_b,),
        ).fetchone()
        assert version_a == (1,)
        assert version_b_before is None  # not seeded by user_a's run

        # Editing user_a's prompt must not affect user_b.
        prompt_store.set_prompt("isolated_agent", user_a, "user_a's private edit")
        supervisor.execute_node(state_b)
        prompt_seen_by_b = get_conn().execute(
            "SELECT prompt FROM agent_prompts WHERE agent_id = 'isolated_agent' AND user_id = %s AND version = 1",
            (user_b,),
        ).fetchone()
        assert prompt_seen_by_b == ("seed prompt for isolated_agent",)  # its own seed, not user_a's edit
    finally:
        shutil.rmtree(AGENTS_ROOT / user_a, ignore_errors=True)
        shutil.rmtree(AGENTS_ROOT / user_b, ignore_errors=True)
        conn = get_conn()
        conn.execute(
            "DELETE FROM agent_prompts WHERE agent_id = 'isolated_agent' AND user_id IN (%s, %s)",
            (user_a, user_b),
        )
        conn.execute(
            "DELETE FROM agent_prompt_current WHERE agent_id = 'isolated_agent' AND user_id IN (%s, %s)",
            (user_a, user_b),
        )
        prompt_store._cache.clear()
