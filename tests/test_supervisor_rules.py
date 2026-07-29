"""The write rule is enforced mechanically, not just by convention."""

import pytest
from langchain_core.messages import AIMessage

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
