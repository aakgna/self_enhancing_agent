"""Tests for the general_qa agent.

Tests use a recording fake LLM, so they run with no network and no
credentials — exactly how they will run inside the generation sandbox.
They are written to catch a lazy or incorrect implementation, not to
trivially pass: they check the write rule, input immutability, and that the
live prompt actually reaches the model.
"""

import copy

from langchain_core.messages import AIMessage, HumanMessage

from agent import AGENT_ID, run


class FakeLLM:
    """Records every .invoke() call; returns a fixed reply."""

    def __init__(self, reply: str = "fake reply"):
        self.reply = reply
        self.calls: list[list] = []

    def invoke(self, messages):
        self.calls.append(list(messages))
        return AIMessage(content=self.reply)


def make_state(**overrides) -> dict:
    state = {
        "messages": [HumanMessage(content="What is a monad?")],
        "user_id": "test_user",
        "active_agent": AGENT_ID,
        "agent_scratchpad": {},
        "routing_history": [],
    }
    state.update(overrides)
    return state


def test_returns_one_ai_message():
    update = run(make_state(), "system prompt", FakeLLM("an answer"))
    assert [type(m) for m in update["messages"]] == [AIMessage]
    assert update["messages"][0].content == "an answer"


def test_write_rule_only_allowed_keys_and_own_scratchpad():
    update = run(make_state(), "system prompt", FakeLLM())
    assert set(update) <= {"messages", "agent_scratchpad"}
    assert set(update["agent_scratchpad"]) == {AGENT_ID}


def test_live_prompt_reaches_the_model_first():
    llm = FakeLLM()
    run(make_state(), "LIVE PROMPT v7", llm)
    sent = llm.calls[0]
    assert sent[0].content == "LIVE PROMPT v7"
    assert sent[1].content == "What is a monad?"


def test_does_not_mutate_input_state():
    state = make_state(agent_scratchpad={"other_agent": {"k": 1}})
    snapshot = copy.deepcopy(state)
    run(state, "system prompt", FakeLLM())
    assert state == snapshot


def test_does_not_touch_other_agents_scratchpad():
    state = make_state(agent_scratchpad={"other_agent": {"k": 1}})
    update = run(state, "system prompt", FakeLLM())
    assert "other_agent" not in update["agent_scratchpad"]


def test_turn_counter_accumulates_from_own_scratchpad():
    state = make_state(agent_scratchpad={AGENT_ID: {"turns_handled": 4}})
    update = run(state, "system prompt", FakeLLM())
    assert update["agent_scratchpad"][AGENT_ID]["turns_handled"] == 5
