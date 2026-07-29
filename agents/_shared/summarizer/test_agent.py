"""Tests for the summarizer agent. No network, no credentials — the same
recording-fake-LLM pattern used across all agent tests (see general_qa)."""

import copy

from langchain_core.messages import AIMessage, HumanMessage

from agent import AGENT_ID, run


class FakeLLM:
    def __init__(self, reply: str = "a summary"):
        self.reply = reply
        self.calls: list[list] = []

    def invoke(self, messages):
        self.calls.append(list(messages))
        return AIMessage(content=self.reply)


SOURCE = "The quarterly report shows revenue grew 12% while costs fell 3%."


def make_state(**overrides) -> dict:
    state = {
        "messages": [HumanMessage(content=SOURCE)],
        "user_id": "test_user",
        "active_agent": AGENT_ID,
        "agent_scratchpad": {},
        "routing_history": [],
    }
    state.update(overrides)
    return state


def test_source_text_is_sent_to_the_model():
    llm = FakeLLM()
    run(make_state(), "system prompt", llm)
    sent = llm.calls[0]
    assert sent[0].content == "system prompt"
    assert SOURCE in sent[1].content


def test_uses_latest_human_message_not_earlier_ones():
    state = make_state(
        messages=[
            HumanMessage(content="old text to ignore"),
            AIMessage(content="earlier reply"),
            HumanMessage(content=SOURCE),
        ]
    )
    llm = FakeLLM()
    run(state, "system prompt", llm)
    assert SOURCE in llm.calls[0][1].content
    assert "old text to ignore" not in llm.calls[0][1].content


def test_write_rule_and_scratchpad_stats():
    update = run(make_state(), "system prompt", FakeLLM("short"))
    assert set(update) <= {"messages", "agent_scratchpad"}
    assert set(update["agent_scratchpad"]) == {AGENT_ID}
    stats = update["agent_scratchpad"][AGENT_ID]
    assert stats["source_chars"] == len(SOURCE)
    assert stats["summary_chars"] == len("short")


def test_does_not_mutate_input_state():
    state = make_state(agent_scratchpad={"general_qa": {"turns_handled": 2}})
    snapshot = copy.deepcopy(state)
    run(state, "system prompt", FakeLLM())
    assert state == snapshot
