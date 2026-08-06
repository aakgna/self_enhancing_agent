"""Tests for the checkin agent: the full state machine, turn by turn, against
a stub LLM. Tests use a recording fake LLM, so they run with no network and
no credentials — exactly how they will run inside the generation sandbox."""

import copy

from langchain_core.messages import AIMessage, HumanMessage

from agent import AGENT_ID, IdentityFields, _classify_yes_no, run


class _Extractor:
    def __init__(self, identity):
        self.identity = identity

    def invoke(self, _prompt):
        return self.identity


class FakeLLM:
    """Records every .invoke() call; returns a fixed reply. with_structured_output
    returns the canned IdentityFields given at construction time."""

    def __init__(self, reply: str = "fake reply", identity: IdentityFields | None = None):
        self.reply = reply
        self.identity = identity
        self.calls: list[list] = []

    def invoke(self, messages):
        self.calls.append(list(messages))
        return AIMessage(content=self.reply)

    def with_structured_output(self, _schema, method=None):
        return _Extractor(self.identity)


def make_state(**overrides) -> dict:
    state = {
        "messages": [HumanMessage(content="I'm here for my appointment")],
        "user_id": "test_user",
        "active_agent": AGENT_ID,
        "agent_scratchpad": {},
        "routing_history": [],
    }
    state.update(overrides)
    return state


def test_start_asks_for_identity_and_sets_awaiting_reply():
    update = run(make_state(), "system prompt", FakeLLM("what's your name and DOB?"))
    scratch = update["agent_scratchpad"][AGENT_ID]
    assert scratch["step"] == "awaiting_identity"
    assert scratch["awaiting_reply"] is True
    assert update["messages"][0].content == "what's your name and DOB?"


def test_identity_found_with_appointment_advances_to_ack():
    state = make_state(
        messages=[HumanMessage(content="Sarah Johnson, 1985-03-12")],
        agent_scratchpad={AGENT_ID: {"step": "awaiting_identity", "awaiting_reply": True, "attempts": 0}},
    )
    llm = FakeLLM("confirming your 9am", IdentityFields(first_name="Sarah", last_name="Johnson", dob="1985-03-12"))
    update = run(state, "system prompt", llm)
    scratch = update["agent_scratchpad"][AGENT_ID]
    assert scratch["step"] == "awaiting_appointment_ack"
    assert scratch["awaiting_reply"] is True
    assert scratch["patient_id"] == "P001"
    assert scratch["appointment_id"] == "A001"


def test_identity_not_found_retries_once_before_escalating():
    state = make_state(
        messages=[HumanMessage(content="Nobody Real, 1900-01-01")],
        agent_scratchpad={AGENT_ID: {"step": "awaiting_identity", "awaiting_reply": True, "attempts": 0}},
    )
    llm = FakeLLM("couldn't find you", IdentityFields(first_name="Nobody", last_name="Real", dob="1900-01-01"))
    update = run(state, "system prompt", llm)
    scratch = update["agent_scratchpad"][AGENT_ID]
    assert scratch["step"] == "awaiting_identity"
    assert scratch["awaiting_reply"] is True
    assert scratch["attempts"] == 1

    # Second miss escalates instead of retrying forever.
    state2 = make_state(
        messages=[HumanMessage(content="Nobody Real, 1900-01-01")],
        agent_scratchpad={AGENT_ID: scratch},
    )
    update2 = run(state2, "system prompt", llm)
    scratch2 = update2["agent_scratchpad"][AGENT_ID]
    assert scratch2["step"] == "escalated"
    assert scratch2["awaiting_reply"] is False


def test_identity_found_with_no_appointment_today():
    state = make_state(
        messages=[HumanMessage(content="Emily Davis, 2001-09-18")],
        agent_scratchpad={AGENT_ID: {"step": "awaiting_identity", "awaiting_reply": True, "attempts": 0}},
    )
    llm = FakeLLM("no appointment today", IdentityFields(first_name="Emily", last_name="Davis", dob="2001-09-18"))
    update = run(state, "system prompt", llm)
    scratch = update["agent_scratchpad"][AGENT_ID]
    assert scratch["step"] == "no_appointment"
    assert scratch["awaiting_reply"] is False


def test_appointment_ack_yes_moves_to_consent():
    state = make_state(
        messages=[HumanMessage(content="yes that's right")],
        agent_scratchpad={AGENT_ID: {"step": "awaiting_appointment_ack", "awaiting_reply": True, "patient_id": "P001"}},
    )
    update = run(state, "system prompt", FakeLLM("do you consent?"))
    scratch = update["agent_scratchpad"][AGENT_ID]
    assert scratch["step"] == "awaiting_consent"
    assert scratch["awaiting_reply"] is True


def test_appointment_ack_no_escalates():
    state = make_state(
        messages=[HumanMessage(content="no that's not right")],
        agent_scratchpad={AGENT_ID: {"step": "awaiting_appointment_ack", "awaiting_reply": True}},
    )
    update = run(state, "system prompt", FakeLLM("please see a staff member"))
    scratch = update["agent_scratchpad"][AGENT_ID]
    assert scratch["step"] == "escalated"
    assert scratch["awaiting_reply"] is False


def test_consent_yes_moves_to_forms():
    state = make_state(
        messages=[HumanMessage(content="yes")],
        agent_scratchpad={AGENT_ID: {"step": "awaiting_consent", "awaiting_reply": True}},
    )
    update = run(state, "system prompt", FakeLLM("forms done?"))
    scratch = update["agent_scratchpad"][AGENT_ID]
    assert scratch["step"] == "awaiting_forms"
    assert scratch["awaiting_reply"] is True


def test_consent_no_escalates():
    state = make_state(
        messages=[HumanMessage(content="no")],
        agent_scratchpad={AGENT_ID: {"step": "awaiting_consent", "awaiting_reply": True}},
    )
    update = run(state, "system prompt", FakeLLM("staff will speak with you"))
    scratch = update["agent_scratchpad"][AGENT_ID]
    assert scratch["step"] == "escalated"
    assert scratch["awaiting_reply"] is False


def test_forms_yes_completes():
    state = make_state(
        messages=[HumanMessage(content="yes all done")],
        agent_scratchpad={AGENT_ID: {"step": "awaiting_forms", "awaiting_reply": True}},
    )
    update = run(state, "system prompt", FakeLLM("you're checked in"))
    scratch = update["agent_scratchpad"][AGENT_ID]
    assert scratch["step"] == "complete"
    assert scratch["awaiting_reply"] is False
    assert scratch["forms"] == "completed"


def test_forms_no_completes_with_pending():
    state = make_state(
        messages=[HumanMessage(content="not yet")],
        agent_scratchpad={AGENT_ID: {"step": "awaiting_forms", "awaiting_reply": True}},
    )
    update = run(state, "system prompt", FakeLLM("staff will bring you the forms"))
    scratch = update["agent_scratchpad"][AGENT_ID]
    assert scratch["step"] == "complete"
    assert scratch["forms"] == "pending"


def test_write_rule_only_allowed_keys_and_own_scratchpad():
    update = run(make_state(), "system prompt", FakeLLM())
    assert set(update) <= {"messages", "agent_scratchpad"}
    assert set(update["agent_scratchpad"]) == {AGENT_ID}


def test_does_not_mutate_input_state():
    state = make_state(agent_scratchpad={"other_agent": {"k": 1}})
    snapshot = copy.deepcopy(state)
    run(state, "system prompt", FakeLLM())
    assert state == snapshot


def test_does_not_touch_other_agents_scratchpad():
    state = make_state(agent_scratchpad={"other_agent": {"k": 1}})
    update = run(state, "system prompt", FakeLLM())
    assert "other_agent" not in update["agent_scratchpad"]


def test_classify_yes_no_affirmative_and_negative_and_ambiguous():
    assert _classify_yes_no("yes") == "yes"
    assert _classify_yes_no("Yeah sounds good.") == "yes"
    assert _classify_yes_no("no thanks") == "no"
    assert _classify_yes_no("nope") == "no"
    assert _classify_yes_no("maybe later today") is None
