"""Offline tests for eval/diagnose.py — the transcript bug-diagnosis tool.
Recording-fake-LLM pattern, no network, no real API calls."""

from langchain_core.messages import AIMessage, HumanMessage

from eval import diagnose
from eval.diagnose import Diagnosis, format_report, transcript_from_state
from orchestrator import agent_loader


class FakeDiagnosisLLM:
    """Replays a queue of canned Diagnosis results, one per
    with_structured_output(...).invoke(...) call — lets tests exercise the
    two-pass (transcript-only, then source-grounded) behavior."""

    def __init__(self, results: list[Diagnosis]):
        self._results = list(results)
        self.prompts: list[str] = []

    def with_structured_output(self, schema, method=None):
        assert schema is Diagnosis
        return self

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return self._results.pop(0)


def test_no_bug_found_short_circuits_to_one_call():
    llm = FakeDiagnosisLLM([Diagnosis(bug_found=False, summary="looks fine")])
    result = diagnose.diagnose("you> hi\n[faq] hello!", llm=llm)
    assert result.bug_found is False
    assert len(llm.prompts) == 1


def test_bug_found_without_user_id_stays_single_pass():
    llm = FakeDiagnosisLLM([
        Diagnosis(bug_found=True, summary="wrong slot", likely_agent_id="appointment_booking"),
    ])
    result = diagnose.diagnose("you> book\n[appointment_booking] wrong slot", llm=llm)
    assert result.bug_found is True
    assert len(llm.prompts) == 1  # no user_id -> can't look up source, no second pass


def test_bug_found_with_user_id_and_real_agent_triggers_second_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_loader, "AGENTS_ROOT", tmp_path)
    user_id = "diag_test_user"
    agent_dir = tmp_path / user_id / "appointment_booking"
    agent_dir.mkdir(parents=True)
    (agent_dir / "agent.py").write_text("AGENT_ID = 'appointment_booking'\n")
    (agent_dir / "manifest.json").write_text('{"id": "appointment_booking"}')

    llm = FakeDiagnosisLLM([
        Diagnosis(bug_found=True, summary="wrong slot", likely_agent_id="appointment_booking", confidence="low"),
        Diagnosis(bug_found=True, summary="confirmed: date filter dropped", likely_agent_id="appointment_booking", confidence="high"),
    ])
    result = diagnose.diagnose("you> book\n[appointment_booking] wrong slot", user_id=user_id, llm=llm)

    assert len(llm.prompts) == 2
    assert "AGENT_ID = 'appointment_booking'" in llm.prompts[1]
    assert result.confidence == "high"


def test_bug_found_but_agent_source_not_found_stays_single_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_loader, "AGENTS_ROOT", tmp_path / "nonexistent")
    llm = FakeDiagnosisLLM([
        Diagnosis(bug_found=True, summary="x", likely_agent_id="nonexistent_agent"),
    ])
    result = diagnose.diagnose("transcript", user_id="someone", llm=llm)
    assert len(llm.prompts) == 1
    assert result.likely_agent_id == "nonexistent_agent"


def test_format_report_no_bug():
    assert format_report(Diagnosis(bug_found=False, summary="all good")) == "No bug found. all good"


def test_format_report_bug_found():
    d = Diagnosis(
        bug_found=True,
        summary="wrong slot offered",
        likely_agent_id="appointment_booking",
        root_cause="date filter dropped on miss",
        evidence='"May 27" offered for an August request',
        suggested_fix="don't drop the date filter",
        confidence="high",
    )
    report = format_report(d)
    assert "BUG FOUND (high confidence): wrong slot offered" in report
    assert "Agent: appointment_booking" in report
    assert "Root cause: date filter dropped on miss" in report
    assert "Suggested fix: don't drop the date filter" in report


def test_transcript_from_state_reconstructs_terminal_format():
    state = {
        "messages": [
            HumanMessage(content="can I schedule an appointment?"),
            AIMessage(content="Sure! Let's get started."),
            HumanMessage(content="yes"),
            AIMessage(content="Great, what's your name?"),
        ],
        "routing_history": [
            {"request": "can I schedule an appointment?", "routed_to": "faq"},
            {"request": "yes", "routed_to": "appointment_booking"},
        ],
    }
    text = transcript_from_state(state)
    assert text == (
        "you> can I schedule an appointment?\n"
        "[faq] Sure! Let's get started.\n"
        "you> yes\n"
        "[appointment_booking] Great, what's your name?"
    )


def test_transcript_from_state_handles_declined_turn_with_no_routed_to():
    """A declined-out-of-scope turn still produces an AIMessage but
    routed_to is a sentinel, not a real agent id — must not crash."""
    state = {
        "messages": [
            HumanMessage(content="what's the weather"),
            AIMessage(content="That's outside what I can help with."),
        ],
        "routing_history": [
            {"request": "what's the weather", "routed_to": "declined_out_of_scope"},
        ],
    }
    text = transcript_from_state(state)
    assert "you> what's the weather" in text
    assert "[declined_out_of_scope] That's outside what I can help with." in text
