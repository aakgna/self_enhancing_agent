"""Domain scope gate: the check() function in isolation, plus route_node's
wiring of it. Order: registry search + classifier run first; the scope gate
only runs on a miss, right before generation would otherwise be attempted —
an existing, already-in-domain agent match never pays for a redundant scope
judgment. Stickiness bypasses both entirely for an agent mid multi-turn
flow."""

import uuid

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from db import prompt_store
from db.conn import get_conn
from orchestrator import classifier, registry, scope_gate, supervisor
from orchestrator.classifier import Classification
from orchestrator.scope_gate import ScopeVerdict, check


class FakeVerdictLLM:
    def __init__(self, verdict: ScopeVerdict):
        self.verdict = verdict

    def with_structured_output(self, _schema):
        return self

    def invoke(self, _prompt):
        return self.verdict


def test_check_in_scope():
    llm = FakeVerdictLLM(ScopeVerdict(in_scope=True, reason="fine"))
    verdict = check("I'm here for my 2pm appointment", scope_gate.DEFAULT_SCOPE_PROMPT, llm=llm)
    assert verdict.in_scope is True


def test_check_out_of_scope_carries_the_decline_reason():
    llm = FakeVerdictLLM(
        ScopeVerdict(in_scope=False, reason="I can only help with check-in and clinic questions here.")
    )
    verdict = check("check my email for anything urgent", scope_gate.DEFAULT_SCOPE_PROMPT, llm=llm)
    assert verdict.in_scope is False
    assert "check-in" in verdict.reason


@pytest.fixture()
def user_id():
    uid = f"_scope_{uuid.uuid4().hex[:8]}"
    yield uid
    conn = get_conn()
    for table in ("traces", "capability_gaps"):
        conn.execute(f"DELETE FROM {table} WHERE user_id = %s", (uid,))


def _state(user_id, text, **overrides):
    state = {
        "messages": [HumanMessage(content=text)],
        "user_id": user_id,
        "active_agent": None,
        "agent_scratchpad": {},
        "routing_history": [],
    }
    state.update(overrides)
    return state


class ExplodingRegistry:
    def search(self, *a, **k):
        raise AssertionError("registry.search must not be called when an agent is mid-flow")


def test_route_node_declines_out_of_scope_after_a_registry_miss(monkeypatch, user_id):
    """New order: registry search + classifier run first (and miss); only
    then does the scope gate get a say, and it can still decline."""
    prompt_store.init_db()
    monkeypatch.setattr(registry, "search", lambda *a, **k: [])
    monkeypatch.setattr(
        scope_gate,
        "fast_llm",
        lambda: FakeVerdictLLM(ScopeVerdict(in_scope=False, reason="That's outside what I can help with here.")),
    )

    result = supervisor.route_node(_state(user_id, "check my email for anything urgent"))

    assert result["active_agent"] is None
    assert result["routing_history"][-1]["routed_to"] == "declined_out_of_scope"
    assert isinstance(result["messages"][0], AIMessage)
    assert "outside" in result["messages"][0].content

    # Unlike the old order, the gap IS logged here: the registry genuinely
    # missed before the scope gate got a say. This is the new order's
    # trade-off, not a regression.
    gaps = get_conn().execute(
        "SELECT count(*) FROM capability_gaps WHERE user_id = %s", (user_id,)
    ).fetchone()
    assert gaps[0] == 1


def test_route_node_skips_scope_gate_when_an_existing_agent_matches(monkeypatch, user_id):
    """The core promise of the new order: an existing, already-in-domain
    agent match is routed directly — it never pays for a redundant scope
    judgment on every turn."""
    registry.init_registry()
    registry.backfill()
    monkeypatch.setattr(scope_gate, "fast_llm", _exploding_fast_llm)
    monkeypatch.setattr(
        classifier, "classify", lambda request, hits: Classification(decision="general_qa", candidates=hits)
    )

    result = supervisor.route_node(_state(user_id, "what is the capital of France"))

    assert result["active_agent"] == "general_qa"
    assert result["routing_history"][-1]["routed_to"] == "general_qa"


def test_after_route_ends_the_graph_on_decline(user_id):
    state = _state(user_id, "check my email")
    state["active_agent"] = None
    state["routing_history"] = [{"request": "check my email", "routed_to": "declined_out_of_scope"}]
    from langgraph.graph import END

    assert supervisor._after_route(state) == END


def _exploding_fast_llm():
    raise AssertionError("scope_gate must not be called when an agent is mid-flow")


def test_route_node_stickiness_bypasses_scope_and_classifier(monkeypatch, user_id):
    """An agent mid multi-turn flow (awaiting_reply=True) must own the next
    turn outright — a short reply like a spoken DOB must not be reclassified
    or declined as out-of-scope."""
    monkeypatch.setattr(scope_gate, "fast_llm", _exploding_fast_llm)
    monkeypatch.setattr(registry, "search", ExplodingRegistry().search)

    state = _state(
        user_id,
        "03/14/1985",
        active_agent="checkin",
        agent_scratchpad={"checkin": {"awaiting_reply": True, "step": "awaiting_identity"}},
    )

    result = supervisor.route_node(state)

    assert result["active_agent"] == "checkin"
    assert result["routing_history"][-1]["routed_to"] == "checkin"
