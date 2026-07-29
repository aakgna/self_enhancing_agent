"""Full graph invocation with stubbed LLMs: routing, dispatch, the scratchpad
merge, and version-stamped tracing — no network. The live-model check is the
REPL (see README)."""

import uuid

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from db import prompt_store
from db.conn import get_conn
from orchestrator import classifier, registry, supervisor
from orchestrator.classifier import ClassifierVerdict


class FakeAgentLLM:
    def invoke(self, messages):
        return AIMessage(content="stub answer")


class FakeClassifierLLM:
    def __init__(self, verdict: ClassifierVerdict):
        self.verdict = verdict

    def with_structured_output(self, _schema):
        return self

    def invoke(self, _prompt):
        return self.verdict


@pytest.fixture()
def user_id():
    uid = f"_e2e_{uuid.uuid4().hex[:8]}"
    yield uid
    conn = get_conn()
    conn.execute("DELETE FROM traces WHERE user_id = %s", (uid,))
    conn.execute("DELETE FROM capability_gaps WHERE user_id = %s", (uid,))


def test_graph_end_to_end(monkeypatch, user_id):
    prompt_store.init_db()
    registry.init_registry()
    registry.backfill()
    monkeypatch.setattr(supervisor, "agent_llm", lambda: FakeAgentLLM())
    monkeypatch.setattr(
        classifier,
        "fast_llm",
        lambda: FakeClassifierLLM(
            ClassifierVerdict(decision="route_to_existing", agent_id="summarizer")
        ),
    )

    graph = supervisor.build_graph()
    result = graph.invoke(
        {
            "messages": [HumanMessage(content="Summarize: revenue grew 12%.")],
            "user_id": user_id,
            "active_agent": None,
            "agent_scratchpad": {"general_qa": {"turns_handled": 3}},
            "routing_history": [],
        }
    )

    assert result["active_agent"] == "summarizer"
    assert result["messages"][-1].content == "stub answer"
    assert result["routing_history"][-1]["routed_to"] == "summarizer"
    # Dispatcher merged the scratchpad: prior agent's key survives.
    assert result["agent_scratchpad"]["general_qa"] == {"turns_handled": 3}
    assert "summarizer" in result["agent_scratchpad"]

    rows = get_conn().execute(
        """
        SELECT node, event, agent_id, agent_version, prompt_version, manifest_version
        FROM traces WHERE user_id = %s ORDER BY id
        """,
        (user_id,),
    ).fetchall()
    assert [(r[0], r[1]) for r in rows] == [("route", "routed"), ("execute", "agent_ran")]
    execute_row = rows[1]
    # The four-part stamp is fully populated on the execute trace.
    assert execute_row[2] == "summarizer"
    assert execute_row[3] == 1  # agent_version
    assert execute_row[4] >= 1  # prompt_version
    assert execute_row[5] == 1  # manifest_version


def test_no_match_path_logs_capability_gap(monkeypatch, user_id):
    prompt_store.init_db()
    registry.init_registry()
    registry.backfill()
    monkeypatch.setattr(
        classifier,
        "fast_llm",
        lambda: FakeClassifierLLM(ClassifierVerdict(decision="no_match_found")),
    )

    graph = supervisor.build_graph()
    result = graph.invoke(
        {
            "messages": [HumanMessage(content="check my email for urgent messages")],
            "user_id": user_id,
            "active_agent": None,
            "agent_scratchpad": {},
            "routing_history": [],
        }
    )

    assert result["active_agent"] is None
    assert "don't have an agent" in result["messages"][-1].content
    assert result["routing_history"][-1]["routed_to"] == "no_match_found"

    gaps = get_conn().execute(
        "SELECT request, rejected_matches FROM capability_gaps WHERE user_id = %s",
        (user_id,),
    ).fetchall()
    assert len(gaps) == 1
    assert gaps[0][0] == "check my email for urgent messages"
