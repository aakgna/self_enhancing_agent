"""Full graph invocation with stubbed LLMs: routing, dispatch, the scratchpad
merge, and version-stamped tracing — no network. The live-model check is the
REPL (see README)."""

import uuid

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from db import prompt_store
from db.conn import get_conn
from orchestrator import router, supervisor
from orchestrator.router import RouteChoice


class FakeAgentLLM:
    def invoke(self, messages):
        return AIMessage(content="stub answer")


class FakeStructured:
    def __init__(self, agent_id):
        self.agent_id = agent_id

    def invoke(self, _prompt):
        return RouteChoice(agent_id=self.agent_id)


class FakeRouterLLM:
    def __init__(self, agent_id):
        self.agent_id = agent_id

    def with_structured_output(self, _schema):
        return FakeStructured(self.agent_id)


@pytest.fixture()
def user_id():
    uid = f"_e2e_{uuid.uuid4().hex[:8]}"
    yield uid
    get_conn().execute("DELETE FROM traces WHERE user_id = %s", (uid,))


def test_graph_end_to_end(monkeypatch, user_id):
    prompt_store.init_db()
    monkeypatch.setattr(supervisor, "agent_llm", lambda: FakeAgentLLM())
    monkeypatch.setattr(router, "fast_llm", lambda: FakeRouterLLM("summarizer"))

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
