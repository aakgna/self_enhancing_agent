"""Phase 4 end-to-end, fully offline: a credentialed agent triggers the
approval interrupt; decline leaves nothing bound; approve binds, appends to
the capability ledger, and the agent talks to the real mock MCP server over
stdio in the same session."""

import json
import shutil
import uuid

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from db import prompt_store
from db.conn import get_conn
from orchestrator import classifier, mcp_bindings, registry, scope_gate, supervisor
from orchestrator.agent_loader import AGENTS_ROOT
from orchestrator.classifier import ClassifierVerdict
from orchestrator.scope_gate import ScopeVerdict


@pytest.fixture(autouse=True)
def isolated_credentials(tmp_path, monkeypatch):
    # Without this, every test run writes real entries into the same
    # local_cache/credentials.json a live REPL session reads from — this was
    # a real bug (found live): stale test-written entries accumulated there
    # and, combined with an old storage-format entry, crashed a real session.
    monkeypatch.setattr(mcp_bindings, "CREDENTIALS_PATH", tmp_path / "credentials.json")

AGENT_PY = '''
from langchain_core.messages import AIMessage

AGENT_ID = "mail_reader"
AGENT_VERSION = 1
DEFAULT_PROMPT = "You read the user's inbox via the provided tools."


def run(state, system_prompt, llm, tools=None):
    if not tools:
        return {
            "messages": [AIMessage(content="Email is not connected yet.")],
            "agent_scratchpad": {AGENT_ID: {"connected": False}},
        }
    list_unread = next(t for t in tools if t.name == "list_unread")
    inbox = list_unread.invoke({"limit": 5})
    return {
        "messages": [AIMessage(content=f"Unread mail:\\n{inbox}")],
        "agent_scratchpad": {AGENT_ID: {"connected": True}},
    }
'''

MANIFEST = {
    "id": "mail_reader",
    "user_id": "",  # filled per-test
    "type": "agent",
    "name": "Mail Reader",
    "description": "Reads the user's email inbox. Requires email credentials.",
    "example_requests": ["check my email"],
    "version": 1,
    "requires_credentials": True,
    "mcp_service": "mock_email",
}


class FakeClassifierLLM:
    def with_structured_output(self, _schema):
        return self

    def invoke(self, _prompt):
        return ClassifierVerdict(decision="route_to_existing", agent_id="mail_reader")


class FakeScopeLLM:
    def with_structured_output(self, _schema):
        return self

    def invoke(self, _prompt):
        return ScopeVerdict(in_scope=True, reason="")


@pytest.fixture()
def cred_user():
    uid = f"_cred_{uuid.uuid4().hex[:8]}"
    agent_dir = AGENTS_ROOT / uid / "mail_reader"
    agent_dir.mkdir(parents=True)
    (agent_dir / "agent.py").write_text(AGENT_PY)
    (agent_dir / "manifest.json").write_text(json.dumps({**MANIFEST, "user_id": uid}))
    prompt_store.init_db()
    registry.init_registry()
    from orchestrator.manifest import Manifest

    manifest = Manifest.model_validate({**MANIFEST, "user_id": uid})
    registry.upsert(manifest)
    yield uid
    shutil.rmtree(AGENTS_ROOT / uid)
    conn = get_conn()
    conn.execute("DELETE FROM manifest_embeddings WHERE manifest_id = %s", (manifest.key,))
    for table in ("traces", "capability_ledger", "capability_gaps", "usage_log"):
        conn.execute(f"DELETE FROM {table} WHERE user_id = %s", (uid,))
    conn.execute("DELETE FROM agent_prompts WHERE agent_id = 'mail_reader'")
    conn.execute("DELETE FROM agent_prompt_current WHERE agent_id = 'mail_reader'")
    prompt_store._cache.clear()


def make_graph(monkeypatch):
    monkeypatch.setattr(classifier, "fast_llm", lambda: FakeClassifierLLM())
    monkeypatch.setattr(scope_gate, "fast_llm", lambda: FakeScopeLLM())
    monkeypatch.setattr(supervisor, "agent_llm", lambda: object())  # agent never uses it
    return supervisor.build_graph(checkpointer=InMemorySaver())


def initial_state(uid):
    return {
        "messages": [HumanMessage(content="check my email for anything urgent")],
        "user_id": uid,
        "active_agent": None,
        "agent_scratchpad": {},
        "routing_history": [],
    }


def test_decline_binds_nothing(monkeypatch, cred_user):
    graph = make_graph(monkeypatch)
    config = {"configurable": {"thread_id": f"{cred_user}-decline"}}

    result = graph.invoke(initial_state(cred_user), config)
    assert "__interrupt__" in result
    payload = result["__interrupt__"][0].value
    assert payload["service"] == "mock_email"
    assert payload["reason"] == "check my email for anything urgent"
    assert "scope" in payload

    result = graph.invoke(Command(resume={"approved": False}), config)
    assert "not connected" in result["messages"][-1].content
    rows = get_conn().execute(
        "SELECT count(*) FROM capability_ledger WHERE user_id = %s", (cred_user,)
    ).fetchone()
    assert rows[0] == 0


def test_approve_binds_and_reads_real_mock_inbox(monkeypatch, cred_user):
    graph = make_graph(monkeypatch)
    config = {"configurable": {"thread_id": f"{cred_user}-approve"}}

    result = graph.invoke(initial_state(cred_user), config)
    assert "__interrupt__" in result

    # Approve: binds, spawns the real stdio MCP server, agent reads the inbox.
    result = graph.invoke(Command(resume={"approved": True}), config)
    reply = result["messages"][-1]
    assert isinstance(reply, AIMessage)
    assert "boss@example.com" in reply.content  # canned data via real MCP roundtrip

    ledger = get_conn().execute(
        """SELECT service, action, triggering_request, approved_at
           FROM capability_ledger WHERE user_id = %s""",
        (cred_user,),
    ).fetchall()
    assert len(ledger) == 1
    service, action, triggering_request, approved_at = ledger[0]
    assert (service, action) == ("mock_email", "bound")
    assert triggering_request == "check my email for anything urgent"
    assert approved_at is not None

    # Second request on the same binding: no interrupt this time (hot, no restart).
    config2 = {"configurable": {"thread_id": f"{cred_user}-second"}}
    result2 = graph.invoke(initial_state(cred_user), config2)
    assert "__interrupt__" not in result2
    assert "boss@example.com" in result2["messages"][-1].content


def test_stale_credential_self_heals_instead_of_crashing(monkeypatch, cred_user):
    """Regression test for a live crash: a binding approved under an older
    storage format (a bare string) left the ledger saying 'bound' while the
    stored credential was unusable by current code. The graph must not
    crash — it should detect this, forget the stale binding, and fall
    through to a fresh approval request exactly as if never bound."""
    import json

    from db.conn import get_conn as _get_conn

    # Simulate the old format directly (bypasses bind(), which now always
    # writes the current dict shape) plus a pre-existing "bound" ledger row.
    mcp_bindings.CREDENTIALS_PATH.parent.mkdir(exist_ok=True)
    mcp_bindings.CREDENTIALS_PATH.write_text(
        json.dumps({f"{cred_user}/mock_email": "demo-old-format-token"})
    )
    _get_conn().execute(
        """
        INSERT INTO capability_ledger (user_id, service, action, triggering_request, approved_at)
        VALUES (%s, 'mock_email', 'bound', 'an earlier request', now())
        """,
        (cred_user,),
    )

    graph = make_graph(monkeypatch)
    config = {"configurable": {"thread_id": f"{cred_user}-stale"}}

    # No crash: self-heals into a normal fresh-approval interrupt.
    result = graph.invoke(initial_state(cred_user), config)
    assert "__interrupt__" in result
    payload = result["__interrupt__"][0].value
    assert payload["service"] == "mock_email"

    heal_trace = _get_conn().execute(
        "SELECT payload FROM traces WHERE user_id = %s AND event = 'credential_reset'",
        (cred_user,),
    ).fetchone()
    assert heal_trace is not None
    assert heal_trace[0]["service"] == "mock_email"

    # Approving from here works exactly like a first-time binding.
    result = graph.invoke(Command(resume={"approved": True}), config)
    assert "boss@example.com" in result["messages"][-1].content
