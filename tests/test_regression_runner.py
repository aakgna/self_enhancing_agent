"""Phase 5 exit criteria, offline: a newly registered agent that changes
routing for previously-successful requests gets flagged — without the
creation flow ever blocking on the replay."""

import time
import uuid

import pytest

from db import prompt_store
from db.conn import get_conn
from eval import regression_runner, usage_logger
from orchestrator import registry
from orchestrator.manifest import Manifest


@pytest.fixture()
def user_id():
    uid = f"_reg_{uuid.uuid4().hex[:8]}"
    prompt_store.init_db()
    registry.init_registry()
    registry.backfill()
    yield uid
    conn = get_conn()
    for table in ("usage_log", "traces"):
        conn.execute(f"DELETE FROM {table} WHERE user_id = %s", (uid,))
    conn.execute("DELETE FROM manifest_embeddings WHERE user_id = %s", (uid,))


def seed_history(uid):
    for request in [
        "summarize this article about markets",
        "tldr these release notes",
        "give me the key points of this memo",
    ]:
        usage_logger.log_turn(uid, request, "summarizer")
    usage_logger.log_turn(uid, "what is a red-black tree", "general_qa")


def register_overlapping_agent(uid) -> Manifest:
    """Deliberately overlaps the summarizer's turf — the doc's test case."""
    manifest = Manifest(
        id="text_condenser",
        user_id=uid,
        type="agent",
        name="Text Condenser",
        description="Condenses and summarizes any text into key points.",
        example_requests=["summarize this text", "condense these notes"],
    )
    registry.upsert(manifest)
    return manifest


def test_routing_change_flags_new_agent(user_id):
    seed_history(user_id)
    manifest = register_overlapping_agent(user_id)

    # Deterministic replacement for the Haiku stage: the overlapping agent now
    # wins every summarize-shaped request.
    def classify_fn(request, hits):
        return "text_condenser" if "summar" in request or "tldr" in request or "key points" in request else "general_qa"

    regression_runner._replay(user_id, "text_condenser", classify_fn=classify_fn)

    hits = registry.search("summarize this text", user_id)
    flagged = next(h for h in hits if h.manifest.id == "text_condenser")
    assert flagged.manifest.flagged_for_review is True

    row = get_conn().execute(
        "SELECT payload FROM traces WHERE user_id = %s AND event = 'flagged_for_review'",
        (user_id,),
    ).fetchone()
    assert row is not None
    regressions = row[0]["regressions"]
    assert len(regressions) == 3  # the three summarizer requests, not the QA one
    assert all(r["was"] == "summarizer" for r in regressions)


def test_no_change_no_flag(user_id):
    seed_history(user_id)
    register_overlapping_agent(user_id)

    regression_runner._replay(
        user_id, "text_condenser", classify_fn=lambda request, hits: (
            "general_qa" if "red-black" in request else "summarizer"
        )
    )

    hits = registry.search("summarize this text", user_id)
    condenser = next(h for h in hits if h.manifest.id == "text_condenser")
    assert condenser.manifest.flagged_for_review is False
    clean = get_conn().execute(
        "SELECT 1 FROM traces WHERE user_id = %s AND event = 'clean'", (user_id,)
    ).fetchone()
    assert clean is not None


def test_trigger_is_non_blocking(user_id, monkeypatch):
    """Registration returns immediately; the replay happens on its own thread."""
    started = time.monotonic()

    def slow_replay(uid, agent_id, classify_fn=None):
        time.sleep(0.5)

    monkeypatch.setattr(regression_runner, "_replay", slow_replay)
    thread = regression_runner.on_agent_registered(user_id, "whatever")
    elapsed = time.monotonic() - started
    assert elapsed < 0.2  # did not wait for the replay
    thread.join(timeout=2)
    assert not thread.is_alive()
