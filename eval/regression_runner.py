"""Background regression testing: async, silent, non-blocking.

Triggered only on new-agent registration (cost control), and only after the
agent is already live — registration and availability never wait on this.
Replays the user's recent successful requests through the router as it exists
NOW (with the new agent in the registry); any request whose routing changed
away from its previously-successful agent flags the new agent's manifest.

Flags are visibility only — nothing is ever auto-removed (create-only
lifecycle).
"""

import threading

from eval import usage_logger
from orchestrator import classifier, registry
from orchestrator.tracing import write_trace

REPLAY_WINDOW = 30


def on_agent_registered(user_id: str, new_agent_id: str) -> threading.Thread:
    """Fire-and-forget. Returns the thread so callers/tests may join it."""
    thread = threading.Thread(
        target=_replay, args=(user_id, new_agent_id), daemon=True, name="regression-runner"
    )
    thread.start()
    return thread


def _replay(user_id: str, new_agent_id: str, classify_fn=None) -> None:
    classify_fn = classify_fn or (
        lambda request, hits: classifier.classify(request, hits).decision
    )
    regressions = []
    for request, previous_agent in usage_logger.recent_successful(user_id, REPLAY_WINDOW):
        if previous_agent == new_agent_id:
            continue
        hits = registry.search(request, user_id, top_k=5, type_filter="agent")
        decision = classify_fn(request, hits)
        if decision != previous_agent:
            regressions.append(
                {"request": request, "was": previous_agent, "now": decision}
            )

    if not regressions:
        write_trace(
            user_id=user_id, node="regression", event="clean",
            agent_id=new_agent_id, payload={"replayed": REPLAY_WINDOW},
        )
        return

    new_agent_key = f"{user_id}/agent/{new_agent_id}"
    registry.set_flagged(new_agent_key, True)
    write_trace(
        user_id=user_id,
        node="regression",
        event="flagged_for_review",
        agent_id=new_agent_id,
        payload={"regressions": regressions},
    )
