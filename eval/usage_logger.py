"""Usage logging: (request, agent_used, outcome) per turn, auto-derived from
real traffic — no manual test authoring. Outcome is explicit feedback when the
user gives it (REPL :bad), otherwise the weak "no immediate correction" proxy
(assumed_ok).
"""

from db.conn import get_conn


def log_turn(user_id: str, request: str, agent_used: str) -> None:
    get_conn().execute(
        "INSERT INTO usage_log (user_id, request, agent_used) VALUES (%s, %s, %s)",
        (user_id, request, agent_used),
    )


def mark_last_corrected(user_id: str) -> None:
    """Explicit negative feedback on the most recent turn."""
    get_conn().execute(
        """
        UPDATE usage_log SET outcome = 'corrected'
        WHERE id = (SELECT max(id) FROM usage_log WHERE user_id = %s)
        """,
        (user_id,),
    )


def recent_successful(user_id: str, limit: int = 30) -> list[tuple[str, str]]:
    """The replay window: (request, agent_used) pairs that went unchallenged."""
    rows = get_conn().execute(
        """
        SELECT request, agent_used FROM usage_log
        WHERE user_id = %s AND outcome = 'assumed_ok'
        ORDER BY id DESC LIMIT %s
        """,
        (user_id, limit),
    ).fetchall()
    return [(r[0], r[1]) for r in rows]
