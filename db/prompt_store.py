"""Versioned, Postgres-backed agent prompts.

Prompts are dynamic: agents never read their own DEFAULT_PROMPT at runtime —
the supervisor fetches the live prompt here on every call. Version rows are
append-only; rollback just moves the current-version pointer.

Prompt resolution is per-user with a shared fallback: (agent_id, user_id)
first, then (agent_id, "_shared") for the hand-written static agents.
"""

from pathlib import Path

from db.conn import get_conn

SHARED_USER = "_shared"

# (agent_id, user_id) -> (prompt, version). Invalidated on set/rollback.
_cache: dict[tuple[str, str], tuple[str, int]] = {}


def init_db() -> None:
    schema = (Path(__file__).parent / "schema.sql").read_text()
    get_conn().execute(schema)


def _fetch(agent_id: str, user_id: str) -> tuple[str, int] | None:
    row = (
        get_conn()
        .execute(
            """
            SELECT p.prompt, p.version
            FROM agent_prompt_current c
            JOIN agent_prompts p
              ON p.agent_id = c.agent_id
             AND p.user_id = c.user_id
             AND p.version = c.current_version
            WHERE c.agent_id = %s AND c.user_id = %s
            """,
            (agent_id, user_id),
        )
        .fetchone()
    )
    return (row[0], row[1]) if row else None


def get_prompt(agent_id: str, user_id: str) -> tuple[str, int] | None:
    """Return (prompt, version), cache-first, with _shared fallback."""
    for uid in (user_id, SHARED_USER):
        key = (agent_id, uid)
        if key in _cache:
            return _cache[key]
        found = _fetch(agent_id, uid)
        if found:
            _cache[key] = found
            return found
    return None


def set_prompt(agent_id: str, user_id: str, new_prompt: str) -> int:
    """Insert a new version row and move the pointer. Returns the new version."""
    conn = get_conn()
    row = conn.execute(
        "SELECT coalesce(max(version), 0) FROM agent_prompts WHERE agent_id = %s AND user_id = %s",
        (agent_id, user_id),
    ).fetchone()
    version = row[0] + 1
    conn.execute(
        "INSERT INTO agent_prompts (agent_id, user_id, version, prompt) VALUES (%s, %s, %s, %s)",
        (agent_id, user_id, version, new_prompt),
    )
    conn.execute(
        """
        INSERT INTO agent_prompt_current (agent_id, user_id, current_version)
        VALUES (%s, %s, %s)
        ON CONFLICT (agent_id, user_id) DO UPDATE SET current_version = EXCLUDED.current_version
        """,
        (agent_id, user_id, version),
    )
    _cache.pop((agent_id, user_id), None)
    return version


def rollback(agent_id: str, user_id: str, to_version: int) -> None:
    """Instant: only the pointer moves. Raises if the target version is absent."""
    conn = get_conn()
    exists = conn.execute(
        "SELECT 1 FROM agent_prompts WHERE agent_id = %s AND user_id = %s AND version = %s",
        (agent_id, user_id, to_version),
    ).fetchone()
    if not exists:
        raise ValueError(f"no version {to_version} for ({agent_id}, {user_id})")
    conn.execute(
        "UPDATE agent_prompt_current SET current_version = %s WHERE agent_id = %s AND user_id = %s",
        (to_version, agent_id, user_id),
    )
    _cache.pop((agent_id, user_id), None)


def seed_prompt(agent_id: str, user_id: str, default_prompt: str) -> tuple[str, int]:
    """Insert version 1 from the agent's DEFAULT_PROMPT if no prompt exists yet."""
    found = get_prompt(agent_id, user_id)
    if found:
        return found
    set_prompt(agent_id, user_id, default_prompt)
    return default_prompt, 1
