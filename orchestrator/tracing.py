"""Trace writer. Every node emits a row stamped with the four-part version:
(agent_id, agent_version, prompt_version, manifest_version).

Set up in Phase 0 so every later phase inherits it instead of retrofitting.
"""

import json

from db.conn import get_conn


def write_trace(
    *,
    user_id: str | None,
    node: str,
    event: str,
    agent_id: str | None = None,
    agent_version: int | None = None,
    prompt_version: int | None = None,
    manifest_version: int | None = None,
    payload: dict | None = None,
) -> None:
    get_conn().execute(
        """
        INSERT INTO traces
          (user_id, node, event, agent_id, agent_version, prompt_version,
           manifest_version, payload)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            user_id,
            node,
            event,
            agent_id,
            agent_version,
            prompt_version,
            manifest_version,
            json.dumps(payload) if payload is not None else None,
        ),
    )
