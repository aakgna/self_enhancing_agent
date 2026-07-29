"""Registration: the only way a generated agent becomes callable.

On pass: normalize + validate the generated manifest, upsert into the
registry — the agent is immediately routable (discovery and loading are
dynamic; nothing restarts). On fail: log to generation_failures and never
register.
"""

import json
from pathlib import Path

from agent_maker.planner import AgentSpec
from db.conn import get_conn
from orchestrator import registry
from orchestrator.manifest import Manifest
from orchestrator.tracing import write_trace


class GenerationIncomplete(RuntimeError):
    pass


REQUIRED_FILES = ("agent.py", "test_agent.py", "manifest.json")


def register(spec: AgentSpec, agent_dir: Path, user_id: str, triggering_request: str) -> Manifest:
    missing = [f for f in REQUIRED_FILES if not (agent_dir / f).exists()]
    if missing:
        raise GenerationIncomplete(f"generator did not produce: {missing}")

    data = json.loads((agent_dir / "manifest.json").read_text())
    # Authoritative fields come from the system, not the generator.
    data.update(
        id=agent_dir.name,
        user_id=user_id,
        type="agent",
        version=1,
        flagged_for_review=False,
        requires_credentials=spec.requires_credentials,
        mcp_service=spec.mcp_service,
    )
    manifest = Manifest.model_validate(data)
    if triggering_request not in manifest.example_requests:
        manifest.example_requests.insert(0, triggering_request)
    (agent_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2))

    registry.upsert(manifest)
    write_trace(
        user_id=user_id,
        node="registrar",
        event="agent_registered",
        agent_id=manifest.id,
        agent_version=1,
        manifest_version=manifest.version,
        payload={"requires_credentials": manifest.requires_credentials},
    )
    return manifest


def log_failure(
    user_id: str, spec: AgentSpec, request: str, reason: str, output: str | None
) -> None:
    get_conn().execute(
        """
        INSERT INTO generation_failures (user_id, agent_id, request, spec, reason, output)
        VALUES (%s, %s, %s, %s, %s, %s)
        """,
        (user_id, spec.proposed_agent_name, request, spec.model_dump_json(), reason, output),
    )
    write_trace(
        user_id=user_id,
        node="registrar",
        event="generation_failed",
        agent_id=spec.proposed_agent_name,
        payload={"reason": reason},
    )
