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


def purge_agent(user_id: str, agent_id: str, delete_files: bool = True) -> None:
    """Atomically remove everything for one generated agent: registry entry,
    prompt versions, and (optionally) its directory.

    Two callers:
      - route_node's self-healing path, when the registry names an agent
        whose files are already gone (drift — see the module docstring
        context: local generated-agent files have no persistence story
        independent of the registry, so a redeploy/disk reset can produce
        exactly this). delete_files=False there, since there's nothing left
        to delete on disk.
      - maintainer/dev cleanup of test artifacts, replacing ad hoc `rm -rf`
        + hand-written SQL — that two-step manual process is what caused the
        drift in the first place (a directory removed without also removing
        its registry row).

    Not a user-facing "delete agent" feature — the product lifecycle stays
    create-only. This only ever removes agents that are already gone or that
    a maintainer is deliberately cleaning up.
    """
    from orchestrator.agent_loader import AGENTS_ROOT

    conn = get_conn()
    conn.execute(
        "DELETE FROM manifest_embeddings WHERE manifest_id = %s", (f"{user_id}/agent/{agent_id}",)
    )
    conn.execute("DELETE FROM agent_prompts WHERE agent_id = %s AND user_id = %s", (agent_id, user_id))
    conn.execute(
        "DELETE FROM agent_prompt_current WHERE agent_id = %s AND user_id = %s", (agent_id, user_id)
    )
    if delete_files:
        import shutil

        shutil.rmtree(AGENTS_ROOT / user_id / agent_id, ignore_errors=True)


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
