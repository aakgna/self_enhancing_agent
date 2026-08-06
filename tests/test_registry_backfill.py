"""registry.backfill(): skips dot-prefixed agent directories, matching
agent_loader._scan()'s existing quarantine convention. Regression test for a
live bug: purging an agent (registrar.purge_agent(delete_files=False)) only
removes its registry row — the directory is still on disk in a normal
location, so the next backfill() call (which several other scripts/tests
call routinely) silently re-registered it, undoing the purge."""

import json
import shutil
import uuid

import pytest

from db import prompt_store
from db.conn import get_conn
from orchestrator import registry
from orchestrator.agent_loader import AGENTS_ROOT

MANIFEST = {
    "id": "placeholder",
    "user_id": "placeholder",
    "type": "agent",
    "name": "Placeholder",
    "description": "test fixture",
    "example_requests": ["do the thing"],
    "version": 1,
}


@pytest.fixture()
def user_id():
    uid = f"_backfill_{uuid.uuid4().hex[:8]}"
    yield uid
    shutil.rmtree(AGENTS_ROOT / uid, ignore_errors=True)
    get_conn().execute("DELETE FROM manifest_embeddings WHERE manifest_id LIKE %s", (f"{uid}/%",))


def _plant(user_id: str, dirname: str, agent_id: str) -> None:
    agent_dir = AGENTS_ROOT / user_id / dirname
    agent_dir.mkdir(parents=True)
    (agent_dir / "manifest.json").write_text(json.dumps({**MANIFEST, "id": agent_id, "user_id": user_id}))


def test_backfill_skips_dot_prefixed_agent_directories(user_id):
    prompt_store.init_db()
    registry.init_registry()
    _plant(user_id, ".quarantined_agent", "quarantined_agent")

    loaded = registry.backfill()

    assert f"{user_id}/agent/quarantined_agent" not in loaded
    row = get_conn().execute(
        "SELECT count(*) FROM manifest_embeddings WHERE manifest_id = %s",
        (f"{user_id}/agent/quarantined_agent",),
    ).fetchone()
    assert row[0] == 0


def test_backfill_still_loads_a_normal_agent_directory(user_id):
    prompt_store.init_db()
    registry.init_registry()
    _plant(user_id, "normal_agent", "normal_agent")

    loaded = registry.backfill()

    assert f"{user_id}/agent/normal_agent" in loaded
