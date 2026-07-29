"""Offline tests for the Agent Maker's deterministic parts: prompt assembly,
write-confinement guard, registrar normalization, and quarantine."""

import asyncio
import json

import pytest

from agent_maker import maker, registrar
from agent_maker.generator import _confinement_guard, build_system_prompt
from agent_maker.planner import AgentSpec
from claude_agent_sdk import PermissionResultAllow, PermissionResultDeny
from db import prompt_store
from db.conn import get_conn


def setup_module():
    prompt_store.init_db()


@pytest.fixture()
def spec():
    return AgentSpec(
        proposed_agent_name="pig_latin_converter",
        purpose="Converts English text to pig latin deterministically.",
        example_requests=["convert this text to pig latin"],
    )


def test_prompt_assembly_fills_every_placeholder(spec, tmp_path):
    prompt = build_system_prompt(spec, tmp_path, "convert this text to pig latin")
    assert "{" not in prompt.replace("{braces}", "") or "state_schema_source" not in prompt
    # The real contract pieces are embedded:
    assert "class AgentState(TypedDict)" in prompt
    assert 'AGENT_ID = "general_qa"' in prompt
    assert 'AGENT_ID = "summarizer"' in prompt
    assert "pig_latin_converter" in prompt
    assert str(tmp_path) in prompt
    assert "convert this text to pig latin" in prompt
    # Maintainer header must not leak into the model prompt.
    assert "versioned in git, not in Postgres" not in prompt


def test_confinement_guard_denies_outside_writes(tmp_path):
    guard = _confinement_guard(tmp_path)

    async def check():
        inside = await guard("Write", {"file_path": str(tmp_path / "agent.py")}, None)
        relative = await guard("Write", {"file_path": "test_agent.py"}, None)
        outside = await guard("Write", {"file_path": "/etc/hosts"}, None)
        traversal = await guard("Edit", {"file_path": "../../orchestrator/state.py"}, None)
        return inside, relative, outside, traversal

    inside, relative, outside, traversal = asyncio.run(check())
    assert isinstance(inside, PermissionResultAllow)
    assert isinstance(relative, PermissionResultAllow)
    assert isinstance(outside, PermissionResultDeny)
    assert isinstance(traversal, PermissionResultDeny)


def test_registrar_normalizes_and_requires_files(spec, tmp_path, monkeypatch):
    upserted = {}
    monkeypatch.setattr(registrar.registry, "upsert", lambda m: upserted.setdefault("m", m))

    with pytest.raises(registrar.GenerationIncomplete):
        registrar.register(spec, tmp_path, "user_x", "convert this")

    (tmp_path / "agent.py").write_text("AGENT_ID='x'\n")
    (tmp_path / "test_agent.py").write_text("def test_a(): pass\n")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "id": "WRONG_ID",
                "user_id": "WRONG_USER",
                "type": "tool",
                "name": "Pig Latin",
                "description": "Converts text to pig latin. Not for translation.",
                "example_requests": ["make this pig latin"],
                "version": 99,
            }
        )
    )
    manifest = registrar.register(spec, tmp_path, "user_x", "convert this")

    # Authoritative fields are system-owned, whatever the generator wrote.
    assert manifest.id == tmp_path.name
    assert manifest.user_id == "user_x"
    assert manifest.type == "agent"
    assert manifest.version == 1
    # Triggering request guaranteed present for retrieval.
    assert "convert this" in manifest.example_requests
    assert upserted["m"].id == manifest.id
    # Normalized manifest written back to disk.
    on_disk = json.loads((tmp_path / "manifest.json").read_text())
    assert on_disk["id"] == tmp_path.name


def test_quarantine_hides_dir_from_loader(tmp_path):
    agent_dir = tmp_path / "bad_agent"
    agent_dir.mkdir()
    (agent_dir / "agent.py").write_text("x = 1\n")
    maker._quarantine(agent_dir)
    assert not agent_dir.exists()
    moved = list((tmp_path / ".failed").iterdir())
    assert len(moved) == 1 and moved[0].name.startswith("bad_agent-")


def test_failure_logging(spec):
    registrar.log_failure("user_y", spec, "convert this", "tests failed after retries", "boom")
    row = get_conn().execute(
        "SELECT agent_id, reason FROM generation_failures WHERE user_id = 'user_y' ORDER BY id DESC LIMIT 1"
    ).fetchone()
    assert row == ("pig_latin_converter", "tests failed after retries")
    get_conn().execute("DELETE FROM generation_failures WHERE user_id = 'user_y'")
