"""Prompt store round-trip against the local Postgres. Uses throwaway ids and
cleans up after itself."""

import uuid

import pytest

from db import prompt_store
from db.conn import get_conn


@pytest.fixture()
def agent_id():
    aid = f"_test_{uuid.uuid4().hex[:8]}"
    yield aid
    conn = get_conn()
    conn.execute("DELETE FROM agent_prompts WHERE agent_id = %s", (aid,))
    conn.execute("DELETE FROM agent_prompt_current WHERE agent_id = %s", (aid,))
    prompt_store._cache.clear()


def setup_module():
    prompt_store.init_db()


def test_seed_then_get(agent_id):
    prompt, version = prompt_store.seed_prompt(agent_id, "u1", "seed prompt")
    assert (prompt, version) == ("seed prompt", 1)
    assert prompt_store.get_prompt(agent_id, "u1") == ("seed prompt", 1)


def test_set_appends_versions_never_overwrites(agent_id):
    prompt_store.seed_prompt(agent_id, "u1", "v1 text")
    assert prompt_store.set_prompt(agent_id, "u1", "v2 text") == 2
    assert prompt_store.get_prompt(agent_id, "u1") == ("v2 text", 2)
    rows = get_conn().execute(
        "SELECT version, prompt FROM agent_prompts WHERE agent_id = %s ORDER BY version",
        (agent_id,),
    ).fetchall()
    assert rows == [(1, "v1 text"), (2, "v2 text")]


def test_rollback_moves_pointer_only(agent_id):
    prompt_store.seed_prompt(agent_id, "u1", "v1 text")
    prompt_store.set_prompt(agent_id, "u1", "v2 text")
    prompt_store.rollback(agent_id, "u1", 1)
    assert prompt_store.get_prompt(agent_id, "u1") == ("v1 text", 1)
    with pytest.raises(ValueError):
        prompt_store.rollback(agent_id, "u1", 99)


def test_shared_fallback(agent_id):
    prompt_store.seed_prompt(agent_id, prompt_store.SHARED_USER, "shared text")
    assert prompt_store.get_prompt(agent_id, "some_user") == ("shared text", 1)


def test_cache_invalidated_on_set(agent_id):
    prompt_store.seed_prompt(agent_id, "u1", "v1 text")
    prompt_store.get_prompt(agent_id, "u1")  # warm cache
    prompt_store.set_prompt(agent_id, "u1", "v2 text")
    assert prompt_store.get_prompt(agent_id, "u1") == ("v2 text", 2)
