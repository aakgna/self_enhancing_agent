"""Offline tests for agent_maker/fixer.py — the :diagnose auto-fix pipeline.

_attempt_fix (the actual Claude Agent SDK session) needs a live/mocked SDK
connection and has no direct unit test here, matching this repo's existing
precedent for agent_maker/generator.py's GenerationSession and
agent_maker/maker.py's _generate_with_retries (see tests/test_agent_maker.py)
— those are exercised via fakes at the orchestration boundary instead. This
file tests the deterministic pieces (_snapshot/_restore) and apply_fix's own
orchestration (provider fallback, restore-on-failure) against a fake
_attempt_fix, the same pattern used for maker.py's fallback tests.
"""

from agent_maker import fixer
from orchestrator import agent_loader


# ---------------------------------------------------------------------------
# _snapshot / _restore
# ---------------------------------------------------------------------------

def test_snapshot_captures_all_files_except_pycache(tmp_path):
    (tmp_path / "agent.py").write_text("x = 1\n")
    (tmp_path / "manifest.json").write_text("{}")
    pycache = tmp_path / "__pycache__"
    pycache.mkdir()
    (pycache / "agent.cpython-312.pyc").write_bytes(b"\x00\x01")

    snapshot = fixer._snapshot(tmp_path)
    assert snapshot == {"agent.py": "x = 1\n", "manifest.json": "{}"}


def test_restore_reverts_edits_and_removes_new_files(tmp_path):
    (tmp_path / "agent.py").write_text("original\n")
    snapshot = fixer._snapshot(tmp_path)

    (tmp_path / "agent.py").write_text("edited\n")
    (tmp_path / "new_file.py").write_text("new\n")

    fixer._restore(tmp_path, snapshot)

    assert (tmp_path / "agent.py").read_text() == "original\n"
    assert not (tmp_path / "new_file.py").exists()


# ---------------------------------------------------------------------------
# apply_fix orchestration
# ---------------------------------------------------------------------------

class _FakeAttemptFix:
    """Records the use_bedrock value of each call; replays a queue of
    (passed, output, upstream_failed) tuples — mirrors
    tests/test_agent_maker.py's _FakeGenerationSession pattern."""

    def __init__(self, results):
        self.results = list(results)
        self.calls: list[bool] = []

    async def __call__(self, agent_dir, bug_report, progress, use_bedrock):
        self.calls.append(use_bedrock)
        return self.results.pop(0)


def _make_agent_dir(tmp_path, user_id, agent_id, content="AGENT_ID = 'x'\n"):
    d = tmp_path / user_id / agent_id
    d.mkdir(parents=True)
    (d / "agent.py").write_text(content)
    (d / "manifest.json").write_text('{"id": "%s"}' % agent_id)
    return d


def test_apply_fix_no_such_agent_returns_false(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_loader, "AGENTS_ROOT", tmp_path)
    assert fixer.apply_fix("someone", "nonexistent_agent", "bug report") is False


def test_apply_fix_succeeds_on_first_attempt(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_loader, "AGENTS_ROOT", tmp_path)
    monkeypatch.setattr(fixer, "bedrock_configured", lambda: False)
    _make_agent_dir(tmp_path, "u1", "my_agent")

    fake = _FakeAttemptFix([(True, "ok", False)])
    monkeypatch.setattr(fixer, "_attempt_fix", fake)

    assert fixer.apply_fix("u1", "my_agent", "bug report") is True
    assert fake.calls == [False]  # bedrock not configured -> straight to Anthropic


def test_apply_fix_prefers_bedrock_when_configured(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_loader, "AGENTS_ROOT", tmp_path)
    monkeypatch.setattr(fixer, "bedrock_configured", lambda: True)
    _make_agent_dir(tmp_path, "u1", "my_agent")

    fake = _FakeAttemptFix([(True, "ok", False)])
    monkeypatch.setattr(fixer, "_attempt_fix", fake)

    assert fixer.apply_fix("u1", "my_agent", "bug report") is True
    assert fake.calls == [True]


def test_apply_fix_falls_back_to_anthropic_on_bedrock_upstream_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_loader, "AGENTS_ROOT", tmp_path)
    monkeypatch.setattr(fixer, "bedrock_configured", lambda: True)
    _make_agent_dir(tmp_path, "u1", "my_agent")

    fake = _FakeAttemptFix([
        (False, "upstream failed", True),
        (True, "ok", False),
    ])
    monkeypatch.setattr(fixer, "_attempt_fix", fake)

    assert fixer.apply_fix("u1", "my_agent", "bug report") is True
    assert fake.calls == [True, False]


def test_apply_fix_does_not_retry_via_anthropic_on_a_genuine_test_failure(tmp_path, monkeypatch):
    """Only an *upstream API* failure justifies burning a second full
    attempt on the other provider — a real, reproducible test failure
    should not (the bug is presumably still a bug regardless of provider)."""
    monkeypatch.setattr(agent_loader, "AGENTS_ROOT", tmp_path)
    monkeypatch.setattr(fixer, "bedrock_configured", lambda: True)
    _make_agent_dir(tmp_path, "u1", "my_agent")

    fake = _FakeAttemptFix([(False, "tests still fail after retries", False)])
    monkeypatch.setattr(fixer, "_attempt_fix", fake)

    assert fixer.apply_fix("u1", "my_agent", "bug report") is False
    assert fake.calls == [True]


def test_apply_fix_restores_original_files_when_no_fix_passes(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_loader, "AGENTS_ROOT", tmp_path)
    monkeypatch.setattr(fixer, "bedrock_configured", lambda: False)
    agent_dir = _make_agent_dir(tmp_path, "u1", "my_agent", content="ORIGINAL\n")

    async def fake_attempt(agent_dir_arg, bug_report, progress, use_bedrock):
        # Simulate the fix session having written (bad) edits before failing.
        (agent_dir_arg / "agent.py").write_text("BROKEN\n")
        (agent_dir_arg / "stray_new_file.py").write_text("oops\n")
        return False, "still failing", False

    monkeypatch.setattr(fixer, "_attempt_fix", fake_attempt)

    assert fixer.apply_fix("u1", "my_agent", "bug report") is False
    assert (agent_dir / "agent.py").read_text() == "ORIGINAL\n"
    assert not (agent_dir / "stray_new_file.py").exists()
