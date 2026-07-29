"""Offline tests for the generic credential-acquisition framework: the
planner-eligibility gate, field validation on bind, direct-source tool
dispatch, and the REPL's approval/collection logic. Complements
test_credential_gate.py, which covers the mock_email end-to-end graph path."""

import imaplib

import pytest

from orchestrator import chat, mcp_bindings
from tests.test_gmail_imap_tool import SAMPLE_HEADERS, FakeIMAP


@pytest.fixture(autouse=True)
def isolated_credentials(tmp_path, monkeypatch):
    monkeypatch.setattr(mcp_bindings, "CREDENTIALS_PATH", tmp_path / "credentials.json")


def test_gmail_imap_is_not_planner_eligible():
    """Real Gmail must never be silently offered to automatic generation —
    it's opt-in, not something a casual 'check my email' request should
    start demanding real credentials for."""
    catalog = mcp_bindings.planner_catalog()
    assert "gmail_imap" not in catalog
    assert "mock_email" in catalog
    assert mcp_bindings.KNOWN_SERVICES["gmail_imap"]["planner_eligible"] is False


def test_bind_none_type_ignores_values_and_auto_issues():
    mcp_bindings.bind("u1", "mock_email", "check my email")
    stored = mcp_bindings._load_credential("u1", "mock_email")
    assert stored["token"].startswith("demo-")


def test_bind_app_password_requires_all_fields():
    with pytest.raises(ValueError, match="missing required credential fields"):
        mcp_bindings.bind("u1", "gmail_imap", "check my gmail", values={"email_address": "a@b.com"})


def test_bind_app_password_stores_exact_fields():
    mcp_bindings.bind(
        "u1", "gmail_imap", "check my gmail",
        values={"email_address": "a@b.com", "app_password": "abcd efgh ijkl mnop"},
    )
    stored = mcp_bindings._load_credential("u1", "gmail_imap")
    assert stored == {"email_address": "a@b.com", "app_password": "abcd efgh ijkl mnop"}


def test_credentials_file_is_owner_only(tmp_path):
    mcp_bindings.bind("u1", "mock_email", "check my email")
    mode = mcp_bindings.CREDENTIALS_PATH.stat().st_mode & 0o777
    assert mode == 0o600


def test_bind_then_get_tools_full_round_trip_against_real_gmail_tool(monkeypatch):
    """End-to-end: approve -> bind -> get_tools -> real StructuredTool ->
    .invoke() -> real (mocked) IMAP call. Proves the exact interface the
    generator's contract prompt promises actually works, not just that the
    dispatch function is reachable."""
    monkeypatch.setattr(imaplib, "IMAP4_SSL", FakeIMAP)
    mcp_bindings.bind(
        "u1", "gmail_imap", "check my gmail",
        values={"email_address": "user@gmail.com", "app_password": "correct-app-password"},
    )

    tools = mcp_bindings.get_tools("u1", "gmail_imap")
    names = {t.name for t in tools}
    assert names == {"list_unread", "search_email"}

    list_unread = next(t for t in tools if t.name == "list_unread")
    result = list_unread.invoke({"limit": 5})
    assert "boss@example.com" in result


def test_get_tools_direct_source_dispatches_with_stored_values(monkeypatch):
    mcp_bindings.bind(
        "u1", "gmail_imap", "check my gmail",
        values={"email_address": "a@b.com", "app_password": "secret"},
    )
    captured = {}

    def fake_build_tools(email_address, app_password):
        captured["args"] = (email_address, app_password)
        return ["not-a-real-tool-but-proves-dispatch"]

    monkeypatch.setitem(mcp_bindings.KNOWN_SERVICES["gmail_imap"], "build_tools", fake_build_tools)
    tools = mcp_bindings.get_tools("u1", "gmail_imap")
    assert tools == ["not-a-real-tool-but-proves-dispatch"]
    assert captured["args"] == ("a@b.com", "secret")


def test_repl_oauth2_shows_instructions_and_auto_declines(capsys):
    payload = {
        "agent_id": "x", "service": "some_oauth_service", "scope": "full access",
        "reason": "do the thing", "note": None, "credential_type": "oauth2",
        "fields": [], "acquisition_instructions": ["Register an OAuth client at ..."],
    }
    resume = chat._collect_credential_approval(payload)
    assert resume == {"approved": False}
    assert "Register an OAuth client" in capsys.readouterr().out


def test_repl_none_type_approves_without_prompting_for_fields(monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda _prompt="": "y")
    payload = {
        "agent_id": "x", "service": "mock_email", "scope": "mock inbox",
        "reason": "check my email", "note": None, "credential_type": "none",
        "fields": [], "acquisition_instructions": [],
    }
    resume = chat._collect_credential_approval(payload)
    assert resume == {"approved": True}


def test_repl_collects_fields_hiding_secrets(monkeypatch, capsys):
    inputs = iter(["y", "a@b.com"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(inputs))
    monkeypatch.setattr("getpass.getpass", lambda _prompt="": "hidden-app-password")

    payload = {
        "agent_id": "x", "service": "gmail_imap", "scope": "real gmail",
        "reason": "check my gmail",
        "note": "This connects to your real Gmail account read-only...",
        "credential_type": "app_password",
        "fields": [
            {"name": "email_address", "label": "Your Gmail address", "secret": False},
            {"name": "app_password", "label": "16-character App Password", "secret": True},
        ],
        "acquisition_instructions": ["Go to https://myaccount.google.com/apppasswords"],
    }
    resume = chat._collect_credential_approval(payload)
    assert resume == {
        "approved": True,
        "values": {"email_address": "a@b.com", "app_password": "hidden-app-password"},
    }
    # The secret itself must never be printed to stdout.
    assert "hidden-app-password" not in capsys.readouterr().out


def test_repl_decline_never_prompts_for_fields(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _prompt="": "n")
    calls = []
    monkeypatch.setattr("getpass.getpass", lambda _prompt="": calls.append(1))

    payload = {
        "agent_id": "x", "service": "gmail_imap", "scope": "real gmail",
        "reason": "check my gmail", "note": None, "credential_type": "app_password",
        "fields": [{"name": "app_password", "label": "App Password", "secret": True}],
        "acquisition_instructions": [],
    }
    resume = chat._collect_credential_approval(payload)
    assert resume == {"approved": False}
    assert calls == []
