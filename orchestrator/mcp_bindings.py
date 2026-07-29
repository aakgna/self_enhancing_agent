"""Credentialed MCP capability binding (Phase 4).

Nothing binds without explicit human approval — the supervisor pauses with a
LangGraph interrupt before any real connection, and every approval lands in
the append-only capability_ledger. Binding is hot: a fresh
MultiServerMCPClient session per call means new bindings work without any
restart.

Tools handed to agents are wrapped sync (agents' run() is synchronous).
"""

import asyncio
import json
import secrets
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.tools import StructuredTool

from db.conn import get_conn

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
CREDENTIALS_PATH = REPO_ROOT / "local_cache" / "credentials.json"

# The catalog of credentialed services the planner may propose. "tools" is the
# ground truth for what a bound service actually exposes — the generator's
# system prompt is built from this, not guessed, so a generated agent calls
# real tool names instead of inventing an interface. Keep this in sync with
# the corresponding MCP server's @mcp.tool() definitions.
#
# In production each entry would carry a real OAuth flow (see
# is_real_service / real_service_note below for how that's surfaced to the
# user); the mock issues a demo token with no external auth.
KNOWN_SERVICES = {
    "mock_email": {
        "scope": "read-only access to your (mock) email inbox",
        "server": str(REPO_ROOT / "tools" / "shared" / "mock_email_mcp" / "server.py"),
        "credential_env": "MOCK_EMAIL_TOKEN",
        "real_service_note": (
            "This is a local mock with sample data, not your real Gmail account. "
            "It does not ask you to sign in, and it cannot read or send real "
            "email. Wiring up real Gmail would require registering a Google "
            "Cloud OAuth client and completing a real consent flow — ask if "
            "you want that built."
        ),
        "tools": [
            {
                "name": "list_unread",
                "description": "List unread emails in the user's inbox, most recent first.",
                "args": {"limit": "int, default 10 — max number of emails to return"},
            },
            {
                "name": "search_email",
                "description": "Search all emails (subject and sender) for a query string.",
                "args": {"query": "str — search string"},
            },
        ],
    },
}


def is_bound(user_id: str, service: str) -> bool:
    row = get_conn().execute(
        "SELECT 1 FROM capability_ledger WHERE user_id = %s AND service = %s AND action = 'bound' LIMIT 1",
        (user_id, service),
    ).fetchone()
    return row is not None


def bind(user_id: str, service: str, triggering_request: str) -> None:
    """Record an approved binding and issue/store the credential (demo)."""
    if service not in KNOWN_SERVICES:
        raise ValueError(f"unknown service: {service}")
    _store_credential(user_id, service, f"demo-{secrets.token_hex(8)}")
    get_conn().execute(
        """
        INSERT INTO capability_ledger
          (user_id, service, action, triggering_request, approved_at, config)
        VALUES (%s, %s, 'bound', %s, %s, %s)
        """,
        (
            user_id,
            service,
            triggering_request,
            datetime.now(timezone.utc),
            json.dumps({"server": KNOWN_SERVICES[service]["server"]}),
        ),
    )


def get_tools(user_id: str, service: str) -> list[StructuredTool]:
    """Live tools for a bound service, wrapped for synchronous agents."""
    from langchain_mcp_adapters.client import MultiServerMCPClient

    entry = KNOWN_SERVICES[service]
    client = MultiServerMCPClient(
        {
            service: {
                "command": str(PYTHON),
                "args": [entry["server"]],
                "env": {entry["credential_env"]: _load_credential(user_id, service)},
                "transport": "stdio",
            }
        }
    )
    async_tools = asyncio.run(client.get_tools())
    return [_sync_wrap(tool) for tool in async_tools]


def _sync_wrap(async_tool) -> StructuredTool:
    def call(**kwargs):
        return asyncio.run(async_tool.ainvoke(kwargs))

    return StructuredTool(
        name=async_tool.name,
        description=async_tool.description,
        args_schema=async_tool.args_schema,
        func=call,
    )


def _store_credential(user_id: str, service: str, token: str) -> None:
    CREDENTIALS_PATH.parent.mkdir(exist_ok=True)
    store = (
        json.loads(CREDENTIALS_PATH.read_text()) if CREDENTIALS_PATH.exists() else {}
    )
    store[f"{user_id}/{service}"] = token
    CREDENTIALS_PATH.write_text(json.dumps(store, indent=2))


def _load_credential(user_id: str, service: str) -> str:
    store = json.loads(CREDENTIALS_PATH.read_text())
    return store[f"{user_id}/{service}"]
