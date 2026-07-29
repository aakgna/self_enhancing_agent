"""Credentialed capability binding (Phase 4).

Nothing binds without explicit human approval — the supervisor pauses with a
LangGraph interrupt before any real connection, shows exactly what's needed
and how to obtain it, and every approval lands in the append-only
capability_ledger. Binding is hot: no restart is needed for a new binding to
take effect.

This is a generic credential-acquisition framework, not one hardcoded email
integration — any future credentialed service declares a `credential_type`
and the fields it needs, and the approval flow (supervisor.py + chat.py)
collects them uniformly. Two tool sources are supported:

  - "mcp": a stdio MCP server (see mock_email below)
  - "direct": a plain Python function building LangChain tools directly,
    for real integrations that don't warrant standing up an MCP server
    (see gmail_imap below)

Tools handed to agents are always sync (agents' run() is synchronous).
"""

import asyncio
import json
import secrets
import stat
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.tools import StructuredTool

from db.conn import get_conn
from tools.shared.gmail_imap import tool as gmail_imap_tool

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
CREDENTIALS_PATH = REPO_ROOT / "local_cache" / "credentials.json"


def _gmail_tools(email_address: str, app_password: str) -> list[StructuredTool]:
    def list_unread(limit: int = 10) -> str:
        """List unread emails in the inbox, most recent first."""
        return gmail_imap_tool.list_unread(email_address, app_password, limit)

    def search_email(query: str) -> str:
        """Search all emails (subject and sender) for a query string."""
        return gmail_imap_tool.search_email(email_address, app_password, query)

    return [
        StructuredTool.from_function(list_unread, name="list_unread"),
        StructuredTool.from_function(search_email, name="search_email"),
    ]


# The catalog of credentialed services the planner may propose. "tools" is the
# ground truth for what a bound service actually exposes — the generator's
# system prompt is built from this, not guessed, so a generated agent calls
# real tool names instead of inventing an interface. Keep this in sync with
# the corresponding MCP server / direct tool implementation.
#
# credential_type drives what the approval flow asks for:
#   "none"        — system auto-issues a token; nothing to collect (demo/mock)
#   "api_key"     — one secret value
#   "app_password"— an identifier (e.g. email address) + one secret value
#   "oauth2"      — needs an externally-registered OAuth client; approval
#                   shows setup instructions rather than collecting a field
#                   (the actual consent-flow code is not implemented here —
#                   see acquisition_instructions for why)
#
# planner_eligible=False keeps a service out of the Agent Maker's automatic
# catalog — it exists and is fully wired, but won't be silently offered to
# every "check my email"-shaped request. Promote it by flipping this flag
# once you've decided the system should offer real Gmail access by default.
KNOWN_SERVICES = {
    "mock_email": {
        "scope": "read-only access to your (mock) email inbox",
        "source": "mcp",
        "server": str(REPO_ROOT / "tools" / "shared" / "mock_email_mcp" / "server.py"),
        "credential_env": "MOCK_EMAIL_TOKEN",
        "credential_type": "none",
        "credential_fields": [],
        "acquisition_instructions": [],
        "planner_eligible": True,
        "real_service_note": (
            "This is a local mock with sample data, not your real Gmail account. "
            "It cannot read or send real email."
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
    "gmail_imap": {
        "scope": "read-only access to your real Gmail inbox via IMAP",
        "source": "direct",
        "build_tools": _gmail_tools,
        "credential_type": "app_password",
        "credential_fields": [
            {"name": "email_address", "label": "Your Gmail address", "secret": False},
            {"name": "app_password", "label": "16-character App Password", "secret": True},
        ],
        "acquisition_instructions": [
            "Requires 2-Step Verification enabled on the Google account.",
            "Go to https://myaccount.google.com/apppasswords",
            "Sign in, create a new app password (any name, e.g. 'enhanced_agent').",
            "Copy the 16-character code shown — that's the App Password below, "
            "not your normal Google password.",
        ],
        "planner_eligible": False,  # see module docstring: opt-in, not auto-offered
        "real_service_note": (
            "This connects to your real Gmail account read-only, over IMAP, "
            "using an app-specific password (not your main Google password and "
            "not full OAuth — no Google Cloud project needed). Revoke it anytime "
            "at myaccount.google.com/apppasswords."
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


def planner_catalog() -> dict:
    return {name: entry for name, entry in KNOWN_SERVICES.items() if entry.get("planner_eligible")}


def is_bound(user_id: str, service: str) -> bool:
    row = get_conn().execute(
        "SELECT 1 FROM capability_ledger WHERE user_id = %s AND service = %s AND action = 'bound' LIMIT 1",
        (user_id, service),
    ).fetchone()
    return row is not None


def bind(user_id: str, service: str, triggering_request: str, values: dict[str, str] | None = None) -> None:
    """Record an approved binding and store the credential.

    `values` must match the service's credential_fields exactly, except for
    credential_type "none", where nothing is collected and a demo token is
    auto-issued (the mock's only "credential" is that a value exists at all).
    """
    if service not in KNOWN_SERVICES:
        raise ValueError(f"unknown service: {service}")
    entry = KNOWN_SERVICES[service]
    field_names = {f["name"] for f in entry["credential_fields"]}

    if entry["credential_type"] == "none":
        stored = {"token": f"demo-{secrets.token_hex(8)}"}
    else:
        values = values or {}
        missing = field_names - set(values)
        if missing:
            raise ValueError(f"missing required credential fields for {service}: {missing}")
        stored = {name: values[name] for name in field_names}

    _store_credential(user_id, service, stored)
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
            json.dumps({"credential_type": entry["credential_type"]}),
        ),
    )


def get_tools(user_id: str, service: str) -> list[StructuredTool]:
    """Live tools for a bound service, wrapped for synchronous agents."""
    entry = KNOWN_SERVICES[service]
    stored = _load_credential(user_id, service)

    if entry["source"] == "direct":
        return entry["build_tools"](**stored)

    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient(
        {
            service: {
                "command": str(PYTHON),
                "args": [entry["server"]],
                "env": {entry["credential_env"]: stored["token"]},
                "transport": "stdio",
            }
        }
    )
    async_tools = asyncio.run(client.get_tools())
    return [_sync_wrap(t) for t in async_tools]


def _sync_wrap(async_tool) -> StructuredTool:
    def call(**kwargs):
        return asyncio.run(async_tool.ainvoke(kwargs))

    return StructuredTool(
        name=async_tool.name,
        description=async_tool.description,
        args_schema=async_tool.args_schema,
        func=call,
    )


def _store_credential(user_id: str, service: str, values: dict[str, str]) -> None:
    # Local dev convenience store, not a production secrets manager — a real
    # deployment should swap this for an OS keychain or a real vault (see
    # Anthropic's Managed Agents vault credentials for the shape of that).
    CREDENTIALS_PATH.parent.mkdir(exist_ok=True)
    CREDENTIALS_PATH.parent.chmod(stat.S_IRWXU)  # 0700: owner only
    store = json.loads(CREDENTIALS_PATH.read_text()) if CREDENTIALS_PATH.exists() else {}
    store[f"{user_id}/{service}"] = values
    CREDENTIALS_PATH.write_text(json.dumps(store, indent=2))
    CREDENTIALS_PATH.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 0600: owner only


def _load_credential(user_id: str, service: str) -> dict[str, str]:
    store = json.loads(CREDENTIALS_PATH.read_text())
    return store[f"{user_id}/{service}"]
