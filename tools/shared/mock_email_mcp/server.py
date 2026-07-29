"""Mock email MCP server (stdio) — the stand-in for Gmail so the Phase 4
credentialed-binding flow is demonstrable end to end without real OAuth.

Like a real credentialed service, it refuses to start without its credential.
"""

import os
import sys

from mcp.server.fastmcp import FastMCP

if not os.environ.get("MOCK_EMAIL_TOKEN"):
    sys.exit("mock_email_mcp: MOCK_EMAIL_TOKEN is required")

mcp = FastMCP("mock-email")

_INBOX = [
    {"from": "boss@example.com", "subject": "URGENT: Q3 numbers needed by Friday", "unread": True},
    {"from": "newsletter@techdigest.io", "subject": "This week in AI", "unread": True},
    {"from": "sarah@example.com", "subject": "Re: lunch on Thursday?", "unread": True},
    {"from": "it-support@example.com", "subject": "Scheduled maintenance Saturday", "unread": False},
]


@mcp.tool()
def list_unread(limit: int = 10) -> str:
    """List unread emails in the user's inbox, most recent first."""
    unread = [m for m in _INBOX if m["unread"]][:limit]
    if not unread:
        return "No unread emails."
    return "\n".join(f"- from {m['from']}: {m['subject']}" for m in unread)


@mcp.tool()
def search_email(query: str) -> str:
    """Search all emails (subject and sender) for a query string."""
    query_lower = query.lower()
    found = [
        m
        for m in _INBOX
        if query_lower in m["subject"].lower() or query_lower in m["from"].lower()
    ]
    if not found:
        return f"No emails matching {query!r}."
    return "\n".join(f"- from {m['from']}: {m['subject']}" for m in found)


if __name__ == "__main__":
    mcp.run()
