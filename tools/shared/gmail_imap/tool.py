"""Real Gmail access over IMAP, authenticated with a per-app password (the
16-character code from https://myaccount.google.com/apppasswords) — not full
OAuth. This needs no registered Google Cloud client and no consent screen;
it's a plain secret, so it's the credential type fully buildable without any
external app registration.

Only ever called from execute_node after explicit user approval and value
collection (see orchestrator/mcp_bindings.py). Never imported by, or
reachable from, the generation sandbox, which denies all network access
regardless.
"""

import email
import imaplib
from email.header import decode_header

IMAP_HOST = "imap.gmail.com"
IMAP_TIMEOUT = 15


class GmailAuthError(RuntimeError):
    """Raised on login failure — bad address/password, 2FA not enabled, etc."""


def _decode(raw: str | None) -> str:
    if not raw:
        return ""
    parts = decode_header(raw)
    return "".join(
        (chunk.decode(enc or "utf-8", errors="replace") if isinstance(chunk, bytes) else chunk)
        for chunk, enc in parts
    )


def _connect(email_address: str, app_password: str) -> imaplib.IMAP4_SSL:
    try:
        conn = imaplib.IMAP4_SSL(IMAP_HOST, timeout=IMAP_TIMEOUT)
        conn.login(email_address, app_password)
    except imaplib.IMAP4.error as exc:
        raise GmailAuthError(
            "Gmail login failed. Common causes: the app password was mistyped, "
            "2-Step Verification isn't enabled on the account (required for app "
            "passwords), or the app password was revoked. "
            f"({exc})"
        ) from exc
    return conn


def _format_message(raw_msg: bytes) -> dict:
    parsed = email.message_from_bytes(raw_msg)
    return {
        "from": _decode(parsed.get("From")),
        "subject": _decode(parsed.get("Subject")),
        "date": parsed.get("Date", ""),
    }


def list_unread(email_address: str, app_password: str, limit: int = 10) -> str:
    """List unread emails, most recent first."""
    conn = _connect(email_address, app_password)
    try:
        conn.select("INBOX", readonly=True)
        status, data = conn.search(None, "UNSEEN")
        if status != "OK":
            return "Could not search the inbox."
        ids = data[0].split()
        if not ids:
            return "No unread emails."
        recent_ids = list(reversed(ids))[:limit]
        lines = []
        for msg_id in recent_ids:
            status, msg_data = conn.fetch(msg_id, "(RFC822.HEADER)")
            if status != "OK" or not msg_data or not msg_data[0]:
                continue
            info = _format_message(msg_data[0][1])
            lines.append(f"- from {info['from']}: {info['subject']}")
        return "\n".join(lines) if lines else "No unread emails."
    finally:
        conn.logout()


def search_email(email_address: str, app_password: str, query: str) -> str:
    """Search subject/sender for a query string."""
    conn = _connect(email_address, app_password)
    try:
        conn.select("INBOX", readonly=True)
        status, data = conn.search(None, f'(OR SUBJECT "{query}" FROM "{query}")')
        if status != "OK":
            return "Could not search the inbox."
        ids = data[0].split()
        if not ids:
            return f"No emails matching {query!r}."
        lines = []
        for msg_id in list(reversed(ids))[:20]:
            status, msg_data = conn.fetch(msg_id, "(RFC822.HEADER)")
            if status != "OK" or not msg_data or not msg_data[0]:
                continue
            info = _format_message(msg_data[0][1])
            lines.append(f"- from {info['from']}: {info['subject']}")
        return "\n".join(lines) if lines else f"No emails matching {query!r}."
    finally:
        conn.logout()
