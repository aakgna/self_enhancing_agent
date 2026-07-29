"""Offline tests for real Gmail IMAP access, exercised against a fake
imaplib.IMAP4_SSL — never touches the network, never needs a real credential."""

import imaplib

import pytest

from tools.shared.gmail_imap import tool

SAMPLE_HEADERS = [
    b"From: boss@example.com\r\nSubject: URGENT: Q3 numbers\r\nDate: Mon\r\n\r\n",
    b"From: sarah@example.com\r\nSubject: lunch Thursday?\r\nDate: Tue\r\n\r\n",
]


class FakeIMAP:
    def __init__(self, host, timeout=None):
        self.host = host
        self.logged_out = False

    def login(self, address, password):
        if password != "correct-app-password":
            raise imaplib.IMAP4.error("AUTHENTICATIONFAILED")

    def select(self, mailbox, readonly=False):
        return "OK", [b"2"]

    def search(self, charset, criteria):
        if "UNSEEN" in criteria:
            return "OK", [b"1 2"]
        if "nomatch" in criteria:
            return "OK", [b""]
        return "OK", [b"1"]

    def fetch(self, msg_id, parts):
        idx = int(msg_id) - 1
        return "OK", [(b"1 (RFC822.HEADER {123}", SAMPLE_HEADERS[idx % len(SAMPLE_HEADERS)])]

    def logout(self):
        self.logged_out = True


@pytest.fixture(autouse=True)
def fake_imap(monkeypatch):
    monkeypatch.setattr(imaplib, "IMAP4_SSL", FakeIMAP)


def test_login_failure_raises_clear_error():
    with pytest.raises(tool.GmailAuthError):
        tool.list_unread("user@gmail.com", "wrong-password")


def test_list_unread_formats_real_headers():
    result = tool.list_unread("user@gmail.com", "correct-app-password", limit=5)
    assert "boss@example.com" in result
    assert "URGENT: Q3 numbers" in result


def test_search_email_no_matches():
    result = tool.search_email("user@gmail.com", "correct-app-password", "nomatch")
    assert "No emails matching" in result


def test_connection_is_always_logged_out(monkeypatch):
    captured = {}
    real_init = FakeIMAP.__init__

    def capturing_init(self, host, timeout=None):
        real_init(self, host, timeout)
        captured["conn"] = self

    monkeypatch.setattr(FakeIMAP, "__init__", capturing_init)
    tool.list_unread("user@gmail.com", "correct-app-password")
    assert captured["conn"].logged_out is True
