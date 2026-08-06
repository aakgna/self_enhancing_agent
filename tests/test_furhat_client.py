"""Offline tests for FurHatClient, exercised against a fake websocket
transport — never opens a real socket, never needs a real FurHat.

The connected path itself is not verified against real hardware (see
orchestrator/furhat_io.py's module docstring) — these tests only establish
that the client speaks the documented protocol correctly and degrades to a
typed FurHatUnavailable (never a bare crash) on failure, timeout, or a
mid-session drop."""

import asyncio
import json

import pytest

from orchestrator import furhat_io
from orchestrator.furhat_io import FurHatClient, FurHatUnavailable


class FakeWebSocket:
    def __init__(self, script):
        self.sent: list[dict] = []
        self._script = list(script)
        self.closed = False

    async def send(self, raw):
        self.sent.append(json.loads(raw))

    async def recv(self):
        if not self._script:
            raise asyncio.TimeoutError("fake transport exhausted")
        return json.dumps(self._script.pop(0))

    async def close(self):
        self.closed = True


def _fake_connect(ws):
    async def connect(uri, **kwargs):
        return ws

    return connect


def _raising_connect(exc):
    async def connect(uri, **kwargs):
        raise exc

    return connect


def _hanging_connect():
    async def connect(uri, **kwargs):
        await asyncio.sleep(999)

    return connect


def test_connect_success(monkeypatch):
    ws = FakeWebSocket([])
    monkeypatch.setattr(furhat_io.websockets, "connect", _fake_connect(ws))
    client = FurHatClient("127.0.0.1")

    asyncio.run(client.connect())

    assert client.ws is ws


def test_connect_failure_raises_furhat_unavailable(monkeypatch):
    monkeypatch.setattr(furhat_io.websockets, "connect", _raising_connect(OSError("refused")))
    client = FurHatClient("127.0.0.1")

    with pytest.raises(FurHatUnavailable):
        asyncio.run(client.connect())


def test_connect_timeout_raises_furhat_unavailable(monkeypatch):
    monkeypatch.setattr(furhat_io.websockets, "connect", _hanging_connect())
    client = FurHatClient("127.0.0.1")

    with pytest.raises(FurHatUnavailable):
        asyncio.run(client.connect(timeout=0.05))


def test_speak_sends_the_documented_event_and_waits_for_end(monkeypatch):
    ws = FakeWebSocket([{"type": "response.speak.end"}])
    monkeypatch.setattr(furhat_io.websockets, "connect", _fake_connect(ws))
    client = FurHatClient("127.0.0.1")
    asyncio.run(client.connect())

    asyncio.run(client.speak("hello there"))

    assert ws.sent == [
        {"type": "request.speak.text", "text": "hello there", "abort": False, "interruptable": False}
    ]


def test_listen_returns_heard_text(monkeypatch):
    ws = FakeWebSocket([{"type": "response.hear.end", "text": "hi there"}])
    monkeypatch.setattr(furhat_io.websockets, "connect", _fake_connect(ws))
    client = FurHatClient("127.0.0.1")
    asyncio.run(client.connect())

    heard = asyncio.run(client.listen())

    assert heard == "hi there"
    assert ws.sent[0]["type"] == "request.listen.start"


def test_listen_silence_returns_none(monkeypatch):
    ws = FakeWebSocket([{"type": "response.listen.end"}])
    monkeypatch.setattr(furhat_io.websockets, "connect", _fake_connect(ws))
    client = FurHatClient("127.0.0.1")
    asyncio.run(client.connect())

    assert asyncio.run(client.listen()) is None


def test_speak_mid_session_drop_raises_furhat_unavailable_not_a_crash(monkeypatch):
    # No scripted response: recv() times out, simulating a dropped connection.
    ws = FakeWebSocket([])
    monkeypatch.setattr(furhat_io.websockets, "connect", _fake_connect(ws))
    client = FurHatClient("127.0.0.1")
    asyncio.run(client.connect())

    with pytest.raises(FurHatUnavailable):
        asyncio.run(client.speak("are you still there"))
