"""Optional FurHat voice I/O for the CLI REPL — connect(), speak(), listen()
over FurHat's own Realtime WebSocket API (ws://<ip>:9000/v1/events).

Trimmed from FurHat/app/robot/furhat.py (the reference implementation this
was ported from): only the conversational surface chat.py needs. LEDs,
gestures, face config, camera, and user-presence detection are not ported —
this repo's checkin agent doesn't scan anything, and those are cosmetic.

NOT VERIFIED AGAINST REAL HARDWARE. This is built against the documented
protocol and FurHat's own reference client, offline-tested only (see
tests/test_furhat_client.py, which fakes the websocket transport), because no
FurHat or Virtual FurHat was available to connect to during development —
same honesty standard this repo already applies to its untested oauth2
credential type (see mcp_bindings.py / README). The CLI fallback in chat.py
is the fully verified path; this is the optional enhancement on top of it.
"""

import asyncio
import json

import websockets

DEFAULT_CONNECT_TIMEOUT = 5.0


class FurHatUnavailable(RuntimeError):
    """Could not connect to (or lost connection to) a FurHat at the
    configured address. Callers should treat this as "fall back to CLI",
    not crash the session."""


class FurHatClient:
    def __init__(self, ip: str, port: int = 9000):
        self.uri = f"ws://{ip}:{port}/v1/events"
        self.ws = None

    async def connect(self, timeout: float = DEFAULT_CONNECT_TIMEOUT) -> None:
        try:
            # FurHat's WebSocket server does not respond to ping frames —
            # automatic pings must stay off or the client disconnects itself
            # after ping_timeout seconds of silence (see furhat.py upstream).
            self.ws = await asyncio.wait_for(
                websockets.connect(self.uri, ping_interval=None, ping_timeout=None),
                timeout=timeout,
            )
        except (OSError, asyncio.TimeoutError, websockets.exceptions.WebSocketException) as exc:
            raise FurHatUnavailable(f"could not connect to FurHat at {self.uri}: {exc}") from exc

    async def reconnect(self, timeout: float = DEFAULT_CONNECT_TIMEOUT) -> None:
        if self.ws is not None:
            try:
                await self.ws.close()
            except Exception:
                pass
        await self.connect(timeout=timeout)

    async def disconnect(self) -> None:
        if self.ws is not None:
            await self.ws.close()

    async def _send(self, payload: dict) -> None:
        await self.ws.send(json.dumps(payload))

    async def _wait_for(self, *event_types: str, timeout: float | None = None) -> dict:
        async def _recv_loop():
            while True:
                raw = await self.ws.recv()
                msg = json.loads(raw)
                if msg.get("type") in event_types:
                    return msg

        return await asyncio.wait_for(_recv_loop(), timeout=timeout)

    async def speak(self, text: str) -> None:
        try:
            await self._send({"type": "request.speak.text", "text": text, "abort": False, "interruptable": False})
            await self._wait_for("response.speak.end", timeout=30.0)
        except (OSError, asyncio.TimeoutError, websockets.exceptions.WebSocketException) as exc:
            raise FurHatUnavailable(f"FurHat connection lost while speaking: {exc}") from exc

    async def listen(self, no_speech_timeout: float = 8.0, end_speech_timeout: float = 1.2) -> str | None:
        try:
            await self._send(
                {
                    "type": "request.listen.start",
                    "partial": False,
                    "stop_no_speech": True,
                    "stop_robot_start": True,
                    "stop_user_end": True,
                    "no_speech_timeout": no_speech_timeout,
                    "end_speech_timeout": end_speech_timeout,
                }
            )
            msg = await self._wait_for(
                "response.hear.end", "response.listen.end", timeout=no_speech_timeout + 10.0
            )
        except (OSError, asyncio.TimeoutError, websockets.exceptions.WebSocketException) as exc:
            raise FurHatUnavailable(f"FurHat connection lost while listening: {exc}") from exc

        if msg["type"] == "response.hear.end":
            return msg.get("text", "").strip() or None
        return None
