import asyncio
import json

import pytest
from openai.types.realtime import RealtimeConversationItemUserMessage
from openai.types.realtime.realtime_conversation_item_user_message import (
    Content as RealtimeUserMessageContent,
)

from bodhi.servicer.realtime_manager import RealtimeWebSocketManager


class FakeWebSocket:
    def __init__(self):
        self.accepted = False
        self.sent_events: list[dict] = []

    async def accept(self) -> None:
        self.accepted = True

    async def send_text(self, payload: str) -> None:
        self.sent_events.append(json.loads(payload))


class FakeUpstream:
    def __init__(self):
        self.connected = False
        self.closed = False
        self.configured_sessions: list[dict] = []
        self.appended_audio: list[bytes] = []
        self.commit_calls = 0
        self.clear_calls = 0
        self.cancel_calls: list[str | None] = []
        self.created_items: list[dict] = []
        self.create_response_calls = 0
        self.function_outputs: list[tuple[str, str]] = []
        self._event_queue: asyncio.Queue = asyncio.Queue()

    async def connect(self) -> None:
        self.connected = True

    async def close(self) -> None:
        self.closed = True
        await self._event_queue.put(None)

    async def configure_session(self, *, tools, instructions: str) -> None:
        self.configured_sessions.append(
            {
                "tools": tools,
                "instructions": instructions,
            }
        )

    async def append_audio(self, audio_bytes: bytes) -> None:
        self.appended_audio.append(audio_bytes)

    async def commit_audio_buffer(self) -> None:
        self.commit_calls += 1

    async def clear_audio_buffer(self) -> None:
        self.clear_calls += 1

    async def create_conversation_item(self, *, item, previous_item_id=None) -> None:
        _ = previous_item_id
        self.created_items.append(item)

    async def create_response(self, *, response=None) -> None:
        _ = response
        self.create_response_calls += 1

    async def cancel_response(self, *, response_id=None) -> None:
        self.cancel_calls.append(response_id)

    async def send_function_call_output(self, *, call_id: str, output: str) -> None:
        self.function_outputs.append((call_id, output))

    async def events(self):
        while True:
            event = await self._event_queue.get()
            if event is None:
                return
            yield event


@pytest.mark.asyncio
async def test_manager_send_audio_user_message_commit_interrupt_and_disconnect(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        "bodhi.servicer.realtime_manager.AUDIO_SAVE_DIR",
        str(tmp_path),
    )

    upstream = FakeUpstream()
    manager = RealtimeWebSocketManager(upstream_factory=lambda: upstream)
    websocket = FakeWebSocket()

    await manager.connect(websocket, "s1")

    assert websocket.accepted is True
    assert upstream.connected is True
    assert len(upstream.configured_sessions) == 1
    assert len(upstream.configured_sessions[0]["tools"]) == 3
    assert upstream.configured_sessions[0]["instructions"]

    await manager.send_audio("s1", b"\x01\x02\x03\x04")
    assert upstream.appended_audio == [b"\x01\x02\x03\x04"]

    await manager.send_user_message(
        "s1",
        RealtimeConversationItemUserMessage(
            type="message",
            role="user",
            content=[RealtimeUserMessageContent(type="input_text", text="hello")],
        ),
    )
    assert [item.model_dump(exclude_none=True) for item in upstream.created_items] == [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "hello"}],
        }
    ]
    assert upstream.create_response_calls == 1

    await manager.send_client_event("s1", {"type": "input_audio_buffer.commit"})
    assert upstream.commit_calls == 1

    await manager.interrupt("s1")
    assert upstream.cancel_calls == [None]
    assert websocket.sent_events[-1] == {"type": "audio_interrupted"}

    await manager.disconnect("s1")

    assert upstream.closed is True
    assert "s1" not in manager.active_sessions


@pytest.mark.asyncio
async def test_manager_ignores_unknown_sessions_and_unknown_client_events():
    upstream = FakeUpstream()
    manager = RealtimeWebSocketManager(upstream_factory=lambda: upstream)

    await manager.send_audio("missing", b"abc")
    await manager.send_user_message(
        "missing",
        RealtimeConversationItemUserMessage(
            type="message",
            role="user",
            content=[],
        ),
    )
    await manager.send_client_event("missing", {"type": "input_audio_buffer.commit"})
    await manager.interrupt("missing")
    await manager.disconnect("missing")

    websocket = FakeWebSocket()
    await manager.connect(websocket, "s2")
    await manager.send_client_event("s2", {"type": "unsupported_event"})

    assert upstream.commit_calls == 0
    assert upstream.clear_calls == 0
    assert upstream.create_response_calls == 0

    await manager.disconnect("s2")
