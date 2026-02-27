import json

import pytest
from openai.types.realtime import RealtimeConversationItemUserMessage
from pydantic import ValidationError

from bodhi.servicer.servicer import SessionMessageService
from bodhi.types.servicer import ClientEventAdapter


class FakeRealtimeManager:
    async def send_audio(self, session_id: str, audio_bytes: bytes) -> None:
        return None

    async def send_user_message(
        self,
        session_id: str,
        message: RealtimeConversationItemUserMessage,
    ) -> None:
        return None

    async def send_client_event(self, session_id: str, event: dict) -> None:
        return None

    async def approve_tool_call(
        self, session_id: str, call_id: str, *, always: bool = False
    ) -> None:
        return None

    async def reject_tool_call(
        self, session_id: str, call_id: str, *, always: bool = False
    ) -> None:
        return None

    async def interrupt(self, session_id: str) -> None:
        return None


class SpyHandlers:
    def __init__(self):
        self.called: list[tuple[str, object]] = []
        self.cleaned: list[str] = []
        self.handlers = {
            "text": self.handle_text,
        }

    async def handle_text(self, session_id: str, event):
        self.called.append((session_id, event))
        return []

    def cleanup_session(self, session_id: str) -> None:
        self.cleaned.append(session_id)


@pytest.mark.parametrize(
    "payload",
    [
        {"type": "audio", "data": [1, -1, 2]},
        {"type": "text", "text": "hello"},
        {"type": "image", "data_url": "data:image/jpeg;base64,AA==", "text": "describe"},
        {"type": "commit_audio"},
        {"type": "image_start", "id": "img1", "text": "look"},
        {"type": "image_chunk", "id": "img1", "chunk": "abc"},
        {"type": "image_end", "id": "img1"},
        {"type": "tool_approval_decision", "call_id": "c1", "approve": True, "always": False},
        {"type": "interrupt"},
    ],
)
def test_client_event_adapter_accepts_supported_events(payload):
    event = ClientEventAdapter.validate_python(payload)
    assert event.type == payload["type"]


def test_client_event_adapter_rejects_unknown_type():
    with pytest.raises(ValidationError):
        ClientEventAdapter.validate_python({"type": "unknown", "foo": "bar"})


def test_client_event_adapter_rejects_missing_required_field():
    with pytest.raises(ValidationError):
        ClientEventAdapter.validate_python({"type": "audio"})


@pytest.mark.asyncio
async def test_service_invalid_json_returns_error():
    service = SessionMessageService(FakeRealtimeManager())

    responses = await service.handle_payload("s1", "{")

    assert len(responses) == 1
    assert responses[0].type == "error"
    assert responses[0].error == "Invalid JSON payload."


@pytest.mark.asyncio
async def test_service_invalid_event_returns_error():
    service = SessionMessageService(FakeRealtimeManager())

    responses = await service.handle_payload("s1", json.dumps({"type": "audio"}))

    assert len(responses) == 1
    assert responses[0].type == "error"
    assert responses[0].error == "Invalid client event."


@pytest.mark.asyncio
async def test_service_dispatches_to_handlers_registry():
    spy_handlers = SpyHandlers()
    service = SessionMessageService(FakeRealtimeManager(), handlers=spy_handlers)

    responses = await service.handle_payload("s1", json.dumps({"type": "text", "text": "hello"}))

    assert responses == []
    assert len(spy_handlers.called) == 1
    assert spy_handlers.called[0][0] == "s1"
    assert spy_handlers.called[0][1].type == "text"


def test_service_cleanup_delegates_to_handlers():
    spy_handlers = SpyHandlers()
    service = SessionMessageService(FakeRealtimeManager(), handlers=spy_handlers)

    service.cleanup_session("s1")

    assert spy_handlers.cleaned == ["s1"]
