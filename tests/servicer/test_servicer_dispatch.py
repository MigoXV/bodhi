import json

import pytest
from openai.types.realtime import RealtimeConversationItemUserMessage

from bodhi.servicer.handler import ClientEventHandlers
from bodhi.types.servicer import ClientEventAdapter


class FakeRealtimeManager:
    def __init__(self):
        self.audio_calls: list[tuple[str, bytes]] = []
        self.user_messages: list[tuple[str, RealtimeConversationItemUserMessage]] = []
        self.client_events: list[tuple[str, dict]] = []
        self.approvals: list[tuple[str, str, bool]] = []
        self.rejections: list[tuple[str, str, bool]] = []
        self.interrupts: list[str] = []

    async def send_audio(self, session_id: str, audio_bytes: bytes) -> None:
        self.audio_calls.append((session_id, audio_bytes))

    async def send_user_message(
        self,
        session_id: str,
        message: RealtimeConversationItemUserMessage,
    ) -> None:
        self.user_messages.append((session_id, message))

    async def send_client_event(self, session_id: str, event: dict) -> None:
        self.client_events.append((session_id, event))

    async def approve_tool_call(
        self, session_id: str, call_id: str, *, always: bool = False
    ) -> None:
        self.approvals.append((session_id, call_id, always))

    async def reject_tool_call(
        self, session_id: str, call_id: str, *, always: bool = False
    ) -> None:
        self.rejections.append((session_id, call_id, always))

    async def interrupt(self, session_id: str) -> None:
        self.interrupts.append(session_id)


def parse_event(payload: dict):
    return ClientEventAdapter.validate_python(payload)


@pytest.mark.asyncio
async def test_audio_handler_success_calls_manager_send_audio():
    manager = FakeRealtimeManager()
    handlers = ClientEventHandlers(manager)

    responses = await handlers.handle_audio("s1", parse_event({"type": "audio", "data": [1, 2, 3]}))

    assert responses == []
    assert len(manager.audio_calls) == 1


@pytest.mark.asyncio
async def test_audio_handler_invalid_int16_returns_error():
    manager = FakeRealtimeManager()
    handlers = ClientEventHandlers(manager)

    responses = await handlers.handle_audio(
        "s1", parse_event({"type": "audio", "data": [99999]})
    )

    assert len(responses) == 1
    assert responses[0].type == "error"
    assert responses[0].error == "Invalid client event."


@pytest.mark.asyncio
async def test_text_handler_blank_returns_error_and_valid_returns_ack():
    manager = FakeRealtimeManager()
    handlers = ClientEventHandlers(manager)

    blank = await handlers.handle_text("s1", parse_event({"type": "text", "text": "   "}))
    assert blank[0].type == "error"
    assert blank[0].error == "Empty text message."

    valid = await handlers.handle_text("s1", parse_event({"type": "text", "text": "hello"}))
    assert valid[0].type == "client_info"
    assert valid[0].info == "text_enqueued"


@pytest.mark.asyncio
async def test_image_direct_and_chunked_paths_enqueue_image_messages():
    manager = FakeRealtimeManager()
    handlers = ClientEventHandlers(manager)

    data_url = "data:image/jpeg;base64,ABCDEFG=="

    direct = await handlers.handle_image(
        "s1",
        parse_event({"type": "image", "data_url": data_url, "text": "describe"}),
    )

    await handlers.handle_image_start(
        "s2", parse_event({"type": "image_start", "id": "img-x", "text": "describe"})
    )
    await handlers.handle_image_chunk(
        "s2", parse_event({"type": "image_chunk", "id": "img-x", "chunk": data_url[:12]})
    )
    chunked = await handlers.handle_image_end(
        "s2", parse_event({"type": "image_end", "id": "img-x"})
    )

    assert direct[0].type == "client_info"
    assert direct[0].info == "image_enqueued"
    assert chunked[0].type == "client_info"
    assert chunked[0].info == "image_enqueued"
    assert len(manager.user_messages) == 2


@pytest.mark.asyncio
async def test_image_chunk_ack_every_ten_and_cleanup_after_end():
    manager = FakeRealtimeManager()
    handlers = ClientEventHandlers(manager)

    await handlers.handle_image_start(
        "s1", parse_event({"type": "image_start", "id": "img-1", "text": "desc"})
    )

    ack = None
    for index in range(10):
        responses = await handlers.handle_image_chunk(
            "s1", parse_event({"type": "image_chunk", "id": "img-1", "chunk": str(index)})
        )
        if responses:
            ack = responses[0]

    assert ack is not None
    assert ack.type == "client_info"
    assert ack.info == "image_chunk_ack"
    assert ack.count == 10

    end = await handlers.handle_image_end(
        "s1", parse_event({"type": "image_end", "id": "img-1"})
    )
    assert end[0].type == "client_info"
    assert "s1" not in handlers.image_buffers


@pytest.mark.asyncio
async def test_image_end_unknown_or_empty_returns_error():
    manager = FakeRealtimeManager()
    handlers = ClientEventHandlers(manager)

    unknown = await handlers.handle_image_end(
        "s1", parse_event({"type": "image_end", "id": "unknown"})
    )
    assert unknown[0].type == "error"
    assert unknown[0].error == "Unknown image id for image_end."

    await handlers.handle_image_start(
        "s1", parse_event({"type": "image_start", "id": "img-2", "text": "desc"})
    )
    empty = await handlers.handle_image_end(
        "s1", parse_event({"type": "image_end", "id": "img-2"})
    )
    assert empty[0].type == "error"
    assert empty[0].error == "Empty image."


@pytest.mark.asyncio
async def test_tool_approval_and_interrupt_handlers():
    manager = FakeRealtimeManager()
    handlers = ClientEventHandlers(manager)

    missing = await handlers.handle_tool_approval_decision(
        "s1", parse_event({"type": "tool_approval_decision", "approve": True})
    )
    assert missing[0].type == "error"
    assert missing[0].error == "Missing call_id for tool approval decision."

    await handlers.handle_tool_approval_decision(
        "s1",
        parse_event(
            {
                "type": "tool_approval_decision",
                "call_id": "call1",
                "approve": True,
                "always": True,
            }
        ),
    )
    await handlers.handle_tool_approval_decision(
        "s1",
        parse_event(
            {
                "type": "tool_approval_decision",
                "call_id": "call2",
                "approve": False,
                "always": False,
            }
        ),
    )
    await handlers.handle_interrupt("s1", parse_event({"type": "interrupt"}))
    await handlers.handle_commit_audio("s1", parse_event({"type": "commit_audio"}))

    assert manager.approvals == [("s1", "call1", True)]
    assert manager.rejections == [("s1", "call2", False)]
    assert manager.interrupts == ["s1"]
    assert manager.client_events == [("s1", {"type": "input_audio_buffer.commit"})]


@pytest.mark.asyncio
async def test_cleanup_session_clears_buffers():
    handlers = ClientEventHandlers(FakeRealtimeManager())

    await handlers.handle_image_start(
        "s1", parse_event({"type": "image_start", "id": "img-1", "text": "x"})
    )
    assert "s1" in handlers.image_buffers

    handlers.cleanup_session("s1")

    assert "s1" not in handlers.image_buffers
