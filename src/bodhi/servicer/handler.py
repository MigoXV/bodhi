import logging
import struct
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Protocol

from openai.types.realtime import RealtimeConversationItemUserMessage
from openai.types.realtime.realtime_conversation_item_user_message import (
    Content as RealtimeUserMessageContent,
)

from bodhi.types.servicer import (
    AudioEvent,
    ClientEventUnion,
    ClientInfoEnvelope,
    CommitAudioEvent,
    ErrorEnvelope,
    ImageChunkEvent,
    ImageEndEvent,
    ImageEvent,
    ImageStartEvent,
    InterruptEvent,
    SetVoiceEvent,
    ServerEnvelope,
    TextEvent,
    ToolApprovalDecisionEvent,
)

logger = logging.getLogger(__name__)


DEFAULT_IMAGE_PROMPT = "Please describe this image."


@dataclass
class ImageBufferState:
    text: str
    chunks: list[str] = field(default_factory=list)


class RealtimeManagerProtocol(Protocol):
    async def send_audio(self, session_id: str, audio_bytes: bytes) -> None: ...

    async def send_user_message(
        self, session_id: str, message: RealtimeConversationItemUserMessage
    ) -> None: ...

    async def send_client_event(self, session_id: str, event: dict[str, Any]) -> None: ...

    async def approve_tool_call(
        self, session_id: str, call_id: str, *, always: bool = False
    ) -> None: ...

    async def reject_tool_call(
        self, session_id: str, call_id: str, *, always: bool = False
    ) -> None: ...

    async def interrupt(self, session_id: str) -> None: ...

    async def set_voice(self, session_id: str, voice: str) -> None: ...


HandlerType = Callable[[str, ClientEventUnion], Awaitable[list[ServerEnvelope]]]


class ClientEventHandlers:
    def __init__(self, realtime_manager: RealtimeManagerProtocol):
        self.realtime_manager = realtime_manager
        self.image_buffers: dict[str, dict[str, ImageBufferState]] = {}
        self.handlers: dict[str, HandlerType] = {
            "audio": self.handle_audio,
            "text": self.handle_text,
            "image": self.handle_image,
            "commit_audio": self.handle_commit_audio,
            "image_start": self.handle_image_start,
            "image_chunk": self.handle_image_chunk,
            "image_end": self.handle_image_end,
            "tool_approval_decision": self.handle_tool_approval_decision,
            "interrupt": self.handle_interrupt,
            "set_voice": self.handle_set_voice,
        }

    def cleanup_session(self, session_id: str) -> None:
        self.image_buffers.pop(session_id, None)

    async def handle_audio(
        self, session_id: str, event: ClientEventUnion
    ) -> list[ServerEnvelope]:
        assert isinstance(event, AudioEvent)
        try:
            audio_bytes = struct.pack(f"{len(event.data)}h", *event.data)
        except struct.error:
            return [ErrorEnvelope(error="Invalid client event.")]

        await self.realtime_manager.send_audio(session_id, audio_bytes)
        return []

    async def handle_text(
        self, session_id: str, event: ClientEventUnion
    ) -> list[ServerEnvelope]:
        assert isinstance(event, TextEvent)

        text = event.text.strip()
        if not text:
            return [ErrorEnvelope(error="Empty text message.")]

        user_text_msg = RealtimeConversationItemUserMessage(
            type="message",
            role="user",
            content=[
                RealtimeUserMessageContent(
                    type="input_text",
                    text=text,
                )
            ],
        )
        await self.realtime_manager.send_user_message(session_id, user_text_msg)
        return [ClientInfoEnvelope(info="text_enqueued", size=len(text))]

    async def handle_image(
        self, session_id: str, event: ClientEventUnion
    ) -> list[ServerEnvelope]:
        assert isinstance(event, ImageEvent)

        if not event.data_url:
            return [ErrorEnvelope(error="No data_url for image message.")]

        prompt_text = event.text or DEFAULT_IMAGE_PROMPT
        await self._enqueue_image_message(session_id, event.data_url, prompt_text)
        return [ClientInfoEnvelope(info="image_enqueued", size=len(event.data_url))]

    async def handle_commit_audio(
        self, session_id: str, event: ClientEventUnion
    ) -> list[ServerEnvelope]:
        assert isinstance(event, CommitAudioEvent)

        await self.realtime_manager.send_client_event(
            session_id, {"type": "input_audio_buffer.commit"}
        )
        return []

    async def handle_image_start(
        self, session_id: str, event: ClientEventUnion
    ) -> list[ServerEnvelope]:
        assert isinstance(event, ImageStartEvent)

        image_id = str(event.id)
        session_buffers = self.image_buffers.setdefault(session_id, {})
        session_buffers[image_id] = ImageBufferState(
            text=event.text or DEFAULT_IMAGE_PROMPT,
        )
        return [ClientInfoEnvelope(info="image_start_ack", id=image_id)]

    async def handle_image_chunk(
        self, session_id: str, event: ClientEventUnion
    ) -> list[ServerEnvelope]:
        assert isinstance(event, ImageChunkEvent)

        image_id = str(event.id)
        session_buffers = self.image_buffers.get(session_id)
        if not session_buffers:
            return []

        state = session_buffers.get(image_id)
        if state is None:
            return []

        state.chunks.append(event.chunk)
        if len(state.chunks) % 10 == 0:
            return [
                ClientInfoEnvelope(
                    info="image_chunk_ack",
                    id=image_id,
                    count=len(state.chunks),
                )
            ]
        return []

    async def handle_image_end(
        self, session_id: str, event: ClientEventUnion
    ) -> list[ServerEnvelope]:
        assert isinstance(event, ImageEndEvent)

        image_id = str(event.id)
        session_buffers = self.image_buffers.get(session_id)
        if not session_buffers:
            return [ErrorEnvelope(error="Unknown image id for image_end.")]

        state = session_buffers.pop(image_id, None)
        if state is None:
            return [ErrorEnvelope(error="Unknown image id for image_end.")]

        if not session_buffers:
            self.image_buffers.pop(session_id, None)

        data_url = "".join(state.chunks) if state.chunks else ""
        if not data_url:
            return [ErrorEnvelope(error="Empty image.")]

        await self._enqueue_image_message(session_id, data_url, state.text)
        return [
            ClientInfoEnvelope(
                info="image_enqueued",
                id=image_id,
                size=len(data_url),
            )
        ]

    async def handle_tool_approval_decision(
        self, session_id: str, event: ClientEventUnion
    ) -> list[ServerEnvelope]:
        assert isinstance(event, ToolApprovalDecisionEvent)

        if not event.call_id:
            return [
                ErrorEnvelope(
                    error="Missing call_id for tool approval decision.",
                )
            ]

        if event.approve:
            await self.realtime_manager.approve_tool_call(
                session_id,
                event.call_id,
                always=bool(event.always),
            )
        else:
            await self.realtime_manager.reject_tool_call(
                session_id,
                event.call_id,
                always=bool(event.always),
            )
        return []

    async def handle_interrupt(
        self, session_id: str, event: ClientEventUnion
    ) -> list[ServerEnvelope]:
        assert isinstance(event, InterruptEvent)

        await self.realtime_manager.interrupt(session_id)
        return []

    async def handle_set_voice(
        self, session_id: str, event: ClientEventUnion
    ) -> list[ServerEnvelope]:
        assert isinstance(event, SetVoiceEvent)

        voice = event.voice.strip()
        if not voice:
            return [ErrorEnvelope(error="Empty voice value.")]

        await self.realtime_manager.set_voice(session_id, voice)
        return [ClientInfoEnvelope(info="voice_updated", id=voice)]

    async def _enqueue_image_message(
        self,
        session_id: str,
        data_url: str,
        prompt_text: str,
    ) -> None:
        logger.info(
            "Forwarding image (structured message) to Realtime API (len=%d).",
            len(data_url),
        )
        content = [
            RealtimeUserMessageContent(
                type="input_image",
                image_url=data_url,
                detail="high",
            )
        ]
        if prompt_text:
            content.append(
                RealtimeUserMessageContent(
                    type="input_text",
                    text=prompt_text,
                )
            )

        user_msg = RealtimeConversationItemUserMessage(
            type="message",
            role="user",
            content=content,
        )
        await self.realtime_manager.send_user_message(session_id, user_msg)
