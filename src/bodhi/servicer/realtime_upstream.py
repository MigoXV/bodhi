from __future__ import annotations

import base64
import logging
import os
from dataclasses import dataclass
from typing import AsyncIterator
from urllib.parse import urlparse, urlunparse

from openai import AsyncOpenAI
from openai.resources.realtime.realtime import (
    AsyncRealtimeConnection,
    AsyncRealtimeConnectionManager,
)
from openai.types.realtime import (
    RealtimeAudioConfig,
    RealtimeAudioConfigInput,
    RealtimeAudioConfigOutput,
    RealtimeConversationItemFunctionCallOutput,
    RealtimeConversationItemUserMessage,
    RealtimeFunctionTool,
    RealtimeResponseCreateParams,
    RealtimeServerEvent,
    RealtimeSessionCreateRequest,
)
from openai.types.realtime.audio_transcription import AudioTranscription
from openai.types.realtime.realtime_audio_formats import AudioPCM
from openai.types.realtime.realtime_audio_input_turn_detection import ServerVad

logger = logging.getLogger(__name__)

DEFAULT_REALTIME_MODEL = "gpt-realtime"
DEFAULT_TRANSCRIPTION_MODEL = "whisper-1"


@dataclass(frozen=True)
class RealtimeUpstreamConfig:
    model_name: str
    transcription_model: str
    api_key: str | None
    base_url: str | None
    websocket_base_url: str | None


class OpenAIRealtimeUpstream:
    def __init__(self, config: RealtimeUpstreamConfig):
        self.config = config
        self._client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            websocket_base_url=config.websocket_base_url,
        )
        self._connection_manager: AsyncRealtimeConnectionManager | None = None
        self._connection: AsyncRealtimeConnection | None = None

    @classmethod
    def from_env(cls) -> "OpenAIRealtimeUpstream":
        realtime_url = os.getenv("OPENAI_REALTIME_URL")
        websocket_base_url = _derive_websocket_base_url(realtime_url)

        config = RealtimeUpstreamConfig(
            model_name=os.getenv("OPENAI_REALTIME_MODEL_NAME", DEFAULT_REALTIME_MODEL),
            transcription_model=os.getenv(
                "OPENAI_REALTIME_TRANSCRIPTION_MODEL",
                DEFAULT_TRANSCRIPTION_MODEL,
            ),
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            websocket_base_url=websocket_base_url,
        )
        return cls(config)

    async def connect(self) -> None:
        manager = self._client.realtime.connect(model=self.config.model_name)
        connection = await manager.__aenter__()
        self._connection_manager = manager
        self._connection = connection

    async def close(self) -> None:
        manager = self._connection_manager
        self._connection = None
        self._connection_manager = None

        if manager is not None:
            await manager.__aexit__(None, None, None)

        if not self._client.is_closed():
            await self._client.close()

    async def configure_session(
        self,
        *,
        tools: list[RealtimeFunctionTool],
        instructions: str,
    ) -> None:
        session = RealtimeSessionCreateRequest(
            type="realtime",
            audio=RealtimeAudioConfig(
                input=RealtimeAudioConfigInput(
                    format=AudioPCM(type="audio/pcm", rate=24000),
                    turn_detection=ServerVad(
                        type="server_vad",
                        prefix_padding_ms=300,
                        silence_duration_ms=500,
                        interrupt_response=True,
                        create_response=True,
                    ),
                    transcription=AudioTranscription(
                        model=self.config.transcription_model,
                    ),
                ),
                output=RealtimeAudioConfigOutput(
                    format=AudioPCM(type="audio/pcm"),
                ),
            ),
            output_modalities=["audio"],
            instructions=instructions,
            tools=tools,
            tool_choice="auto",
        )
        await self._require_connection().session.update(session=session)

    async def events(self) -> AsyncIterator[RealtimeServerEvent]:
        async for event in self._require_connection():
            yield event

    async def append_audio(self, audio_bytes: bytes) -> None:
        encoded = base64.b64encode(audio_bytes).decode("utf-8")
        await self._require_connection().input_audio_buffer.append(audio=encoded)

    async def commit_audio_buffer(self) -> None:
        await self._require_connection().input_audio_buffer.commit()

    async def clear_audio_buffer(self) -> None:
        await self._require_connection().input_audio_buffer.clear()

    async def create_conversation_item(
        self,
        *,
        item: RealtimeConversationItemUserMessage
        | RealtimeConversationItemFunctionCallOutput,
        previous_item_id: str | None = None,
    ) -> None:
        if previous_item_id:
            await self._require_connection().conversation.item.create(
                item=item,
                previous_item_id=previous_item_id,
            )
            return
        await self._require_connection().conversation.item.create(item=item)

    async def create_response(
        self,
        *,
        response: RealtimeResponseCreateParams | None = None,
    ) -> None:
        if response is None:
            await self._require_connection().response.create()
            return
        await self._require_connection().response.create(response=response)

    async def cancel_response(self, *, response_id: str | None = None) -> None:
        if response_id:
            await self._require_connection().response.cancel(response_id=response_id)
            return
        await self._require_connection().response.cancel()

    async def send_function_call_output(self, *, call_id: str, output: str) -> None:
        await self.create_conversation_item(
            item=RealtimeConversationItemFunctionCallOutput(
                type="function_call_output",
                call_id=call_id,
                output=output,
            )
        )

    def _require_connection(self) -> AsyncRealtimeConnection:
        if self._connection is None:
            raise RuntimeError("Realtime upstream is not connected.")
        return self._connection


def _derive_websocket_base_url(realtime_url: str | None) -> str | None:
    if not realtime_url:
        return None

    parsed = urlparse(realtime_url)
    if not parsed.scheme or not parsed.netloc:
        logger.warning(
            "Invalid OPENAI_REALTIME_URL=%s; ignoring websocket_base_url", realtime_url
        )
        return None

    path = (parsed.path or "").rstrip("/")
    if path.endswith("/realtime"):
        path = path[: -len("/realtime")]

    return urlunparse((parsed.scheme, parsed.netloc, path or "/", "", "", ""))
