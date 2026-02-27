from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import wave
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from fastapi import WebSocket
from openai.types.realtime import ResponseDoneEvent, RealtimeConversationItemUserMessage

from bodhi.servicer.realtime_events import RealtimeEventMapper
from bodhi.servicer.realtime_tools import RealtimeToolCoordinator, RealtimeToolRegistry
from bodhi.servicer.realtime_upstream import OpenAIRealtimeUpstream

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AUDIO_SAVE_DIR = "data-bin/agent/test01"

UpstreamFactory = Callable[[], OpenAIRealtimeUpstream]


@dataclass
class SessionRuntime:
    websocket: WebSocket
    upstream: OpenAIRealtimeUpstream
    event_mapper: RealtimeEventMapper
    tool_coordinator: RealtimeToolCoordinator
    event_task: asyncio.Task[None] | None = None
    audio_buffer: bytearray = field(default_factory=bytearray)


class RealtimeWebSocketManager:
    def __init__(
        self,
        *,
        upstream_factory: UpstreamFactory | None = None,
        tool_registry: RealtimeToolRegistry | None = None,
    ):
        self._upstream_factory = upstream_factory or OpenAIRealtimeUpstream.from_env
        self._tool_registry = tool_registry or RealtimeToolRegistry()
        self.active_sessions: dict[str, SessionRuntime] = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()

        upstream = self._upstream_factory()
        event_mapper = RealtimeEventMapper()
        tool_coordinator = RealtimeToolCoordinator(
            registry=self._tool_registry,
            emit_event=lambda event: self._emit_to_frontend(session_id, event),
            send_function_call_output=lambda call_id, output: upstream.send_function_call_output(
                call_id=call_id,
                output=output,
            ),
            create_response=upstream.create_response,
        )
        runtime = SessionRuntime(
            websocket=websocket,
            upstream=upstream,
            event_mapper=event_mapper,
            tool_coordinator=tool_coordinator,
        )
        self.active_sessions[session_id] = runtime

        try:
            await upstream.connect()
            await upstream.configure_session(
                tools=self._tool_registry.tool_definitions(),
                instructions=self._tool_registry.instructions,
            )
        except Exception:
            self.active_sessions.pop(session_id, None)
            await upstream.close()
            raise

        runtime.event_task = asyncio.create_task(
            self._process_events(session_id),
            name=f"realtime-events-{session_id}",
        )

    async def disconnect(self, session_id: str):
        runtime = self.active_sessions.pop(session_id, None)
        if runtime is None:
            return

        self._save_audio_buffer(session_id, runtime.audio_buffer)

        if runtime.event_task is not None:
            runtime.event_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await runtime.event_task

        await runtime.upstream.close()

    async def send_audio(self, session_id: str, audio_bytes: bytes):
        runtime = self.active_sessions.get(session_id)
        if runtime is None:
            return

        runtime.audio_buffer.extend(audio_bytes)
        await runtime.upstream.append_audio(audio_bytes)

    async def send_client_event(self, session_id: str, event: dict[str, Any]):
        runtime = self.active_sessions.get(session_id)
        if runtime is None:
            return

        event_type = event.get("type")
        if event_type == "input_audio_buffer.commit":
            await runtime.upstream.commit_audio_buffer()
            return

        if event_type == "input_audio_buffer.clear":
            await runtime.upstream.clear_audio_buffer()
            return

        if event_type == "response.cancel":
            response_id = event.get("response_id")
            await runtime.upstream.cancel_response(
                response_id=response_id if isinstance(response_id, str) else None
            )
            return

        if event_type == "response.create":
            await runtime.upstream.create_response()
            return

        logger.warning("Unsupported client event: %s", event)

    async def send_user_message(
        self,
        session_id: str,
        message: RealtimeConversationItemUserMessage,
    ):
        runtime = self.active_sessions.get(session_id)
        if runtime is None:
            return

        await runtime.upstream.create_conversation_item(item=message)
        await runtime.upstream.create_response()

    async def approve_tool_call(
        self,
        session_id: str,
        call_id: str,
        *,
        always: bool = False,
    ):
        runtime = self.active_sessions.get(session_id)
        if runtime is None:
            return

        await runtime.tool_coordinator.approve_tool_call(call_id, always=always)

    async def reject_tool_call(
        self,
        session_id: str,
        call_id: str,
        *,
        always: bool = False,
    ):
        runtime = self.active_sessions.get(session_id)
        if runtime is None:
            return

        await runtime.tool_coordinator.reject_tool_call(call_id, always=always)

    async def interrupt(self, session_id: str) -> None:
        runtime = self.active_sessions.get(session_id)
        if runtime is None:
            return

        await runtime.upstream.cancel_response()
        await self._emit_to_frontend(session_id, {"type": "audio_interrupted"})

    async def _process_events(self, session_id: str):
        runtime = self.active_sessions.get(session_id)
        if runtime is None:
            return

        try:
            async for event in runtime.upstream.events():
                try:
                    event_type = getattr(event, "type", "unknown")
                    if (
                        "input_audio_transcription" in event_type
                        or event_type in {"input_audio_buffer.committed", "session.updated"}
                    ):
                        logger.info(
                            "Realtime event for session %s: %s",
                            session_id,
                            event_type,
                        )

                    frontend_events = runtime.event_mapper.map_event(event)
                    for frontend_event in frontend_events:
                        await self._emit_to_frontend(session_id, frontend_event)

                    if isinstance(event, ResponseDoneEvent):
                        await runtime.tool_coordinator.handle_response_done(event)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    event_type = getattr(event, "type", "unknown")
                    logger.error(
                        "Error processing event %s for session %s: %s",
                        event_type,
                        session_id,
                        exc,
                    )
                    await self._emit_to_frontend(
                        session_id,
                        {"type": "error", "error": f"Event {event_type}: {exc}"},
                    )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("Event stream error for session %s: %s", session_id, exc)
            await self._emit_to_frontend(
                session_id,
                {
                    "type": "error",
                    "error": str(exc),
                },
            )

    async def _emit_to_frontend(self, session_id: str, event: dict[str, Any]) -> None:
        runtime = self.active_sessions.get(session_id)
        if runtime is None:
            return

        await runtime.websocket.send_text(json.dumps(event))

    def _save_audio_buffer(self, session_id: str, audio_data: bytearray):
        if not audio_data:
            logger.info("No audio data to save for session %s.", session_id)
            return

        os.makedirs(AUDIO_SAVE_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(AUDIO_SAVE_DIR, f"{timestamp}.wav")

        try:
            with wave.open(filepath, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                wf.writeframes(bytes(audio_data))
            logger.info(
                "Saved %d bytes of audio to %s for session %s.",
                len(audio_data),
                filepath,
                session_id,
            )
        except Exception as exc:
            logger.error("Failed to save audio for session %s: %s", session_id, exc)


manager = RealtimeWebSocketManager()
