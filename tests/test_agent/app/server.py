import asyncio
import base64
import json
import logging
import os
import struct
import wave
from contextlib import asynccontextmanager
from datetime import datetime
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from typing_extensions import assert_never

from agents.realtime import RealtimeRunner, RealtimeSession, RealtimeSessionEvent
from agents.realtime.config import RealtimeUserInputMessage
from agents.realtime.items import RealtimeItem
from agents.realtime.model import RealtimeModelConfig
from agents.realtime.model_inputs import RealtimeModelSendRawMessage

# Import TwilioHandler class - handle both module and package use cases
if TYPE_CHECKING:
    # For type checking, use the relative import
    from .agent import get_starting_agent
else:
    # At runtime, try both import styles
    try:
        # Try relative import first (when used as a package)
        from .agent import get_starting_agent
    except ImportError:
        # Fall back to direct import (when run as a script)
        from agent import get_starting_agent


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


AUDIO_SAVE_DIR = "data-bin/agent/test01"


class RealtimeWebSocketManager:
    def __init__(self):
        self.active_sessions: dict[str, RealtimeSession] = {}
        self.session_contexts: dict[str, Any] = {}
        self.websockets: dict[str, WebSocket] = {}
        self.audio_buffers: dict[str, bytearray] = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.websockets[session_id] = websocket
        self.audio_buffers[session_id] = bytearray()

        agent = get_starting_agent()
        runner = RealtimeRunner(agent)
        # If you want to customize the runner behavior, you can pass options:
        # runner_config = RealtimeRunConfig(async_tool_calls=False)
        # runner = RealtimeRunner(agent, config=runner_config)
        model_config: RealtimeModelConfig = {
            "initial_model_settings": {
                "turn_detection": {
                    "type": "server_vad",
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 500,
                    "interrupt_response": True,
                    "create_response": True,
                },
                "input_audio_transcription": {
                    "model": "whisper-1",
                },
            },
        }
        session_context = await runner.run(model_config=model_config)
        session = await session_context.__aenter__()
        self.active_sessions[session_id] = session
        self.session_contexts[session_id] = session_context

        # 注册原始事件监听器来捕获delta事件
        self._register_raw_event_listener(session_id, session.model)
        
        # 同时监听高级事件流
        # Start event processing task
        asyncio.create_task(self._process_events(session_id))

    async def disconnect(self, session_id: str):
        # Save accumulated audio to WAV file before cleanup
        self._save_audio_buffer(session_id)

        if session_id in self.session_contexts:
            await self.session_contexts[session_id].__aexit__(None, None, None)
            del self.session_contexts[session_id]
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
        if session_id in self.websockets:
            del self.websockets[session_id]
        if session_id in self.audio_buffers:
            del self.audio_buffers[session_id]

    async def send_audio(self, session_id: str, audio_bytes: bytes):
        if session_id in self.active_sessions:
            # Accumulate audio data for later saving
            if session_id in self.audio_buffers:
                self.audio_buffers[session_id].extend(audio_bytes)
            await self.active_sessions[session_id].send_audio(audio_bytes)

    async def send_client_event(self, session_id: str, event: dict[str, Any]):
        """Send a raw client event to the underlying realtime model."""
        session = self.active_sessions.get(session_id)
        if not session:
            return
        await session.model.send_event(
            RealtimeModelSendRawMessage(
                message={
                    "type": event["type"],
                    "other_data": {k: v for k, v in event.items() if k != "type"},
                }
            )
        )

    async def send_user_message(self, session_id: str, message: RealtimeUserInputMessage):
        """Send a structured user message via the higher-level API (supports input_image)."""
        session = self.active_sessions.get(session_id)
        if not session:
            return
        await session.send_message(message)  # delegates to RealtimeModelSendUserInput path

    async def approve_tool_call(self, session_id: str, call_id: str, *, always: bool = False):
        """Approve a pending tool call for a session."""
        session = self.active_sessions.get(session_id)
        if not session:
            return
        await session.approve_tool_call(call_id, always=always)

    async def reject_tool_call(self, session_id: str, call_id: str, *, always: bool = False):
        """Reject a pending tool call for a session."""
        session = self.active_sessions.get(session_id)
        if not session:
            return
        await session.reject_tool_call(call_id, always=always)

    async def interrupt(self, session_id: str) -> None:
        """Interrupt current model playback/response for a session."""
        session = self.active_sessions.get(session_id)
        if not session:
            return
        await session.interrupt()

    def _save_audio_buffer(self, session_id: str):
        """Save accumulated audio buffer as a WAV file."""
        audio_data = self.audio_buffers.get(session_id)
        if not audio_data or len(audio_data) == 0:
            logger.info("No audio data to save for session %s.", session_id)
            return

        os.makedirs(AUDIO_SAVE_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(AUDIO_SAVE_DIR, f"{timestamp}.wav")

        try:
            with wave.open(filepath, "wb") as wf:
                wf.setnchannels(1)        # mono
                wf.setsampwidth(2)        # 16-bit (2 bytes)
                wf.setframerate(24000)    # 24kHz (OpenAI Realtime API default)
                wf.writeframes(bytes(audio_data))
            logger.info(
                "Saved %d bytes of audio to %s for session %s.",
                len(audio_data), filepath, session_id,
            )
        except Exception as e:
            logger.error("Failed to save audio for session %s: %s", session_id, e)

    async def _process_events(self, session_id: str):
        """处理高级封装事件流（history_updated, tool_start等）"""
        try:
            session = self.active_sessions[session_id]
            websocket = self.websockets[session_id]

            async for event in session:
                event_data = await self._serialize_event(event)
                await websocket.send_text(json.dumps(event_data))
        except Exception as e:
            print(e)
            logger.error(f"Error processing events for session {session_id}: {e}")

    def _register_raw_event_listener(self, session_id: str, model):
        """注册监听器以捕获原始model事件（delta等）"""
        logger.info("Registering raw event listener for session %s", session_id)
        
        def event_listener(event):
            """处理原始model事件的回调函数"""
            try:
                # 这是同步回调，需要用asyncio在事件循环中执行
                asyncio.create_task(self._handle_raw_model_event(session_id, event))
            except Exception as e:
                logger.error("Error in event listener: %s", e)
        
        # 添加监听器到model
        model.add_listener(event_listener)
        logger.info("Successfully registered raw event listener for session %s", session_id)
    
    async def _handle_raw_model_event(self, session_id: str, raw_event):
        """处理单个原始model事件"""
        try:
            websocket = self.websockets.get(session_id)
            if not websocket:
                return
                
            # 检查事件类型
            event_type = getattr(raw_event, 'type', None) or (raw_event.get('type') if isinstance(raw_event, dict) else None)
            if not event_type:
                return
            
            # 只处理我们关心的delta和transcription事件
            if event_type not in [
                "response.text.delta",
                "response.output_text.delta",
                "response.audio_transcript.delta",
                "response.output_audio_transcript.delta",
                "conversation.item.input_audio_transcription.completed",
                "conversation.item.input_audio_transcription.delta",
                "response.text.done",
                "response.output_text.done",
                "response.audio_transcript.done",
                "response.output_audio_transcript.done",
            ]:
                return
            
            base_event: dict[str, Any] = {
                "type": event_type,
            }
            
            # 将事件对象转换为字典以访问属性
            if hasattr(raw_event, '__dict__'):
                event_dict = raw_event.__dict__
            elif isinstance(raw_event, dict):
                event_dict = raw_event
            else:
                logger.warning("Unknown event format: %s", type(raw_event))
                return
            
            # 语音输入转写完成
            if event_type == "conversation.item.input_audio_transcription.completed":
                base_event["transcription"] = event_dict.get("transcript", "")
                base_event["item_id"] = event_dict.get("item_id")
            
            # 语音输入转写delta（流式）
            elif event_type == "conversation.item.input_audio_transcription.delta":
                base_event["delta"] = event_dict.get("delta", "")
                base_event["item_id"] = event_dict.get("item_id")
            
            # 文本delta（打字机效果）
            elif event_type in ("response.text.delta", "response.output_text.delta"):
                base_event["text_delta"] = event_dict.get("delta", "")
                base_event["response_id"] = event_dict.get("response_id")
                base_event["item_id"] = event_dict.get("item_id")
                base_event["output_index"] = event_dict.get("output_index")
                base_event["content_index"] = event_dict.get("content_index")
            
            # 音频转写delta（打字机效果）
            elif event_type in ("response.audio_transcript.delta", "response.output_audio_transcript.delta"):
                base_event["audio_transcript_delta"] = event_dict.get("delta", "")
                base_event["response_id"] = event_dict.get("response_id")
                base_event["item_id"] = event_dict.get("item_id")
                base_event["output_index"] = event_dict.get("output_index")
                base_event["content_index"] = event_dict.get("content_index")
            
            # 文本完成
            elif event_type in ("response.text.done", "response.output_text.done"):
                base_event["text"] = event_dict.get("text", "")
                base_event["response_id"] = event_dict.get("response_id")
                base_event["item_id"] = event_dict.get("item_id")
                base_event["output_index"] = event_dict.get("output_index")
                base_event["content_index"] = event_dict.get("content_index")
            
            # 音频转写完成
            elif event_type in ("response.audio_transcript.done", "response.output_audio_transcript.done"):
                base_event["audio_transcript"] = event_dict.get("transcript", "")
                base_event["response_id"] = event_dict.get("response_id")
                base_event["item_id"] = event_dict.get("item_id")
                base_event["output_index"] = event_dict.get("output_index")
                base_event["content_index"] = event_dict.get("content_index")
            
            await websocket.send_text(json.dumps(base_event))
            logger.debug("Sent raw event: %s", event_type)
                    
        except Exception as e:
            logger.error("Error handling raw model event: %s", e)

    def _sanitize_history_item(self, item: RealtimeItem) -> dict[str, Any]:
        """Remove large binary payloads from history items while keeping transcripts."""
        item_dict = item.model_dump()
        content = item_dict.get("content")
        if isinstance(content, list):
            sanitized_content: list[Any] = []
            for part in content:
                if isinstance(part, dict):
                    sanitized_part = part.copy()
                    if sanitized_part.get("type") in {"audio", "input_audio"}:
                        sanitized_part.pop("audio", None)
                    sanitized_content.append(sanitized_part)
                else:
                    sanitized_content.append(part)
            item_dict["content"] = sanitized_content
        return item_dict

    async def _serialize_event(self, event: RealtimeSessionEvent) -> dict[str, Any]:
        base_event: dict[str, Any] = {
            "type": event.type,
        }

        if event.type == "agent_start":
            base_event["agent"] = event.agent.name
        elif event.type == "agent_end":
            base_event["agent"] = event.agent.name
        elif event.type == "handoff":
            base_event["from"] = event.from_agent.name
            base_event["to"] = event.to_agent.name
        elif event.type == "tool_start":
            base_event["tool"] = event.tool.name
        elif event.type == "tool_end":
            base_event["tool"] = event.tool.name
            base_event["output"] = str(event.output)
        elif event.type == "tool_approval_required":
            base_event["tool"] = event.tool.name
            base_event["call_id"] = event.call_id
            base_event["arguments"] = event.arguments
            base_event["agent"] = event.agent.name
        elif event.type == "audio":
            base_event["audio"] = base64.b64encode(event.audio.data).decode("utf-8")
        elif event.type == "audio_interrupted":
            pass
        elif event.type == "audio_end":
            pass
        elif event.type == "history_updated":
            base_event["history"] = [self._sanitize_history_item(item) for item in event.history]
        elif event.type == "history_added":
            # Provide the added item so the UI can render incrementally.
            try:
                base_event["item"] = self._sanitize_history_item(event.item)
            except Exception:
                base_event["item"] = None
        elif event.type == "guardrail_tripped":
            base_event["guardrail_results"] = [
                {"name": result.guardrail.name} for result in event.guardrail_results
            ]
        elif event.type == "raw_model_event":
            raw_type = event.data.type
            base_event["type"] = raw_type  # Use actual event type directly
            
            # 处理语音转写完成事件
            if raw_type == "conversation.item.input_audio_transcription.completed":
                base_event["transcription"] = event.data.transcript
                base_event["item_id"] = event.data.item_id
            
            # 处理语音输入转写delta
            elif raw_type == "conversation.item.input_audio_transcription.delta":
                base_event["delta"] = getattr(event.data, 'delta', '') or ''
                base_event["item_id"] = event.data.item_id
            
            # 处理文本delta事件（打字机效果）
            elif raw_type in ("response.text.delta", "response.output_text.delta"):
                base_event["text_delta"] = event.data.delta
                base_event["response_id"] = event.data.response_id
                base_event["item_id"] = event.data.item_id
                base_event["output_index"] = event.data.output_index
                base_event["content_index"] = event.data.content_index
            
            # 处理音频转写delta事件
            elif raw_type in ("response.audio_transcript.delta", "response.output_audio_transcript.delta"):
                base_event["audio_transcript_delta"] = event.data.delta
                base_event["response_id"] = event.data.response_id
                base_event["item_id"] = event.data.item_id
                base_event["output_index"] = event.data.output_index
                base_event["content_index"] = event.data.content_index
            
            # 处理文本完成事件
            elif raw_type in ("response.text.done", "response.output_text.done"):
                base_event["text"] = event.data.text
                base_event["response_id"] = event.data.response_id
                base_event["item_id"] = event.data.item_id
                base_event["output_index"] = event.data.output_index
                base_event["content_index"] = event.data.content_index
            
            # 处理音频转写完成事件
            elif raw_type in ("response.audio_transcript.done", "response.output_audio_transcript.done"):
                base_event["audio_transcript"] = event.data.transcript
                base_event["response_id"] = event.data.response_id
                base_event["item_id"] = event.data.item_id
                base_event["output_index"] = event.data.output_index
                base_event["content_index"] = event.data.content_index
        elif event.type == "error":
            base_event["error"] = str(event.error) if hasattr(event, "error") else "Unknown error"
        elif event.type == "input_audio_timeout_triggered":
            pass
        else:
            assert_never(event)

        return base_event


manager = RealtimeWebSocketManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(lifespan=lifespan)


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await manager.connect(websocket, session_id)
    image_buffers: dict[str, dict[str, Any]] = {}
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            if message["type"] == "audio":
                # Convert int16 array to bytes
                int16_data = message["data"]
                audio_bytes = struct.pack(f"{len(int16_data)}h", *int16_data)
                await manager.send_audio(session_id, audio_bytes)
            elif message["type"] == "text":
                text = (message.get("text") or "").strip()
                if not text:
                    await websocket.send_text(
                        json.dumps({"type": "error", "error": "Empty text message."})
                    )
                    continue

                user_text_msg: RealtimeUserInputMessage = {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}],
                }
                await manager.send_user_message(session_id, user_text_msg)
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "client_info",
                            "info": "text_enqueued",
                            "size": len(text),
                        }
                    )
                )
            elif message["type"] == "image":
                logger.info("Received image message from client (session %s).", session_id)
                # Build a conversation.item.create with input_image (and optional input_text)
                data_url = message.get("data_url")
                prompt_text = message.get("text") or "Please describe this image."
                if data_url:
                    logger.info(
                        "Forwarding image (structured message) to Realtime API (len=%d).",
                        len(data_url),
                    )
                    user_msg: RealtimeUserInputMessage = {
                        "type": "message",
                        "role": "user",
                        "content": (
                            [
                                {"type": "input_image", "image_url": data_url, "detail": "high"},
                                {"type": "input_text", "text": prompt_text},
                            ]
                            if prompt_text
                            else [{"type": "input_image", "image_url": data_url, "detail": "high"}]
                        ),
                    }
                    await manager.send_user_message(session_id, user_msg)
                    # Acknowledge to client UI
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "client_info",
                                "info": "image_enqueued",
                                "size": len(data_url),
                            }
                        )
                    )
                else:
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "error",
                                "error": "No data_url for image message.",
                            }
                        )
                    )
            elif message["type"] == "commit_audio":
                # Force close the current input audio turn
                await manager.send_client_event(session_id, {"type": "input_audio_buffer.commit"})
            elif message["type"] == "image_start":
                img_id = str(message.get("id"))
                image_buffers[img_id] = {
                    "text": message.get("text") or "Please describe this image.",
                    "chunks": [],
                }
                await websocket.send_text(
                    json.dumps({"type": "client_info", "info": "image_start_ack", "id": img_id})
                )
            elif message["type"] == "image_chunk":
                img_id = str(message.get("id"))
                chunk = message.get("chunk", "")
                if img_id in image_buffers:
                    image_buffers[img_id]["chunks"].append(chunk)
                    if len(image_buffers[img_id]["chunks"]) % 10 == 0:
                        await websocket.send_text(
                            json.dumps(
                                {
                                    "type": "client_info",
                                    "info": "image_chunk_ack",
                                    "id": img_id,
                                    "count": len(image_buffers[img_id]["chunks"]),
                                }
                            )
                        )
            elif message["type"] == "image_end":
                img_id = str(message.get("id"))
                buf = image_buffers.pop(img_id, None)
                if buf is None:
                    await websocket.send_text(
                        json.dumps({"type": "error", "error": "Unknown image id for image_end."})
                    )
                else:
                    data_url = "".join(buf["chunks"]) if buf["chunks"] else None
                    prompt_text = buf["text"]
                    if data_url:
                        logger.info(
                            "Forwarding chunked image (structured message) to Realtime API (len=%d).",
                            len(data_url),
                        )
                        user_msg2: RealtimeUserInputMessage = {
                            "type": "message",
                            "role": "user",
                            "content": (
                                [
                                    {
                                        "type": "input_image",
                                        "image_url": data_url,
                                        "detail": "high",
                                    },
                                    {"type": "input_text", "text": prompt_text},
                                ]
                                if prompt_text
                                else [
                                    {"type": "input_image", "image_url": data_url, "detail": "high"}
                                ]
                            ),
                        }
                        await manager.send_user_message(session_id, user_msg2)
                        await websocket.send_text(
                            json.dumps(
                                {
                                    "type": "client_info",
                                    "info": "image_enqueued",
                                    "id": img_id,
                                    "size": len(data_url),
                                }
                            )
                        )
                    else:
                        await websocket.send_text(
                            json.dumps({"type": "error", "error": "Empty image."})
                        )
            elif message["type"] == "tool_approval_decision":
                call_id = message.get("call_id")
                approve = bool(message.get("approve"))
                always = bool(message.get("always", False))
                if not call_id:
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "error",
                                "error": "Missing call_id for tool approval decision.",
                            }
                        )
                    )
                    continue
                if approve:
                    await manager.approve_tool_call(session_id, call_id, always=always)
                else:
                    await manager.reject_tool_call(session_id, call_id, always=always)
            elif message["type"] == "interrupt":
                await manager.interrupt(session_id)

    except WebSocketDisconnect:
        await manager.disconnect(session_id)


app.mount("/", StaticFiles(directory="tests/test_agent/app/static", html=True), name="static")


@app.get("/")
async def read_index():
    return FileResponse("tests/test_agent/app/static/index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        # Increased WebSocket frame size to comfortably handle image data URLs.
        ws_max_size=16 * 1024 * 1024,
    )
