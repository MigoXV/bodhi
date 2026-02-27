from __future__ import annotations

import logging
from typing import Any

from openai.types.realtime import (
    ConversationItem,
    ConversationItemAdded,
    ConversationItemCreatedEvent,
    ConversationItemDone,
    ConversationItemInputAudioTranscriptionCompletedEvent,
    ConversationItemInputAudioTranscriptionDeltaEvent,
    ConversationItemInputAudioTranscriptionFailedEvent,
    ConversationItemInputAudioTranscriptionSegment,
    InputAudioBufferCommittedEvent,
    InputAudioBufferSpeechStartedEvent,
    InputAudioBufferSpeechStoppedEvent,
    InputAudioBufferTimeoutTriggered,
    RealtimeErrorEvent,
    RealtimeServerEvent,
    ResponseAudioDeltaEvent,
    ResponseAudioTranscriptDeltaEvent,
    ResponseAudioTranscriptDoneEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
    SessionCreatedEvent,
    SessionUpdatedEvent,
)

logger = logging.getLogger(__name__)


class RealtimeEventMapper:
    def __init__(self):
        self._history_items: dict[str, dict[str, Any]] = {}
        self._history_order: list[str] = []

    def map_event(self, event: RealtimeServerEvent) -> list[dict[str, Any]]:
        if isinstance(event, (SessionCreatedEvent, SessionUpdatedEvent)):
            session = event.session.model_dump(exclude_none=True)
            transcription_model = (
                session.get("audio", {})
                .get("input", {})
                .get("transcription", {})
                .get("model")
            )
            return [
                {
                    "type": event.type,
                    "session": session,
                    "transcription_model": transcription_model,
                }
            ]

        if isinstance(event, ResponseAudioDeltaEvent):
            return [{"type": "audio", "audio": event.delta}]

        if isinstance(event, InputAudioBufferTimeoutTriggered):
            return [{"type": "input_audio_timeout_triggered"}]

        if isinstance(event, InputAudioBufferCommittedEvent):
            return [
                {
                    "type": event.type,
                    "item_id": event.item_id,
                    "previous_item_id": event.previous_item_id,
                }
            ]

        if isinstance(event, InputAudioBufferSpeechStartedEvent):
            return [{"type": "audio_interrupted"}]

        if isinstance(event, InputAudioBufferSpeechStoppedEvent):
            return [{"type": event.type}]

        if isinstance(event, (ConversationItemAdded, ConversationItemCreatedEvent)):
            item = self._sanitize_history_item(event.item)
            self._upsert_history_item(item, event.previous_item_id)
            return [{"type": "history_added", "item": item}]

        if isinstance(event, ConversationItemDone):
            item = self._sanitize_history_item(event.item)
            self._upsert_history_item(item, event.previous_item_id)
            return [{"type": "history_updated", "history": self.history_snapshot()}]

        if isinstance(event, ConversationItemInputAudioTranscriptionCompletedEvent):
            return [
                {
                    "type": event.type,
                    "transcription": event.transcript,
                    "item_id": event.item_id,
                }
            ]

        if isinstance(event, ConversationItemInputAudioTranscriptionDeltaEvent):
            return [
                {
                    "type": event.type,
                    "delta": event.delta or "",
                    "item_id": event.item_id,
                }
            ]

        if isinstance(event, ConversationItemInputAudioTranscriptionSegment):
            return [
                {
                    "type": event.type,
                    "item_id": event.item_id,
                    "content_index": event.content_index,
                    "segment_id": event.id,
                    "speaker": event.speaker,
                    "start": event.start,
                    "end": event.end,
                    "text": event.text,
                }
            ]

        if isinstance(event, ConversationItemInputAudioTranscriptionFailedEvent):
            return [
                {
                    "type": event.type,
                    "item_id": event.item_id,
                    "content_index": event.content_index,
                    "error": event.error.message or "input audio transcription failed",
                    "error_code": event.error.code,
                    "error_type": event.error.type,
                    "error_param": event.error.param,
                }
            ]

        if isinstance(event, ResponseTextDeltaEvent):
            return [
                {
                    "type": event.type,
                    "text_delta": event.delta,
                    "response_id": event.response_id,
                    "item_id": event.item_id,
                    "output_index": event.output_index,
                    "content_index": event.content_index,
                }
            ]

        if isinstance(event, ResponseAudioTranscriptDeltaEvent):
            return [
                {
                    "type": event.type,
                    "audio_transcript_delta": event.delta,
                    "response_id": event.response_id,
                    "item_id": event.item_id,
                    "output_index": event.output_index,
                    "content_index": event.content_index,
                }
            ]

        if isinstance(event, ResponseTextDoneEvent):
            return [
                {
                    "type": event.type,
                    "text": event.text,
                    "response_id": event.response_id,
                    "item_id": event.item_id,
                    "output_index": event.output_index,
                    "content_index": event.content_index,
                }
            ]

        if isinstance(event, ResponseAudioTranscriptDoneEvent):
            return [
                {
                    "type": event.type,
                    "audio_transcript": event.transcript,
                    "response_id": event.response_id,
                    "item_id": event.item_id,
                    "output_index": event.output_index,
                    "content_index": event.content_index,
                }
            ]

        if isinstance(event, RealtimeErrorEvent):
            return [{"type": "error", "error": event.error.message}]

        # Catch-all: forward any unhandled upstream event so the frontend
        # raw-events panel can still display it with its original type.
        event_type = getattr(event, "type", None)
        if event_type:
            logger.debug("Forwarding unhandled upstream event: %s", event_type)
            return [{"type": event_type}]

        return []

    def history_snapshot(self) -> list[dict[str, Any]]:
        history: list[dict[str, Any]] = []
        for item_id in self._history_order:
            item = self._history_items.get(item_id)
            if item is not None:
                history.append(item)
        return history

    def _upsert_history_item(
        self,
        item: dict[str, Any],
        previous_item_id: str | None,
    ) -> None:
        item_id = item.get("item_id")
        if not isinstance(item_id, str) or not item_id:
            return

        self._history_items[item_id] = item
        if item_id in self._history_order:
            return

        if previous_item_id and previous_item_id in self._history_order:
            index = self._history_order.index(previous_item_id) + 1
            self._history_order.insert(index, item_id)
            return

        self._history_order.append(item_id)

    def _sanitize_history_item(self, item: ConversationItem) -> dict[str, Any]:
        payload = item.model_dump(exclude_none=True)
        item_id = payload.pop("id", None)
        if item_id:
            payload["item_id"] = item_id

        content = payload.get("content")
        if isinstance(content, list):
            sanitized_content: list[Any] = []
            for part in content:
                if not isinstance(part, dict):
                    sanitized_content.append(part)
                    continue

                sanitized_part = {
                    key: value
                    for key, value in part.items()
                    if key not in {"audio", "input_audio", "output_audio"}
                }
                sanitized_content.append(sanitized_part)
            payload["content"] = sanitized_content

        return payload
