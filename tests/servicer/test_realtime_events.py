import pytest

from openai.types.realtime import (
    ConversationItemAdded,
    ConversationItemDone,
    ConversationItemInputAudioTranscriptionCompletedEvent,
    InputAudioBufferTimeoutTriggered,
    RealtimeError,
    RealtimeErrorEvent,
    RealtimeConversationItemUserMessage,
    ResponseAudioDeltaEvent,
    ResponseAudioTranscriptDeltaEvent,
    ResponseAudioTranscriptDoneEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
)

from bodhi.servicer.realtime_events import RealtimeEventMapper


@pytest.mark.parametrize(
    ("event", "expected"),
    [
        (
            ResponseAudioDeltaEvent(
                type="response.output_audio.delta",
                event_id="e1",
                response_id="r1",
                item_id="i1",
                output_index=0,
                content_index=0,
                delta="QUJD",
            ),
            {"type": "audio", "audio": "QUJD"},
        ),
        (
            InputAudioBufferTimeoutTriggered(
                type="input_audio_buffer.timeout_triggered",
                event_id="e2",
                item_id="i2",
                audio_start_ms=0,
                audio_end_ms=1200,
            ),
            {"type": "input_audio_timeout_triggered"},
        ),
        (
            RealtimeErrorEvent(
                type="error",
                event_id="e3",
                error=RealtimeError(type="server_error", message="boom"),
            ),
            {"type": "error", "error": "boom"},
        ),
    ],
)
def test_basic_event_mapping(event, expected):
    mapper = RealtimeEventMapper()

    mapped = mapper.map_event(event)

    assert mapped == [expected]


def test_history_mapping_uses_item_id_and_strips_audio_payloads():
    mapper = RealtimeEventMapper()

    item = RealtimeConversationItemUserMessage(
        type="message",
        role="user",
        id="msg-1",
        content=[
            {
                "type": "input_audio",
                "audio": "AAABBB",
                "transcript": "hello",
            }
        ],
    )

    added = ConversationItemAdded(
        type="conversation.item.added",
        event_id="e1",
        item=item,
        previous_item_id=None,
    )
    done = ConversationItemDone(
        type="conversation.item.done",
        event_id="e2",
        item=item,
        previous_item_id=None,
    )

    added_payload = mapper.map_event(added)[0]
    done_payload = mapper.map_event(done)[0]

    assert added_payload["type"] == "history_added"
    assert added_payload["item"]["item_id"] == "msg-1"
    assert "id" not in added_payload["item"]
    assert "audio" not in added_payload["item"]["content"][0]
    assert added_payload["item"]["content"][0]["transcript"] == "hello"

    assert done_payload["type"] == "history_updated"
    assert done_payload["history"][0]["item_id"] == "msg-1"
    assert "audio" not in done_payload["history"][0]["content"][0]


def test_delta_and_done_events_use_sdk_type_names():
    mapper = RealtimeEventMapper()

    text_delta = ResponseTextDeltaEvent(
        type="response.output_text.delta",
        event_id="e1",
        response_id="r1",
        item_id="i1",
        output_index=0,
        content_index=0,
        delta="he",
    )
    transcript_delta = ResponseAudioTranscriptDeltaEvent(
        type="response.output_audio_transcript.delta",
        event_id="e2",
        response_id="r1",
        item_id="i1",
        output_index=0,
        content_index=0,
        delta="llo",
    )
    text_done = ResponseTextDoneEvent(
        type="response.output_text.done",
        event_id="e3",
        response_id="r1",
        item_id="i1",
        output_index=0,
        content_index=0,
        text="hello",
    )
    transcript_done = ResponseAudioTranscriptDoneEvent(
        type="response.output_audio_transcript.done",
        event_id="e4",
        response_id="r1",
        item_id="i1",
        output_index=0,
        content_index=0,
        transcript="hello",
    )
    transcription_completed = ConversationItemInputAudioTranscriptionCompletedEvent(
        type="conversation.item.input_audio_transcription.completed",
        event_id="e5",
        item_id="u1",
        content_index=0,
        transcript="user text",
        usage={"type": "duration", "seconds": 1.2},
    )

    mapped = [
        mapper.map_event(text_delta)[0],
        mapper.map_event(transcript_delta)[0],
        mapper.map_event(text_done)[0],
        mapper.map_event(transcript_done)[0],
        mapper.map_event(transcription_completed)[0],
    ]

    # Events now use their native SDK type directly (no raw_model_event wrapper)
    assert mapped[0]["type"] == "response.output_text.delta"
    assert mapped[0]["text_delta"] == "he"
    assert "raw_model_event" not in mapped[0]

    assert mapped[1]["type"] == "response.output_audio_transcript.delta"
    assert mapped[1]["audio_transcript_delta"] == "llo"
    assert "raw_model_event" not in mapped[1]

    assert mapped[2]["type"] == "response.output_text.done"
    assert mapped[2]["text"] == "hello"
    assert "raw_model_event" not in mapped[2]

    assert mapped[3]["type"] == "response.output_audio_transcript.done"
    assert mapped[3]["audio_transcript"] == "hello"
    assert "raw_model_event" not in mapped[3]

    assert mapped[4]["type"] == "conversation.item.input_audio_transcription.completed"
    assert mapped[4]["transcription"] == "user text"
    assert mapped[4]["item_id"] == "u1"
    assert "raw_model_event" not in mapped[4]
