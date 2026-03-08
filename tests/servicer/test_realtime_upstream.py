from types import SimpleNamespace

import pytest

from bodhi.servicer.realtime_upstream import (
    DEFAULT_AUDIO_SAMPLE_RATE,
    OpenAIRealtimeUpstream,
    RealtimeUpstreamConfig,
)


class FakeSessionAPI:
    def __init__(self) -> None:
        self.updated_sessions: list[object] = []

    async def update(self, *, session) -> None:
        self.updated_sessions.append(session)


@pytest.mark.asyncio
async def test_configure_session_sets_fixed_audio_sample_rate():
    upstream = object.__new__(OpenAIRealtimeUpstream)
    upstream.config = RealtimeUpstreamConfig(
        model_name="gpt-realtime",
        transcription_model="whisper-1",
        voice="alloy",
        api_key=None,
        base_url=None,
        websocket_base_url=None,
    )
    session_api = FakeSessionAPI()
    upstream._connection = SimpleNamespace(session=session_api)

    await upstream.configure_session(tools=[], instructions="test")

    assert len(session_api.updated_sessions) == 1
    session = session_api.updated_sessions[0]
    payload = session.model_dump(exclude_none=True)

    assert payload["audio"]["input"]["format"] == {
        "type": "audio/pcm",
        "rate": DEFAULT_AUDIO_SAMPLE_RATE,
    }
    assert payload["audio"]["output"]["format"] == {
        "type": "audio/pcm",
        "rate": DEFAULT_AUDIO_SAMPLE_RATE,
    }
    assert payload["audio"]["output"]["voice"] == "alloy"


@pytest.mark.asyncio
async def test_configure_session_overrides_voice():
    upstream = object.__new__(OpenAIRealtimeUpstream)
    upstream.config = RealtimeUpstreamConfig(
        model_name="gpt-realtime",
        transcription_model="whisper-1",
        voice="alloy",
        api_key=None,
        base_url=None,
        websocket_base_url=None,
    )
    session_api = FakeSessionAPI()
    upstream._connection = SimpleNamespace(session=session_api)

    await upstream.configure_session(tools=[], instructions="test", voice="rita")

    payload = session_api.updated_sessions[0].model_dump(exclude_none=True)
    assert payload["audio"]["output"]["voice"] == "rita"
