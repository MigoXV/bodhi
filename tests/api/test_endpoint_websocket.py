from fastapi import FastAPI
from fastapi.testclient import TestClient
from openai.types.realtime import RealtimeConversationItemUserMessage

from bodhi.api import endpoint
from bodhi.api.endpoint import get_message_service, get_realtime_manager, router
from bodhi.servicer.servicer import SessionMessageService


class FakeRealtimeManager:
    def __init__(self):
        self.connected: list[str] = []
        self.disconnected: list[str] = []
        self.user_messages: list[tuple[str, RealtimeConversationItemUserMessage]] = []

    async def connect(self, websocket, session_id: str):
        await websocket.accept()
        self.connected.append(session_id)

    async def disconnect(self, session_id: str):
        self.disconnected.append(session_id)

    async def send_audio(self, session_id: str, audio_bytes: bytes) -> None:
        return None

    async def send_user_message(
        self,
        session_id: str,
        message: RealtimeConversationItemUserMessage,
    ) -> None:
        self.user_messages.append((session_id, message))

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


def test_websocket_endpoint_routes_to_service_and_disconnect_cleanup():
    app = FastAPI()
    app.include_router(router)

    manager = FakeRealtimeManager()
    service = SessionMessageService(manager)

    app.dependency_overrides[get_realtime_manager] = lambda: manager
    app.dependency_overrides[get_message_service] = lambda: service

    with TestClient(app) as client:
        with client.websocket_connect("/ws/session-a") as websocket:
            websocket.send_text("{")
            invalid_json_response = websocket.receive_json()
            assert invalid_json_response == {
                "type": "error",
                "error": "Invalid JSON payload.",
            }

            websocket.send_json({"type": "text", "text": "hello"})
            text_response = websocket.receive_json()
            assert text_response == {
                "type": "client_info",
                "info": "text_enqueued",
                "id": None,
                "size": 5,
                "count": None,
            }

            websocket.send_json({"type": "image_start", "id": "img-1", "text": "x"})
            start_response = websocket.receive_json()
            assert start_response["type"] == "client_info"
            assert start_response["info"] == "image_start_ack"
            websocket.close()

    assert manager.connected == ["session-a"]
    assert manager.disconnected == ["session-a"]
    assert "session-a" not in service.handlers.image_buffers
    assert len(manager.user_messages) == 1


def test_config_endpoint_returns_default_voice(monkeypatch):
    app = FastAPI()
    app.include_router(router)

    monkeypatch.setattr(
        endpoint,
        "load_realtime_config_from_env",
        lambda: type("Config", (), {"voice": "paimon"})(),
    )

    with TestClient(app) as client:
        response = client.get("/config")

    assert response.status_code == 200
    assert response.json() == {"default_voice": "paimon"}
