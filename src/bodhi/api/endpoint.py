from typing import Annotated

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from bodhi.servicer.realtime_manager import RealtimeWebSocketManager, manager
from bodhi.servicer.servicer import SessionMessageService

router = APIRouter()

_message_service = SessionMessageService(manager)


def get_realtime_manager() -> RealtimeWebSocketManager:
    return manager


def get_message_service() -> SessionMessageService:
    return _message_service


@router.websocket("/ws/{session_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    session_id: str,
    realtime_manager: Annotated[
        RealtimeWebSocketManager,
        Depends(get_realtime_manager),
    ],
    message_service: Annotated[
        SessionMessageService,
        Depends(get_message_service),
    ],
):
    await realtime_manager.connect(websocket, session_id)

    try:
        while True:
            payload = await websocket.receive_text()
            events = await message_service.handle_payload(session_id, payload)
            for event in events:
                await websocket.send_text(event.model_dump_json())
    except WebSocketDisconnect:
        pass
    finally:
        message_service.cleanup_session(session_id)
        await realtime_manager.disconnect(session_id)


@router.get("/")
async def read_index():
    return FileResponse("static/index.html")
