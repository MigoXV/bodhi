import json

from pydantic import ValidationError

from bodhi.servicer.handler import ClientEventHandlers, RealtimeManagerProtocol
from bodhi.types.servicer import ClientEventAdapter, ErrorEnvelope, ServerEnvelope


class SessionMessageService:
    def __init__(
        self,
        realtime_manager: RealtimeManagerProtocol,
        handlers: ClientEventHandlers | None = None,
    ):
        self.handlers = handlers or ClientEventHandlers(realtime_manager)

    async def handle_payload(
        self, session_id: str, payload_text: str
    ) -> list[ServerEnvelope]:
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError:
            return [ErrorEnvelope(error="Invalid JSON payload.")]

        try:
            event = ClientEventAdapter.validate_python(payload)
        except ValidationError:
            return [ErrorEnvelope(error="Invalid client event.")]

        handler = self.handlers.handlers.get(event.type)
        if handler is None:
            return [ErrorEnvelope(error="Invalid client event.")]

        return await handler(session_id, event)

    def cleanup_session(self, session_id: str) -> None:
        self.handlers.cleanup_session(session_id)
