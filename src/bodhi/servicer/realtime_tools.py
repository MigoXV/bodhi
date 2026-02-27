from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from openai.types.realtime import (
    RealtimeConversationItemFunctionCall,
    RealtimeFunctionTool,
    ResponseDoneEvent,
)


TOOL_INSTRUCTIONS = """
You are an airline support assistant.
1. Use faq_lookup_tool for FAQ-style policy questions and avoid unsupported claims.
2. Use update_seat for seat changes and wait for approval if required.
3. Use get_weather for weather requests.
If a tool fails, explain the issue briefly and continue helping the user.
""".strip()


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    needs_approval: bool


@dataclass(frozen=True)
class PendingToolCall:
    call_id: str
    tool: str
    arguments: str


class RealtimeToolRegistry:
    def __init__(self):
        self._specs: dict[str, ToolSpec] = {
            "faq_lookup_tool": ToolSpec(
                name="faq_lookup_tool",
                description="Lookup frequently asked questions.",
                parameters={
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The user's FAQ question.",
                        }
                    },
                    "required": ["question"],
                    "additionalProperties": False,
                },
                needs_approval=False,
            ),
            "update_seat": ToolSpec(
                name="update_seat",
                description="Update a passenger seat by confirmation number.",
                parameters={
                    "type": "object",
                    "properties": {
                        "confirmation_number": {
                            "type": "string",
                            "description": "Passenger confirmation number.",
                        },
                        "new_seat": {
                            "type": "string",
                            "description": "New seat number, e.g. 12A.",
                        },
                    },
                    "required": ["confirmation_number", "new_seat"],
                    "additionalProperties": False,
                },
                needs_approval=True,
            ),
            "get_weather": ToolSpec(
                name="get_weather",
                description="Get weather information for a city.",
                parameters={
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "City name.",
                        }
                    },
                    "required": ["city"],
                    "additionalProperties": False,
                },
                needs_approval=False,
            ),
        }

    @property
    def instructions(self) -> str:
        return TOOL_INSTRUCTIONS

    def tool_definitions(self) -> list[RealtimeFunctionTool]:
        definitions: list[RealtimeFunctionTool] = []
        for spec in self._specs.values():
            definitions.append(
                RealtimeFunctionTool(
                    type="function",
                    name=spec.name,
                    description=spec.description,
                    parameters=spec.parameters,
                )
            )
        return definitions

    def needs_approval(self, name: str) -> bool:
        spec = self._specs.get(name)
        return bool(spec and spec.needs_approval)

    async def execute(self, *, name: str, arguments: str) -> str:
        parsed_args = _parse_arguments(arguments)

        if name == "faq_lookup_tool":
            question = _require_string(parsed_args, "question")
            return _faq_lookup_tool(question)

        if name == "update_seat":
            confirmation_number = _require_string(parsed_args, "confirmation_number")
            new_seat = _require_string(parsed_args, "new_seat")
            return _update_seat(confirmation_number, new_seat)

        if name == "get_weather":
            city = _require_string(parsed_args, "city")
            return _get_weather(city)

        raise ValueError(f"Unknown tool: {name}")


class RealtimeToolCoordinator:
    def __init__(
        self,
        *,
        registry: RealtimeToolRegistry,
        emit_event: Callable[[dict[str, Any]], Awaitable[None]],
        send_function_call_output: Callable[[str, str], Awaitable[None]],
        create_response: Callable[[], Awaitable[None]],
    ):
        self._registry = registry
        self._emit_event = emit_event
        self._send_function_call_output = send_function_call_output
        self._create_response = create_response
        self._pending_calls: dict[str, PendingToolCall] = {}

    async def handle_response_done(self, event: ResponseDoneEvent) -> None:
        outputs = event.response.output or []
        for output in outputs:
            if not isinstance(output, RealtimeConversationItemFunctionCall):
                continue

            call_id = output.call_id
            if not call_id:
                continue

            if self._registry.needs_approval(output.name):
                pending = PendingToolCall(
                    call_id=call_id,
                    tool=output.name,
                    arguments=output.arguments,
                )
                self._pending_calls[call_id] = pending
                await self._emit_event(
                    {
                        "type": "tool_approval_required",
                        "tool": pending.tool,
                        "call_id": pending.call_id,
                        "arguments": pending.arguments,
                    }
                )
                continue

            await self._execute_call(call_id=call_id, tool=output.name, arguments=output.arguments)

    async def approve_tool_call(self, call_id: str, *, always: bool = False) -> None:
        _ = always
        pending = self._pending_calls.pop(call_id, None)
        if pending is None:
            return

        await self._execute_call(
            call_id=pending.call_id,
            tool=pending.tool,
            arguments=pending.arguments,
        )

    async def reject_tool_call(self, call_id: str, *, always: bool = False) -> None:
        _ = always
        pending = self._pending_calls.pop(call_id, None)
        if pending is None:
            return

        output = f"Tool call '{pending.tool}' was rejected by user approval."
        await self._emit_event(
            {
                "type": "tool_end",
                "tool": pending.tool,
                "output": output,
            }
        )
        await self._send_function_call_output(pending.call_id, output)
        await self._create_response()

    async def _execute_call(self, *, call_id: str, tool: str, arguments: str) -> None:
        await self._emit_event({"type": "tool_start", "tool": tool})

        try:
            output = await self._registry.execute(name=tool, arguments=arguments)
        except Exception as exc:
            output = f"Tool '{tool}' failed: {exc}"

        await self._emit_event(
            {
                "type": "tool_end",
                "tool": tool,
                "output": output,
            }
        )
        await self._send_function_call_output(call_id, output)
        await self._create_response()


def _parse_arguments(arguments: str) -> dict[str, Any]:
    if not arguments:
        return {}

    payload = json.loads(arguments)
    if not isinstance(payload, dict):
        raise ValueError("Tool arguments must be a JSON object.")
    return payload


def _require_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing or invalid '{key}'.")
    return value.strip()


def _faq_lookup_tool(question: str) -> str:
    q = question.lower()
    if "wifi" in q or "wi-fi" in q:
        return "We have free wifi on the plane, join Airline-Wifi."
    if "bag" in q or "baggage" in q:
        return (
            "You may bring one carry-on bag under 50 pounds with max size "
            "22 x 14 x 9 inches."
        )
    if "seat" in q or "plane" in q:
        return (
            "There are 120 seats total: 22 business and 98 economy. "
            "Exit rows are 4 and 16. Rows 5-8 are Economy Plus."
        )
    return "I'm sorry, I don't know the answer to that question."


def _update_seat(confirmation_number: str, new_seat: str) -> str:
    return (
        f"Updated seat to {new_seat} for confirmation number {confirmation_number}."
    )


def _get_weather(city: str) -> str:
    return f"The weather in {city} is sunny."
