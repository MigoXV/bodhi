import pytest

from openai.types.realtime import (
    RealtimeConversationItemFunctionCall,
    RealtimeResponse,
    ResponseDoneEvent,
)

from bodhi.servicer.realtime_tools import RealtimeToolCoordinator, RealtimeToolRegistry


class ToolHarness:
    def __init__(self):
        self.events: list[dict] = []
        self.outputs: list[tuple[str, str]] = []
        self.create_response_calls = 0

    async def emit_event(self, event: dict) -> None:
        self.events.append(event)

    async def send_function_call_output(self, call_id: str, output: str) -> None:
        self.outputs.append((call_id, output))

    async def create_response(self) -> None:
        self.create_response_calls += 1


def _response_done_with_call(
    *,
    call_id: str,
    name: str,
    arguments: str,
) -> ResponseDoneEvent:
    call_item = RealtimeConversationItemFunctionCall(
        type="function_call",
        id=f"item-{call_id}",
        call_id=call_id,
        name=name,
        arguments=arguments,
    )

    return ResponseDoneEvent(
        type="response.done",
        event_id=f"event-{call_id}",
        response=RealtimeResponse(
            id="resp-1",
            status="completed",
            output=[call_item],
        ),
    )


@pytest.mark.asyncio
async def test_tool_auto_execution_for_non_approval_tool():
    harness = ToolHarness()
    coordinator = RealtimeToolCoordinator(
        registry=RealtimeToolRegistry(),
        emit_event=harness.emit_event,
        send_function_call_output=harness.send_function_call_output,
        create_response=harness.create_response,
    )

    event = _response_done_with_call(
        call_id="call-weather",
        name="get_weather",
        arguments='{"city":"Shanghai"}',
    )
    await coordinator.handle_response_done(event)

    assert harness.events[0] == {"type": "tool_start", "tool": "get_weather"}
    assert harness.events[1]["type"] == "tool_end"
    assert harness.events[1]["tool"] == "get_weather"
    assert "Shanghai" in harness.events[1]["output"]
    assert harness.outputs == [("call-weather", harness.events[1]["output"])]
    assert harness.create_response_calls == 1


@pytest.mark.asyncio
async def test_tool_requires_approval_then_approve_path_executes():
    harness = ToolHarness()
    coordinator = RealtimeToolCoordinator(
        registry=RealtimeToolRegistry(),
        emit_event=harness.emit_event,
        send_function_call_output=harness.send_function_call_output,
        create_response=harness.create_response,
    )

    event = _response_done_with_call(
        call_id="call-seat",
        name="update_seat",
        arguments='{"confirmation_number":"ABC123","new_seat":"12A"}',
    )
    await coordinator.handle_response_done(event)

    assert harness.events == [
        {
            "type": "tool_approval_required",
            "tool": "update_seat",
            "call_id": "call-seat",
            "arguments": '{"confirmation_number":"ABC123","new_seat":"12A"}',
        }
    ]
    assert harness.outputs == []
    assert harness.create_response_calls == 0

    await coordinator.approve_tool_call("call-seat", always=True)

    assert harness.events[1] == {"type": "tool_start", "tool": "update_seat"}
    assert harness.events[2]["type"] == "tool_end"
    assert "Updated seat to 12A" in harness.events[2]["output"]
    assert harness.outputs == [("call-seat", harness.events[2]["output"])]
    assert harness.create_response_calls == 1


@pytest.mark.asyncio
async def test_tool_reject_path_returns_function_output_and_resumes_model():
    harness = ToolHarness()
    coordinator = RealtimeToolCoordinator(
        registry=RealtimeToolRegistry(),
        emit_event=harness.emit_event,
        send_function_call_output=harness.send_function_call_output,
        create_response=harness.create_response,
    )

    event = _response_done_with_call(
        call_id="call-reject",
        name="update_seat",
        arguments='{"confirmation_number":"ABC123","new_seat":"12A"}',
    )
    await coordinator.handle_response_done(event)
    await coordinator.reject_tool_call("call-reject", always=False)

    assert harness.events[1]["type"] == "tool_end"
    assert "rejected" in harness.events[1]["output"].lower()
    assert harness.outputs == [("call-reject", harness.events[1]["output"])]
    assert harness.create_response_calls == 1


@pytest.mark.asyncio
async def test_unknown_or_invalid_tool_calls_fail_gracefully_and_resume_response():
    harness = ToolHarness()
    coordinator = RealtimeToolCoordinator(
        registry=RealtimeToolRegistry(),
        emit_event=harness.emit_event,
        send_function_call_output=harness.send_function_call_output,
        create_response=harness.create_response,
    )

    unknown_event = _response_done_with_call(
        call_id="call-unknown",
        name="not_exists",
        arguments="{}",
    )
    invalid_args_event = _response_done_with_call(
        call_id="call-invalid",
        name="faq_lookup_tool",
        arguments="{}",
    )

    await coordinator.handle_response_done(unknown_event)
    await coordinator.handle_response_done(invalid_args_event)

    assert harness.events[0] == {"type": "tool_start", "tool": "not_exists"}
    assert harness.events[1]["type"] == "tool_end"
    assert "failed" in harness.events[1]["output"].lower()

    assert harness.events[2] == {"type": "tool_start", "tool": "faq_lookup_tool"}
    assert harness.events[3]["type"] == "tool_end"
    assert "failed" in harness.events[3]["output"].lower()

    assert len(harness.outputs) == 2
    assert harness.outputs[0][0] == "call-unknown"
    assert harness.outputs[1][0] == "call-invalid"
    assert harness.create_response_calls == 2
