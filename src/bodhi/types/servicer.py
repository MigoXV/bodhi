from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, Field, TypeAdapter


class AudioEvent(BaseModel):
    type: Literal["audio"]
    data: list[int]


class TextEvent(BaseModel):
    type: Literal["text"]
    text: str


class ImageEvent(BaseModel):
    type: Literal["image"]
    data_url: str
    text: str | None = None


class CommitAudioEvent(BaseModel):
    type: Literal["commit_audio"]


class ImageStartEvent(BaseModel):
    type: Literal["image_start"]
    id: str | int
    text: str | None = None


class ImageChunkEvent(BaseModel):
    type: Literal["image_chunk"]
    id: str | int
    chunk: str = ""


class ImageEndEvent(BaseModel):
    type: Literal["image_end"]
    id: str | int


class ToolApprovalDecisionEvent(BaseModel):
    type: Literal["tool_approval_decision"]
    call_id: str | None = None
    approve: bool
    always: bool = False


class InterruptEvent(BaseModel):
    type: Literal["interrupt"]


class SetVoiceEvent(BaseModel):
    type: Literal["set_voice"]
    voice: str


ClientEventUnion: TypeAlias = Annotated[
    (
        AudioEvent
        | TextEvent
        | ImageEvent
        | CommitAudioEvent
        | ImageStartEvent
        | ImageChunkEvent
        | ImageEndEvent
        | ToolApprovalDecisionEvent
        | InterruptEvent
        | SetVoiceEvent
    ),
    Field(discriminator="type"),
]
ClientEventAdapter = TypeAdapter(ClientEventUnion)


class ClientInfoEnvelope(BaseModel):
    type: Literal["client_info"] = "client_info"
    info: str
    id: str | None = None
    size: int | None = None
    count: int | None = None


class ErrorEnvelope(BaseModel):
    type: Literal["error"] = "error"
    error: str


ServerEnvelope: TypeAlias = ClientInfoEnvelope | ErrorEnvelope
