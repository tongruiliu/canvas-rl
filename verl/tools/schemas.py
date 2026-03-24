"""Dataclass-based tool schemas used by the generic agentic rollout path."""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class OpenAIFunctionPropertySchema:
    """Schema for one tool argument property."""

    type: str | list[str]
    description: Optional[str] = None
    enum: Optional[list[str]] = None
    additionalProperties: Optional[bool] = None

@dataclass
class OpenAIFunctionParametersSchema:
    """Schema for a tool function's parameters object."""

    type: str = "object"
    properties: dict[str, OpenAIFunctionPropertySchema] = field(default_factory=dict)
    required: list[str] = field(default_factory=list)


@dataclass
class OpenAIFunctionSchema:
    """Schema for one callable tool function."""

    name: str
    description: str
    parameters: OpenAIFunctionParametersSchema = field(default_factory=OpenAIFunctionParametersSchema)
    strict: bool = False


@dataclass
class OpenAIFunctionToolSchema:
    """OpenAI-style wrapper around a function schema."""

    type: str
    function: OpenAIFunctionSchema


@dataclass
class OpenAIFunctionCallSchema:
    """Parsed tool call with validated dict arguments."""

    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResponse:
    """Generic tool response container for text / image / video outputs."""

    text: Optional[str] = None
    image: Optional[list[Any]] = None
    video: Optional[list[Any]] = None

    def __post_init__(self):
        """Keep the multimodal fields normalized as lists."""
        if self.image is not None and not isinstance(self.image, list):
            raise ValueError("ToolResponse.image must be a list when provided.")
        if self.video is not None and not isinstance(self.video, list):
            raise ValueError("ToolResponse.video must be a list when provided.")

    def is_empty(self) -> bool:
        """Return True when the tool produced no visible output."""
        return not self.text and not self.image and not self.video
