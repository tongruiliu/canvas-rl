"""Base tool abstraction for the generic agentic rollout path."""

from typing import Any, Optional
from uuid import uuid4

from .schemas import OpenAIFunctionToolSchema, ToolResponse


class BaseTool:
    def __init__(self, config: dict, tool_schema: Optional[OpenAIFunctionToolSchema] = None):
        """Store tool config and expose a normalized schema / name."""
        self.config = config
        resolved_schema = tool_schema if tool_schema is not None else self.get_openai_tool_schema()
        if resolved_schema is None:
            raise ValueError("Tool schema is not set.")
        self.tool_schema = resolved_schema
        self.name = self.tool_schema.function.name

    def get_openai_tool_schema(self) -> Optional[OpenAIFunctionToolSchema]:
        """Return the OpenAI-style schema used for prompting the model."""
        return None

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        """Create one tool session for a trajectory or sample."""
        return (instance_id or str(uuid4()), ToolResponse())

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Execute the tool once and return response / reward / metrics."""
        return ToolResponse(text="Tool execution is not implemented."), 0.0, {}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """Optional hook for tool-specific reward shaping."""
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        """Release any per-instance tool state."""
        return None