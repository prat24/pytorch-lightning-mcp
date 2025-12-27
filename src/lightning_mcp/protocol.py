from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, Field, model_validator


class MCPRequest(BaseModel):
    """Incoming MCP request."""

    id: str
    method: str
    params: dict[str, Any] = Field(default_factory=dict)

class MCPError(BaseModel):
    """MCP error object."""

    code: int
    message: str
    data: Any | None = None

class MCPResponse(BaseModel):
    id: str
    result: dict | None = None
    error: MCPError | None = None

    @model_validator(mode="after")
    def check_result_or_error(self):
        if self.result is not None and self.error is not None:
            raise ValueError(
                "MCPResponse cannot contain both `result` and `error`"
            )
        if self.result is None and self.error is None:
            raise ValueError(
                "MCPResponse must contain either `result` or `error`"
            )
        return self

class MCPMethod:
    """Known MCP methods exposed by lightning-mcp."""

    TRAIN: Literal["lightning.train"] = "lightning.train"
    INSPECT: Literal["lightning.inspect"] = "lightning.inspect"