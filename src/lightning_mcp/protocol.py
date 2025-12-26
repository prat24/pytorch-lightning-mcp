from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, Field


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
    """Outgoing MCP response."""

    id: str
    result: Any | None = None
    error: MCPError | None = None


class MCPMethod:
    """Known MCP methods exposed by lightning-mcp."""

    TRAIN: Literal["lightning/train"] = "lightning/train"
    INSPECT: Literal["lightning/inspect"] = "lightning/inspect"