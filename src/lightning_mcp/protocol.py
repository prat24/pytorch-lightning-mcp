from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, model_validator


class MCPRequest(BaseModel):
    """Incoming MCP request."""

    jsonrpc: Literal["2.0"] = "2.0"
    id: str
    method: str
    params: dict[str, Any] = {}


class MCPError(BaseModel):
    """MCP error object."""

    code: int
    message: str
    data: Any | None = None


class MCPResponse(BaseModel):
    """MCP response - fully compliant with JSON-RPC 2.0 and MCP spec."""

    jsonrpc: Literal["2.0"] = "2.0"
    id: str | None
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
