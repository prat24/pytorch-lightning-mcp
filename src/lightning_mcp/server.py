from __future__ import annotations

import json
import logging
import sys
import traceback
from typing import TextIO

from lightning_mcp.constants import PROTOCOL_VERSION, SERVER_VERSION
from lightning_mcp.handlers.checkpoint import CheckpointHandler
from lightning_mcp.handlers.inspect import InspectHandler
from lightning_mcp.handlers.predict import PredictHandler
from lightning_mcp.handlers.test import TestHandler
from lightning_mcp.handlers.train import TrainHandler
from lightning_mcp.handlers.validate import ValidateHandler
from lightning_mcp.protocol import MCPError, MCPRequest, MCPResponse
from lightning_mcp.tools import list_tools

# Suppress non-critical logs
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class InvalidRequestError(Exception):
    """Raised when JSON-RPC request is invalid (code -32600)."""
    pass


class MCPServer:
    """Stdio-based MCP server.

    Reads MCP requests as JSON objects (one per line) from stdin
    and writes MCP responses as JSON objects to stdout.

    Fully compliant with JSON-RPC 2.0 and MCP 2024-11-05 specification.
    """

    def __init__(
        self,
        stdin: TextIO | None = None,
        stdout: TextIO | None = None,
    ) -> None:
        self.stdin = stdin or sys.stdin
        self.stdout = stdout or sys.stdout

        self._train_handler = TrainHandler()
        self._inspect_handler = InspectHandler()
        self._validate_handler = ValidateHandler()
        self._test_handler = TestHandler()
        self._predict_handler = PredictHandler()
        self._checkpoint_handler = CheckpointHandler()

    def serve_forever(self) -> None:
        """Run the MCP server loop."""
        for line in self.stdin:
            line = line.strip()
            if not line:
                continue

            response = None
            data = None  # Initialize for error handling scope
            try:
                # Parse as dict first to distinguish requests from notifications
                data = json.loads(line)

                # Check if this is a notification (no id field)
                if "id" not in data:
                    # Notifications require no response, skip processing
                    continue

                # This is a request, parse it properly
                request = self._parse_request(line)
                response = self._dispatch(request)

            except json.JSONDecodeError as exc:
                # Parse error: id MUST be null per JSON-RPC 2.0 spec
                response = MCPResponse(
                    id=None,
                    error=MCPError(
                        code=-32700,  # Parse error
                        message="Parse error: Invalid JSON",
                        data={"details": str(exc)},
                    ),
                )
            except InvalidRequestError as exc:
                # Invalid Request: id MUST be null if not extractable
                request_id = None
                if isinstance(data, dict) and "id" in data:
                    request_id = str(data["id"])
                response = MCPResponse(
                    id=request_id,
                    error=MCPError(
                        code=-32600,  # Invalid Request
                        message=f"Invalid Request: {exc}",
                    ),
                )
            except Exception as exc:
                # Internal error: preserve request ID if available
                request_id = None
                if isinstance(data, dict) and "id" in data:
                    request_id = str(data["id"])
                response = self._handle_fatal_error(exc, request_id)

            if response:
                self._write_response(response)

    def _parse_request(self, raw: str) -> MCPRequest:
        """Parse incoming request line to MCPRequest.

        Raises:
            InvalidRequestError: If the request is malformed.
        """
        data = json.loads(raw)

        # Validate it's an object
        if not isinstance(data, dict):
            raise InvalidRequestError("Request must be a JSON object")

        # Validate required fields
        if "method" not in data:
            raise InvalidRequestError("Request must have 'method' field")

        # Ensure id is a string (MCP requires string IDs)
        if "id" in data and not isinstance(data["id"], str):
            data["id"] = str(data["id"])

        return MCPRequest(**data)

    def _dispatch(self, request: MCPRequest) -> MCPResponse:
        """Dispatch request to appropriate handler."""
        # Handle MCP core methods
        if request.method == "initialize":
            return MCPResponse(
                id=request.id,
                result={
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {
                        "tools": {},  # MCP spec: servers supporting tools MUST declare this
                    },
                    "serverInfo": {
                        "name": "lightning-mcp",
                        "version": SERVER_VERSION,
                    },
                },
            )

        if request.method == "tools/list":
            return MCPResponse(
                id=request.id,
                result={"tools": list_tools()},
            )

        # Handle tools/call wrapper (standard MCP method used by SDK)
        if request.method == "tools/call":
            tool_name = request.params.get("name")
            tool_params = request.params.get("arguments", {})

            # Validate tool name is provided
            if not tool_name:
                return MCPResponse(
                    id=request.id,
                    error=MCPError(
                        code=-32602,  # Invalid params
                        message="Missing required parameter: name",
                    ),
                )

            # Route to tool handler
            return self._dispatch_tool(request.id, tool_name, tool_params)

        # Handle Lightning-specific methods (direct calls, not via tools/call)
        if request.method == "lightning.train":
            return self._call_handler(request, self._train_handler)

        if request.method == "lightning.inspect":
            return self._call_handler(request, self._inspect_handler)

        if request.method == "lightning.validate":
            return self._call_handler(request, self._validate_handler)

        if request.method == "lightning.test":
            return self._call_handler(request, self._test_handler)

        if request.method == "lightning.predict":
            return self._call_handler(request, self._predict_handler)

        # Unknown method (not a tool, not a core MCP method)
        return MCPResponse(
            id=request.id,
            error=MCPError(
                code=-32601,  # JSON-RPC 2.0: Method not found
                message=f"Unknown method '{request.method}'",
            ),
        )

    def _dispatch_tool(
        self, request_id: str, tool_name: str, tool_params: dict
    ) -> MCPResponse:
        """Dispatch tools/call to appropriate handler.

        Per MCP spec, unknown tools return -32602 (Invalid params).
        """
        # Map tool names to handlers
        tool_handlers = {
            "lightning.train": self._train_handler,
            "lightning.inspect": self._inspect_handler,
            "lightning.validate": self._validate_handler,
            "lightning.test": self._test_handler,
            "lightning.predict": self._predict_handler,
            "lightning.checkpoint": self._checkpoint_handler,
        }

        handler = tool_handlers.get(tool_name)
        if handler is None:
            # MCP spec: unknown tool returns -32602 Invalid params
            return MCPResponse(
                id=request_id,
                error=MCPError(
                    code=-32602,  # Invalid params (per MCP spec for unknown tools)
                    message=f"Unknown tool: {tool_name}",
                ),
            )

        # Create synthetic request for the handler
        synthetic_request = MCPRequest(
            id=request_id,
            method=tool_name,
            params=tool_params,
        )
        return self._call_handler(synthetic_request, handler)

    def _call_handler(self, request: MCPRequest, handler) -> MCPResponse:
        """Call handler with proper error code mapping."""
        try:
            return handler.handle(request)
        except (ValueError, TypeError) as exc:
            # Invalid params (bad model config, missing fields, etc.)
            return MCPResponse(
                id=request.id,
                error=MCPError(
                    code=-32602,  # JSON-RPC 2.0: Invalid params
                    message=str(exc),
                ),
            )
        except Exception as exc:
            # Internal error
            return MCPResponse(
                id=request.id,
                error=MCPError(
                    code=-32603,  # JSON-RPC 2.0: Internal error
                    message=str(exc),
                    data={"traceback": traceback.format_exc()},
                ),
            )

    def _handle_fatal_error(self, exc: Exception, request_id: str | None) -> MCPResponse:
        """Handle fatal errors during request processing."""
        return MCPResponse(
            id=request_id,
            error=MCPError(
                code=-32603,  # JSON-RPC 2.0: Internal error
                message=str(exc),
                data={"traceback": traceback.format_exc()},
            ),
        )

    def _write_response(self, response: MCPResponse) -> None:
        """Write response to stdout as JSON."""
        # exclude_none=True per JSON-RPC 2.0: error MUST NOT exist on success
        json.dump(response.model_dump(exclude_none=True), self.stdout)
        self.stdout.write("\n")
        self.stdout.flush()


def main() -> None:
    """Entry point for MCP server."""
    server = MCPServer()
    server.serve_forever()


if __name__ == "__main__":
    main()
