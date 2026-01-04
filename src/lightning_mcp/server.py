from __future__ import annotations

import json
import logging
import sys
import traceback
from typing import TextIO

from lightning_mcp.constants import PROTOCOL_VERSION, SERVER_VERSION
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

    def serve_forever(self) -> None:
        """Run the MCP server loop."""
        for line in self.stdin:
            line = line.strip()
            if not line:
                continue

            response = None
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

            except Exception as exc:
                # Extract request ID safely
                try:
                    request_id = str(data.get("id", "unknown")) if isinstance(data, dict) else "unknown"
                except Exception:
                    request_id = "unknown"
                response = self._handle_fatal_error(exc, request_id)

            if response:
                self._write_response(response)

    def _parse_request(self, raw: str) -> MCPRequest:
        """Parse incoming request line to MCPRequest."""
        data = json.loads(raw)
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
                    "capabilities": {},
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
            # Extract the actual tool name and params from nested structure
            tool_name = request.params.get("name")
            tool_params = request.params.get("arguments", {})

            # Create a synthetic request for the actual tool
            actual_request = MCPRequest(
                id=request.id,
                method=tool_name,
                params=tool_params
            )
            return self._dispatch(actual_request)

        # Handle Lightning-specific methods
        if request.method == "lightning.train":
            return self._train_handler.handle(request)

        if request.method == "lightning.inspect":
            return self._inspect_handler.handle(request)

        if request.method == "lightning.validate":
            return self._validate_handler.handle(request)

        if request.method == "lightning.test":
            return self._test_handler.handle(request)

        if request.method == "lightning.predict":
            return self._predict_handler.handle(request)

        # Unknown method
        return MCPResponse(
            id=request.id,
            error=MCPError(
                code=-32601,  # JSON-RPC 2.0: Method not found
                message=f"Unknown method '{request.method}'",
            ),
        )

    def _handle_fatal_error(self, exc: Exception, request_id: str = "unknown") -> MCPResponse:
        """Handle fatal errors during request processing."""
        # Ensure ID is a string
        request_id = str(request_id)
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
        json.dump(response.model_dump(), self.stdout)
        self.stdout.write("\n")
        self.stdout.flush()


def main() -> None:
    """Entry point for MCP server."""
    server = MCPServer()
    server.serve_forever()


if __name__ == "__main__":
    main()
