from __future__ import annotations

import json
import logging
import sys
import traceback
from typing import TextIO

from lightning_mcp.handlers.inspect import InspectHandler
from lightning_mcp.handlers.train import TrainHandler
from lightning_mcp.protocol import (
    InitializeResult,
    MCPError,
    MCPNotification,
    MCPRequest,
    MCPResponse,
    MCPMethod,
    ServerInfo,
)
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

    # MCP Protocol version
    PROTOCOL_VERSION = "2024-11-05"
    SERVER_VERSION = "0.1.0"

    def __init__(
        self,
        stdin: TextIO | None = None,
        stdout: TextIO | None = None,
    ) -> None:
        self.stdin = stdin or sys.stdin
        self.stdout = stdout or sys.stdout
        self._initialized = False

        self._train_handler = TrainHandler()
        self._inspect_handler = InspectHandler()

    def serve_forever(self) -> None:
        """Run the MCP server loop."""
        for line in self.stdin:
            line = line.strip()
            if not line:
                continue

            try:
                # Parse as dict first to distinguish requests from notifications
                data = json.loads(line)
                
                # Check if this is a notification (no id field)
                if "id" not in data:
                    # Process notification but don't send response
                    self._handle_notification(MCPNotification(**data))
                    continue
                
                # This is a request, parse it properly
                request = self._parse_request(line)
                response = self._dispatch(request)
                
            except Exception as exc:
                response = self._handle_fatal_error(exc, data.get("id") if isinstance(data, dict) else "unknown")

            self._write_response(response)

    def _parse_request(self, raw: str) -> MCPRequest:
        """Parse incoming request line to MCPRequest."""
        data = json.loads(raw)
        return MCPRequest(**data)

    def _dispatch(self, request: MCPRequest) -> MCPResponse:
        """Dispatch request to appropriate handler."""
        # Handle MCP core methods
        if request.method == MCPMethod.INITIALIZE:
            self._initialized = True
            return MCPResponse(
                id=request.id,
                result={
                    "protocolVersion": self.PROTOCOL_VERSION,
                    "capabilities": {},
                    "serverInfo": {
                        "name": "lightning-mcp",
                        "version": self.SERVER_VERSION,
                    },
                },
            )

        if request.method == MCPMethod.TOOLS_LIST:
            return MCPResponse(
                id=request.id,
                result={"tools": list_tools()},
            )

        # Handle Lightning-specific methods
        if request.method == MCPMethod.TRAIN:
            return self._train_handler.handle(request)

        if request.method == MCPMethod.INSPECT:
            return self._inspect_handler.handle(request)

        # Unknown method
        return MCPResponse(
            id=request.id,
            error=MCPError(
                code=404,
                message=f"Unknown method '{request.method}'",
            ),
        )

    def _handle_notification(self, notification: MCPNotification) -> None:
        """Handle notification message (requires no response)."""
        # Currently, we don't need to do anything special with notifications
        # but this allows for future extensibility
        pass

    def _handle_fatal_error(self, exc: Exception, request_id: str = "unknown") -> MCPResponse:
        """Handle fatal errors during request processing."""
        return MCPResponse(
            id=request_id,
            error=MCPError(
                code=500,
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
