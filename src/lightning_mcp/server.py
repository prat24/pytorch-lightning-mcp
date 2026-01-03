from __future__ import annotations

import json
import sys
import traceback
from typing import TextIO

from lightning_mcp.handlers.inspect import InspectHandler
from lightning_mcp.handlers.train import TrainHandler
from lightning_mcp.protocol import MCPError, MCPRequest, MCPResponse
from lightning_mcp.tools import list_tools


class MCPServer:
    """Stdio-based MCP server.

    Reads MCP requests as JSON objects (one per line) from stdin
    and writes MCP responses as JSON objects to stdout.
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

    def serve_forever(self) -> None:
        """Run the MCP server loop."""
        for line in self.stdin:
            line = line.strip()
            if not line:
                continue

            try:
                request = self._parse_request(line)
                response = self._dispatch(request)
            except Exception as exc:
                response = self._handle_fatal_error(exc)

            self._write_response(response)

    def _parse_request(self, raw: str) -> MCPRequest:
        data = json.loads(raw)
        return MCPRequest(**data)

    def _dispatch(self, request: MCPRequest) -> MCPResponse:
        if request.method == "tools/list":
            return MCPResponse(
                id=request.id,
                result={"tools": list_tools()},
            )

        if request.method == "lightning.train":
            return self._train_handler.handle(request)

        if request.method == "lightning.inspect":
            return self._inspect_handler.handle(request)

        return MCPResponse(
            id=request.id,
            error=MCPError(
                code=404,
                message=f"Unknown MCP method '{request.method}'",
            ),
        )

    def _handle_fatal_error(self, exc: Exception) -> MCPResponse:
        return MCPResponse(
            id="unknown",
            error=MCPError(
                code=500,
                message=str(exc),
                data={"traceback": traceback.format_exc()},
            ),
        )

    def _write_response(self, response: MCPResponse) -> None:
        json.dump(response.model_dump(), self.stdout)
        self.stdout.write("\n")
        self.stdout.flush()


def main() -> None:
    server = MCPServer()
    server.serve_forever()


if __name__ == "__main__":
    main()
