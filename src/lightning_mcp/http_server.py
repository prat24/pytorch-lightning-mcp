from __future__ import annotations
import traceback
from typing import Any
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from lightning_mcp.protocol import MCPRequest, MCPResponse, MCPError
from lightning_mcp.handlers.train import TrainHandler
from lightning_mcp.handlers.inspect import InspectHandler

app = FastAPI(
    title="lightning-mcp",
    description="HTTP MCP server for PyTorch Lightning",
    version="0.1.0",
)

_train_handler = TrainHandler()
_inspect_handler = InspectHandler()


@app.post("/mcp", response_model=MCPResponse)
def handle_mcp(request: MCPRequest) -> MCPResponse:
    try:
        if request.method == "lightning/train":
            return _train_handler.handle(request)

        if request.method == "lightning/inspect":
            return _inspect_handler.handle(request)

        return MCPResponse(
            id=request.id,
            error=MCPError(
                code=404,
                message=f"Unknown MCP method '{request.method}'",
            ),
        )

    except Exception as exc:
        return MCPResponse(
            id=request.id,
            error=MCPError(
                code=500,
                message=str(exc),
                data={"traceback": traceback.format_exc()},
            ),
        )