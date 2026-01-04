import traceback

from fastapi import FastAPI

from lightning_mcp.constants import PROTOCOL_VERSION, SERVER_VERSION
from lightning_mcp.handlers.checkpoint import CheckpointHandler
from lightning_mcp.handlers.inspect import InspectHandler
from lightning_mcp.handlers.predict import PredictHandler
from lightning_mcp.handlers.test import TestHandler
from lightning_mcp.handlers.train import TrainHandler
from lightning_mcp.handlers.validate import ValidateHandler
from lightning_mcp.protocol import MCPError, MCPRequest, MCPResponse
from lightning_mcp.tools import list_tools

app = FastAPI(title="Lightning MCP Server")

train_handler = TrainHandler()
inspect_handler = InspectHandler()
validate_handler = ValidateHandler()
test_handler = TestHandler()
predict_handler = PredictHandler()
checkpoint_handler = CheckpointHandler()

# Map tool names to handlers
_tool_handlers = {
    "lightning.train": train_handler,
    "lightning.inspect": inspect_handler,
    "lightning.validate": validate_handler,
    "lightning.test": test_handler,
    "lightning.predict": predict_handler,
    "lightning.checkpoint": checkpoint_handler,
}


def _call_handler(request: MCPRequest, handler) -> MCPResponse:
    """Call handler with proper JSON-RPC 2.0 error code mapping."""
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


def _dispatch_tool(request_id: str, tool_name: str, tool_params: dict) -> MCPResponse:
    """Dispatch tools/call to appropriate handler.

    Per MCP spec, unknown tools return -32602 (Invalid params).
    """
    if not tool_name:
        return MCPResponse(
            id=request_id,
            error=MCPError(
                code=-32602,  # Invalid params
                message="Missing required parameter: name",
            ),
        )

    handler = _tool_handlers.get(tool_name)
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
    return _call_handler(synthetic_request, handler)


@app.post("/mcp", response_model_exclude_none=True)
def handle_mcp(request: MCPRequest) -> MCPResponse:
    try:
        # Core MCP methods
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

        # Standard MCP tools/call wrapper
        if request.method == "tools/call":
            tool_name = request.params.get("name")
            tool_params = request.params.get("arguments", {})
            return _dispatch_tool(request.id, tool_name, tool_params)

        # Lightning-specific tool methods (direct calls, not via tools/call)
        if request.method == "lightning.train":
            return _call_handler(request, train_handler)

        if request.method == "lightning.inspect":
            return _call_handler(request, inspect_handler)

        if request.method == "lightning.validate":
            return _call_handler(request, validate_handler)

        if request.method == "lightning.test":
            return _call_handler(request, test_handler)

        if request.method == "lightning.predict":
            return _call_handler(request, predict_handler)

        return MCPResponse(
            id=request.id,
            error=MCPError(
                code=-32601,  # JSON-RPC Method not found
                message=f"Unknown MCP method '{request.method}'",
            ),
        )

    except Exception as e:
        return MCPResponse(
            id=request.id if hasattr(request, "id") else None,
            error=MCPError(
                code=-32603,  # JSON-RPC Internal error
                message=str(e),
                data={"error_type": type(e).__name__},
            ),
        )
