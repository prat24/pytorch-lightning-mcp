from fastapi import FastAPI

from lightning_mcp.constants import PROTOCOL_VERSION, SERVER_VERSION
from lightning_mcp.handlers.inspect import InspectHandler
from lightning_mcp.handlers.train import TrainHandler
from lightning_mcp.protocol import MCPError, MCPRequest, MCPResponse
from lightning_mcp.tools import list_tools

app = FastAPI(title="Lightning MCP Server")

train_handler = TrainHandler()
inspect_handler = InspectHandler()


@app.post("/mcp")
def handle_mcp(request: MCPRequest) -> MCPResponse:
    try:
        # Core MCP methods
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

        # Standard MCP tools/call wrapper
        if request.method == "tools/call":
            tool_name = request.params.get("name")
            tool_params = request.params.get("arguments", {})
            synthetic_request = MCPRequest(
                id=request.id,
                method=tool_name,
                params=tool_params,
            )
            return handle_mcp(synthetic_request)

        # Lightning-specific tool methods
        if request.method == "lightning.train":
            return train_handler.handle(request)

        if request.method == "lightning.inspect":
            return inspect_handler.handle(request)

        return MCPResponse(
            id=request.id,
            error=MCPError(
                code=-32601,  # JSON-RPC Method not found
                message=f"Unknown MCP method '{request.method}'",
            ),
        )

    except Exception as e:
        return MCPResponse(
            id=getattr(request, "id", None),
            error=MCPError(
                code=-32603,  # JSON-RPC Internal error
                message="Internal MCP server error",
                data=str(e),
            ),
        )
