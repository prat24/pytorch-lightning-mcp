from fastapi import FastAPI

from lightning_mcp.protocol import MCPRequest, MCPResponse, MCPError
from lightning_mcp.handlers.train import TrainHandler
from lightning_mcp.handlers.inspect import InspectHandler
from lightning_mcp.tools import list_tools

app = FastAPI(title="Lightning MCP Server")

train_handler = TrainHandler()
inspect_handler = InspectHandler()


@app.post("/mcp")
def handle_mcp(request: MCPRequest) -> MCPResponse:
    if request.method == "tools/list":
        return MCPResponse(
            id=request.id,
            result={"tools": list_tools()},
        )

    if request.method == "lightning.train":
        return train_handler.handle(request)

    if request.method == "lightning.inspect":
        return inspect_handler.handle(request)

    return MCPResponse(
        id=request.id,
        error=MCPError(
            code=404,
            message=f"Unknown MCP method '{request.method}'",
        ),
    )