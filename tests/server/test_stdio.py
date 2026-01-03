import io
import json

from lightning_mcp.server import MCPServer


def test_stdio_server_initialize():
    """
    Test MCP initialize handshake via stdio.

    Verifies:
    - initialize method is supported
    - returns correct protocolVersion
    - includes serverInfo
    """

    request = {
        "id": "init-1",
        "method": "initialize",
        "params": {},
    }

    stdin = io.StringIO(json.dumps(request) + "\n")
    stdout = io.StringIO()

    server = MCPServer(stdin=stdin, stdout=stdout)
    server.serve_forever()

    stdout.seek(0)
    response = json.loads(stdout.readline())

    assert response["id"] == "init-1"
    assert response["error"] is None

    result = response["result"]
    assert result["protocolVersion"] == "2024-11-05"
    assert result["serverInfo"]["name"] == "lightning-mcp"
    assert "capabilities" in result


def test_stdio_server_inspect_environment_roundtrip():
    """
    End-to-end test for the stdio MCP server.

    Verifies:
    - input JSON is parsed
    - request is dispatched
    - response is serialized in CallToolResult format
    """

    request = {
        "id": "stdio-1",
        "method": "lightning.inspect",
        "params": {
            "what": "environment",
        },
    }

    stdin = io.StringIO(json.dumps(request) + "\n")
    stdout = io.StringIO()

    server = MCPServer(stdin=stdin, stdout=stdout)
    server.serve_forever()

    stdout.seek(0)
    response = json.loads(stdout.readline())

    assert response["id"] == "stdio-1"
    assert response["error"] is None
    assert "content" in response["result"]
    assert "structuredContent" in response["result"]
    assert "isError" in response["result"]
    assert response["result"]["isError"] is False

    structured = response["result"]["structuredContent"]
    assert "python" in structured
    assert "torch" in structured
