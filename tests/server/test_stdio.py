import io
import json

from lightning_mcp.server import MCPServer


def test_stdio_server_inspect_environment_roundtrip():
    """
    End-to-end test for the stdio MCP server.

    Verifies:
    - input JSON is parsed
    - request is dispatched
    - response is serialized
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
    assert "python" in response["result"]
    assert "torch" in response["result"]
