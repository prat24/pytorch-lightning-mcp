from fastapi.testclient import TestClient

from lightning_mcp.http_server import app

client = TestClient(app)


def test_http_inspect_environment():
    """
    End-to-end test for the HTTP MCP server.

    Verifies:
    - POST /mcp accepts valid MCP requests
    - response structure matches MCPResponse
    """

    response = client.post(
        "/mcp",
        json={
            "id": "http-1",
            "method": "lightning.inspect",
            "params": {
                "what": "environment",
            },
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["id"] == "http-1"
    assert payload["error"] is None
    assert "python" in payload["result"]
    assert "torch" in payload["result"]
    assert "lightning" in payload["result"]
