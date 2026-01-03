from fastapi.testclient import TestClient

from lightning_mcp.http_server import app

client = TestClient(app)


def test_http_initialize():
    """Test HTTP initialize handshake."""
    response = client.post(
        "/mcp",
        json={
            "id": "init-1",
            "method": "initialize",
            "params": {},
        },
    )

    assert response.status_code == 200
    payload = response.json()

    assert payload["id"] == "init-1"
    assert payload["error"] is None
    assert payload["result"]["protocolVersion"] == "2024-11-05"
    assert payload["result"]["serverInfo"]["name"] == "lightning-mcp"
    assert "capabilities" in payload["result"]


def test_http_tools_list():
    """Test HTTP tools/list endpoint."""
    response = client.post(
        "/mcp",
        json={
            "id": "tools-1",
            "method": "tools/list",
            "params": {},
        },
    )

    assert response.status_code == 200
    payload = response.json()

    assert payload["id"] == "tools-1"
    assert payload["error"] is None

    tools = payload["result"]["tools"]
    names = {tool["name"] for tool in tools}
    assert "lightning.train" in names
    assert "lightning.inspect" in names


def test_http_tools_call_wrapper():
    """Test HTTP tools/call wrapper method."""
    response = client.post(
        "/mcp",
        json={
            "id": "call-1",
            "method": "tools/call",
            "params": {
                "name": "lightning.inspect",
                "arguments": {"what": "environment"},
            },
        },
    )

    assert response.status_code == 200
    payload = response.json()

    assert payload["id"] == "call-1"
    assert payload["error"] is None
    assert "content" in payload["result"]
    assert "structuredContent" in payload["result"]
    assert payload["result"]["isError"] is False

    structured = payload["result"]["structuredContent"]
    assert "python" in structured
    assert "torch" in structured


def test_http_inspect_environment():
    """End-to-end test for the HTTP MCP server."""

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
    assert "content" in payload["result"]
    assert "structuredContent" in payload["result"]
    assert "isError" in payload["result"]
    assert payload["result"]["isError"] is False

    structured = payload["result"]["structuredContent"]
    assert "python" in structured
    assert "torch" in structured
    assert "lightning" in structured
