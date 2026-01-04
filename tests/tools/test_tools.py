import io
import json

from lightning_mcp.server import MCPServer
from lightning_mcp.tools import list_tools


def test_list_tools_returns_expected_tools():
    """
    list_tools() should return all supported MCP tools.
    """

    tools = list_tools()
    names = {tool["name"] for tool in tools}

    assert "lightning.train" in names
    assert "lightning.inspect" in names


def test_tools_have_required_fields():
    """
    Each tool must declare name, description, and inputSchema (MCP spec requires camelCase).
    """

    tools = list_tools()

    for tool in tools:
        assert "name" in tool
        assert "description" in tool
        assert "inputSchema" in tool

        assert isinstance(tool["name"], str)
        assert isinstance(tool["description"], str)
        assert isinstance(tool["inputSchema"], dict)


def test_tools_input_schema_is_json_schema_like():
    """
    inputSchema should look like a JSON Schema object.
    """

    tools = list_tools()

    for tool in tools:
        schema = tool["inputSchema"]

        assert "type" in schema
        assert schema["type"] == "object"
        assert "properties" in schema
        assert isinstance(schema["properties"], dict)


def test_stdio_server_tools_list_roundtrip():
    """
    End-to-end stdio test for tools/list.

    Verifies:
    - MCP request parsing
    - tools/list dispatch
    - MCP response serialization
    """

    request = {
        "id": "tools-stdio-1",
        "method": "tools/list",
        "params": {},
    }

    stdin = io.StringIO(json.dumps(request) + "\n")
    stdout = io.StringIO()

    server = MCPServer(stdin=stdin, stdout=stdout)
    server.serve_forever()

    stdout.seek(0)
    response = json.loads(stdout.readline())

    assert response["id"] == "tools-stdio-1"
    assert "error" not in response

    tools = response["result"]["tools"]
    names = {tool["name"] for tool in tools}

    assert "lightning.train" in names
    assert "lightning.inspect" in names


def test_stdio_server_tools_call_wrapper():
    """
    Test tools/call wrapper method (standard MCP SDK pattern).

    Verifies that tools/call correctly routes to underlying tool method.
    """

    request = {
        "id": "call-1",
        "method": "tools/call",
        "params": {
            "name": "lightning.inspect",
            "arguments": {"what": "environment"},
        },
    }

    stdin = io.StringIO(json.dumps(request) + "\n")
    stdout = io.StringIO()

    server = MCPServer(stdin=stdin, stdout=stdout)
    server.serve_forever()

    stdout.seek(0)
    response = json.loads(stdout.readline())

    assert response["id"] == "call-1"
    assert "error" not in response
    assert "content" in response["result"]
    assert "structuredContent" in response["result"]
    assert response["result"]["isError"] is False

    structured = response["result"]["structuredContent"]
    assert "python" in structured
    assert "torch" in structured
