import json
import io

from lightning_mcp.tools import list_tools
from lightning_mcp.server import MCPServer


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
    Each tool must declare name, description, and input_schema.
    """

    tools = list_tools()

    for tool in tools:
        assert "name" in tool
        assert "description" in tool
        assert "input_schema" in tool

        assert isinstance(tool["name"], str)
        assert isinstance(tool["description"], str)
        assert isinstance(tool["input_schema"], dict)


def test_tools_input_schema_is_json_schema_like():
    """
    input_schema should look like a JSON Schema object.
    """

    tools = list_tools()

    for tool in tools:
        schema = tool["input_schema"]

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
    assert response["error"] is None

    tools = response["result"]["tools"]
    names = {tool["name"] for tool in tools}

    assert "lightning.train" in names
    assert "lightning.inspect" in names