import argparse

from lightning_mcp.server import MCPServer


def main() -> None:
    parser = argparse.ArgumentParser("lightning-mcp")

    parser.add_argument(
        "--http",
        action="store_true",
        help="Run HTTP MCP server instead of stdio",
    )

    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()

    if args.http:
        import uvicorn
        from lightning_mcp.http_server import app

        uvicorn.run(app, host=args.host, port=args.port)
    else:
        MCPServer().serve_forever()