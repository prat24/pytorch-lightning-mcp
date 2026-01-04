"""Lightning MCP CLI entry point."""

import argparse
import os
import sys
import warnings

# Suppress all warnings at import time to prevent polluting stdio MCP stream
warnings.filterwarnings("ignore")

# Suppress PyTorch/Lightning specific warnings
os.environ.setdefault("PYTHONWARNINGS", "ignore")
os.environ.setdefault("PL_DISABLE_FORK_CHECK", "1")


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
        # For stdio mode, redirect stderr to devnull to keep JSON stream clean
        with open(os.devnull, "w") as devnull:
            sys.stderr = devnull

            from lightning_mcp.server import MCPServer

            MCPServer().serve_forever()


if __name__ == "__main__":
    main()
