"""
CLI entry point for the multi-agent RAG MCP server.

Run:
    python -m src.mcp_server                       # streamable-http, 127.0.0.1:8100
    python -m src.mcp_server --transport stdio     # stdio (e.g. Claude Desktop)
    python -m src.mcp_server --host 0.0.0.0 --port 9000

Or skip running this entirely and point the MCP Inspector at the server
module directly:
    mcp dev src/mcp_server/server.py
"""

from __future__ import annotations

import argparse
import sys

from src.mcp_server.server import DEFAULT_HOST, DEFAULT_PORT, mcp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Multi-Agent RAG MCP server.")
    parser.add_argument(
        "--transport",
        choices=["streamable-http", "stdio"],
        default="streamable-http",
        help="MCP transport to use. Default: streamable-http.",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"Host to bind for streamable-http. Default: {DEFAULT_HOST} (or $MCP_SERVER_HOST).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Port to bind for streamable-http. Default: {DEFAULT_PORT} (or $MCP_SERVER_PORT).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # FastMCP reads settings.host/settings.port lazily, inside
    # run_streamable_http_async, at the moment the transport actually starts
    # -- so overriding them here (after construction, before run) takes effect.
    mcp.settings.host = args.host
    mcp.settings.port = args.port

    if args.transport == "streamable-http":
        print(f"MCP server: http://{args.host}:{args.port}/mcp", file=sys.stderr)
        print(
            "Hint: for the MCP Inspector, run `mcp dev src/mcp_server/server.py` instead.",
            file=sys.stderr,
        )
    try:
        mcp.run(transport=args.transport)
    except KeyboardInterrupt:
        print("MCP server stopped.", file=sys.stderr)


if __name__ == "__main__":
    main()
