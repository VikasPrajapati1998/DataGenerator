"""Start the MCP server.

stdio (for Claude Desktop, Claude Code, Cursor, the example Ollama client...)::

    python -m app.mcp_server

Streamable HTTP (for clients that connect to a URL)::

    python -m app.mcp_server --transport http --port 8765      # endpoint: http://127.0.0.1:8765/mcp
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> None:
    """Parse arguments and run the server on the chosen transport."""
    parser = argparse.ArgumentParser(prog="python -m app.mcp_server", description="Synthetic data generator MCP server")
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--api-url", help="forward all calls to a running API")
    args = parser.parse_args()

    # Clients often start the server from another folder; the engine expects the project root.
    root = Path(__file__).resolve().parent.parent.parent
    os.chdir(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from app.mcp_server.backend import make_backend  # pylint: disable=import-outside-toplevel
    from app.mcp_server.server import build_server  # pylint: disable=import-outside-toplevel
    from app.service.config import ServiceSettings  # pylint: disable=import-outside-toplevel

    limits = ServiceSettings(api_url=args.api_url.rstrip("/") if args.api_url else None)
    server = build_server(make_backend(limits), limits)
    if args.transport == "stdio":
        server.run("stdio")  # stdout carries the protocol; all logging goes to stderr and the log file
    else:
        server.run("streamable-http", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
