"""Launcher for MCP clients that cannot set a working directory (Claude Desktop, some IDEs).

    D:\\Study\\DataGenerator\\.venv\\Scripts\\python.exe D:\\Study\\DataGenerator\\run_mcp_server.py

Same options as ``python -m app.mcp_server`` (``--transport http``, ``--port``, ``--api-url``).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.mcp_server.__main__ import main  # noqa: E402

if __name__ == "__main__":
    main()
