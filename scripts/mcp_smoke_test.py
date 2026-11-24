"""Smoke test of the MCP server alone (mock mode, offline, no Ollama and no API server needed).

Run from the project root::

    python scripts\\mcp_smoke_test.py

It starts ``python -m app.mcp_server`` over stdio and checks the tools (inline rows, a job with an Excel file,
an invalid schema), then starts it with ``--transport http`` and calls a tool over streamable HTTP.
``scripts\\api_smoke_test.py`` runs these checks too, plus the HTTP API and MCP forwarding to the API.
Exit code 0 = all passed, 1 = something failed.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import api_smoke_test as smoke  # noqa: E402


def main() -> int:
    work = Path(tempfile.mkdtemp(prefix="sde_mcp_smoke_"))
    env = {**os.environ, "SDE_OUTPUT_DIR": str(work / "output"), "SDE_LOG_DIR": str(work / "logs"),
           "PYTHONIOENCODING": "utf-8"}
    print(f"MCP smoke test: mock mode, files in {work}\n")
    started = time.perf_counter()
    asyncio.run(smoke.mcp_checks(None, env, work))
    print(f"\nFinished in {time.perf_counter() - started:.0f}s")
    shutil.rmtree(work, ignore_errors=True)
    if smoke.FAILURES:
        print(f"MCP SMOKE TEST FAILED: {len(smoke.FAILURES)} problem(s)")
        for line in smoke.FAILURES:
            print(f"  - {line}")
        return 1
    print("MCP SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
