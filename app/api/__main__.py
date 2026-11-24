"""Start the API server: ``python -m app.api [--host 127.0.0.1] [--port 8000]``."""

from __future__ import annotations

import argparse


def main() -> None:
    """Parse arguments and run uvicorn (one process; jobs run in background threads)."""
    import uvicorn  # pylint: disable=import-outside-toplevel

    from app.api.server import create_app  # pylint: disable=import-outside-toplevel

    parser = argparse.ArgumentParser(prog="python -m app.api", description="Synthetic data generator HTTP API")
    parser.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to accept connections from other machines")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()
    # One worker process: background jobs live in this process (their state is also saved to disk).
    uvicorn.run(create_app(), host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
