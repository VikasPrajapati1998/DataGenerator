"""Fixed limits of the HTTP API and the MCP server.

There is nothing to configure: the API and MCP server use these defaults, and the engine settings
(models, output folder, batch sizes...) come from ``.env`` as for the CLI.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class ServiceSettings(BaseModel):
    """Limits and locations of the API / MCP service.

    Attributes:
        max_records: Largest job accepted (the engine itself caps one request at 50,000,000).
        max_concurrent_jobs: Jobs that run at the same time (each one uses all CPU queues).
        sync_concurrency: Small MCP samples generated at the same time (returned inline).
        sync_wait_seconds: How long such a sample waits for a free slot.
        chunk_size: Bytes per chunk when streaming or compressing files.
        state_dir: Where job records (``<job_id>.json``) are stored (default: ``<output_dir>/api_jobs``).
        mcp_inline_max_records: ``generate_data`` returns rows inline up to this size; larger runs become jobs.
        mcp_preview_max_rows: Most rows ``preview_job_output`` returns.
        api_url: MCP only (``--api-url``): forward every call to this running API instead of generating itself.
    """

    max_records: int = Field(default=50_000_000, ge=1, le=50_000_000)
    max_concurrent_jobs: int = Field(default=1, ge=1, le=64)
    sync_concurrency: int = Field(default=2, ge=1, le=64)
    sync_wait_seconds: float = Field(default=300.0, ge=0)
    chunk_size: int = Field(default=1024 * 1024, ge=4096)
    state_dir: Path | None = None
    mcp_inline_max_records: int = Field(default=20, ge=1, le=1000)
    mcp_preview_max_rows: int = Field(default=50, ge=1, le=1000)
    api_url: str | None = None


def get_service_settings() -> ServiceSettings:
    """The fixed service limits."""
    return ServiceSettings()
