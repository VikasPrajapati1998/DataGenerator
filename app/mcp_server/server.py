"""MCP server exposing the generator as tools for any MCP client (Claude, ChatGPT/agents, Ollama bridges...).

Large outputs never pass through the model's context: ``generate_data`` returns rows inline only for tiny
samples (up to 20 records). Anything bigger becomes a background job, and the
model gets a job id, progress, a short preview and finally a file path / download link.
"""

from __future__ import annotations

import functools
import time
from collections.abc import Callable
from typing import Annotated, Any, Literal

from pydantic import Field

from mcp.server.mcpserver import MCPServer

from app.core.exceptions import EngineError
from app.mcp_server.backend import Backend, check, make_backend, samples
from app.service.config import ServiceSettings, get_service_settings
from app.service.requests import RequestError, load_payload, load_sample

RequestArg = Annotated[dict[str, Any] | str | None, Field(
    description='Request object {"schema": {...}, "total_records": N, "output_format": "csv"}, a bare schema '
                '{"field": "description", ...}, or the same as JSON text. Omit when using sample.')]
SampleArg = Annotated[str | None, Field(description="Name of a bundled sample schema (see list_sample_schemas)")]
RecordsArg = Annotated[int | None, Field(description="Number of records; overrides total_records", gt=0)]
JobIdArg = Annotated[str, Field(description="Job id returned by generate_data")]

INSTRUCTIONS = """\
Synthetic data generator. Typical flow:
1. list_sample_schemas / get_sample_schema to see the request format (or write your own schema).
2. validate_schema to check a request before running it.
3. generate_data: tiny runs return rows inline; larger runs return a job_id.
4. job_status(job_id) until status is completed, then list_job_files(job_id) for the file paths / URLs,
   and preview_job_output(job_id) to show a few rows. Never ask for all rows of a large dataset.
5. stop_job / delete_job manage jobs. Choices per run: records, output_format (csv, json, jsonl, excel),
   mode (mock = offline test data, llm = the server's configured models) and compress (none, zip, gzip).
A request looks like {"schema": {"id": "unique integer", "name": "realistic full name",
"department": ["IT", "HR"], "salary": "realistic salary based on department"}, "total_records": 1000,
"output_format": "csv"}."""


def _safe(fn: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
    """Turn expected errors into a JSON answer the model can read and act on."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
        try:
            return fn(*args, **kwargs)
        except RequestError as exc:
            return {"ok": False, "error": str(exc), "problems": exc.problems}
        except EngineError as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    return wrapper


def build_server(backend: Backend | None = None, limits: ServiceSettings | None = None) -> MCPServer:
    """Create the MCP server with its tools.

    Args:
        backend: Where work is done (default: in-process, or the API when ``--api-url`` is given).
        limits: Service limits (default: the fixed defaults).
    """
    limits = limits or get_service_settings()
    backend = backend or make_backend(limits)
    server = MCPServer("synthetic-data-generator", instructions=INSTRUCTIONS, version="2.0.0")

    def _inline(report: dict[str, Any]) -> bool:
        return not report.get("multi_table") and report["total_records"] <= limits.mcp_inline_max_records

    def resolve(request: dict[str, Any] | str | None, sample: str | None, records: int | None,
                output_format: str | None = None, mode: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        if request is None and not sample:
            raise RequestError("pass either request (a request/schema object) or sample (a sample schema name)")
        payload = load_payload(request) if request is not None else load_sample(sample or "")
        name = sample if request is None else None
        return payload, {"records": records, "output_format": output_format, "mode": mode, "schema_name": name}

    @server.tool()
    @_safe
    def list_sample_schemas() -> dict[str, Any]:
        """List the bundled sample schemas (templates showing the schema format)."""
        return {"samples": [{k: v for k, v in s.items() if k != "pins_models"} for s in samples()]}

    @server.tool()
    @_safe
    def get_sample_schema(name: Annotated[str, Field(description="sample name")]) -> dict[str, Any]:
        """Return the full JSON of one sample (use it as a template for your own schema)."""
        return {"name": name, "request": load_sample(name)}

    @server.tool()
    @_safe
    def validate_schema(request: RequestArg = None, sample: SampleArg = None,
                        records: RecordsArg = None) -> dict[str, Any]:
        """Check a schema without generating data: columns, generators, unique and LLM fields, warnings."""
        payload, choices = resolve(request, sample, records)
        report = check(payload, choices, limits)
        report["mode"] = "inline" if _inline(report) else "job"
        return {"ok": True, **report}

    @server.tool()
    @_safe
    def generate_data(
        request: RequestArg = None, sample: SampleArg = None, records: RecordsArg = None,
        output_format: Annotated[Literal["csv", "json", "jsonl", "excel"] | None,
                                 Field(description="file format of a job (default csv)")] = None,
        mode: Annotated[Literal["mock", "llm"], Field(description="mock = offline test data; llm = the models "
                                                                  "configured on the server")] = "llm",
        compress: Annotated[Literal["none", "zip", "gzip"], Field(description="packaging of the file")] = "none",
        wait_seconds: Annotated[float, Field(description="for jobs: wait up to this many seconds (max 120)",
                                             ge=0)] = 0,
    ) -> dict[str, Any]:
        """Generate a dataset. Tiny runs return rows inline; larger runs start a job and return its id."""
        payload, choices = resolve(request, sample, records, output_format, mode)
        report = check(payload, choices, limits)  # fail fast with readable problems
        if _inline(report):
            return {"ok": True, **backend.generate_inline(payload, choices, limits.mcp_inline_max_records)}
        job = backend.submit(payload, choices, compress)
        job_id = job["job_id"]
        deadline = time.monotonic() + min(wait_seconds, 120)
        while job.get("status") in ("queued", "running") and time.monotonic() < deadline:
            time.sleep(1.0)
            job = backend.status(job_id)
        result = {"ok": True, "mode": "job", "job_id": job_id, "status": job.get("status"),
                  "progress": job.get("progress"),
                  "next_step": "call job_status(job_id) until completed, then list_job_files(job_id)"}
        if job.get("status") == "completed":
            result["files"] = backend.files(job_id)["files"]
        elif job.get("error"):
            result.update(error=job.get("error"), hint=job.get("hint"))
        return result

    @server.tool()
    @_safe
    def job_status(job_id: JobIdArg) -> dict[str, Any]:
        """Status and progress of a job (records done, percent, phase); the error and a hint when it failed."""
        status = backend.status(job_id)
        status.pop("links", None)
        return {"ok": True, **status}

    @server.tool()
    @_safe
    def list_jobs() -> dict[str, Any]:
        """All jobs, newest first."""
        return {"jobs": backend.jobs()}

    @server.tool()
    @_safe
    def stop_job(job_id: JobIdArg) -> dict[str, Any]:
        """Stop a queued or running job (finished batches are kept; the job can be resumed)."""
        return {"ok": True, **backend.stop(job_id)}

    @server.tool()
    @_safe
    def delete_job(job_id: JobIdArg) -> dict[str, Any]:
        """Delete a job that is not running, with all its files."""
        return {"ok": True, **backend.delete(job_id)}

    @server.tool()
    @_safe
    def list_job_files(job_id: JobIdArg) -> dict[str, Any]:
        """Files of a finished job (dataset and quality report): name, size, local path, and URL via the API."""
        return {"ok": True, **backend.files(job_id)}

    @server.tool()
    @_safe
    def preview_job_output(job_id: JobIdArg,
                           rows: Annotated[int, Field(description="rows to return", ge=1)] = 10,
                           table: Annotated[str | None, Field(description="table name (multi-table jobs; "
                                                                          "default: the first table)")] = None
                           ) -> dict[str, Any]:
        """First rows of a finished job's dataset, or of one table of a multi-table job (at most 50)."""
        return {"ok": True, **backend.preview(job_id, max(1, min(rows, limits.mcp_preview_max_rows)), table)}

    return server
