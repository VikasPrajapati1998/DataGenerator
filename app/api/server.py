"""FastAPI backend: upload a schema JSON, generate in the background, manage jobs, download the files.

The flow is always the same, whatever the size: ``POST /jobs`` with the schema file returns a job id at once;
``GET /jobs/{id}`` shows progress; when the job is ``completed``, ``GET /jobs/{id}/files`` lists the files and
``GET /jobs/{id}/files/{name}`` downloads one (streamed from disk with HTTP range support, so millions of rows
never sit in memory and an interrupted download can continue).

Only four choices are made per job: the number of records, the output format (csv, json, jsonl, excel), the
mode (``mock`` for an offline test, ``llm`` for the models configured in ``.env``) and the packaging (none, zip,
gzip). Everything else (model pool, batch sizes, retries, pooled mode...) comes from ``.env`` defaults.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Literal

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse

from app.config.model_registry import ModelRegistry
from app.config.settings import Settings
from app.core.exceptions import ConfigurationError
from app.service.config import ServiceSettings, get_service_settings
from app.service.jobs import MEDIA_TYPES, JobManager
from app.service.requests import RequestError, prepare_job, validate_request

Format = Literal["csv", "json", "jsonl", "excel"]
Mode = Literal["mock", "llm"]
Compress = Literal["none", "zip", "gzip"]


def create_app(limits: ServiceSettings | None = None, settings: Settings | None = None,
               registry: ModelRegistry | None = None) -> FastAPI:
    """Build the API application.

    Args:
        limits: Service limits (default: the fixed defaults).
        settings: Engine settings (default: environment).
        registry: Model registry (default: ``models.json``).
    """
    limits = limits or get_service_settings()
    state: dict[str, JobManager] = {}

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        state["jobs"] = _app.state.jobs = JobManager(limits, settings, registry)
        try:
            yield
        finally:
            state["jobs"].shutdown()

    app = FastAPI(title="Synthetic Data Generator API", version="2.0.0", lifespan=lifespan,
                  description="Upload a schema JSON to start a generation job, follow its progress, stop or "
                              "delete it, and download the generated files (csv, json, jsonl or excel; "
                              "optionally zip or gzip).")

    def jobs() -> JobManager:
        return state["jobs"]

    def base_url(request: Request) -> str:
        return str(request.base_url).rstrip("/")

    def job_or_404(job_id: str) -> Any:
        try:
            return jobs().get(job_id)
        except ConfigurationError as exc:
            raise HTTPException(404, str(exc)) from exc

    def with_links(request: Request, data: dict[str, Any]) -> dict[str, Any]:
        root = f"{base_url(request)}/jobs/{data['job_id']}"
        data["links"] = {"status": root, "files": f"{root}/files", "stop": f"{root}/stop", "preview": f"{root}/preview"}
        return data

    def file_entries(request: Request, job_id: str) -> list[dict[str, Any]]:
        root = f"{base_url(request)}/jobs/{job_id}/files"
        return [{k: v for k, v in item.items() if k != "path"} | {"url": f"{root}/{item['name']}"}
                for item in jobs().files(job_id)]

    async def read_upload(file: UploadFile) -> bytes:
        payload = await file.read()
        if not payload.strip():
            raise RequestError("the uploaded file is empty")
        return payload

    @app.exception_handler(RequestError)
    async def request_error(_request: Request, exc: RequestError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"error": str(exc), "problems": exc.problems})

    @app.get("/health", tags=["service"])
    async def health() -> dict[str, Any]:
        """Server status and the accepted choices."""
        return {"status": "ok", "max_records": limits.max_records, "max_concurrent_jobs": limits.max_concurrent_jobs,
                "formats": ["csv", "json", "jsonl", "excel"], "modes": ["mock", "llm"],
                "compress": ["none", "zip", "gzip"]}

    @app.post("/jobs", status_code=202, tags=["jobs"])
    async def start_job(
        request: Request,
        file: UploadFile = File(..., description="schema JSON file (a bare schema or a request with 'schema')"),
        records: int | None = Form(None, gt=0, description="number of records (default: total_records in the file; "
                                                           "leave empty for a multi-table file)"),
        format: Format | None = Form(None, description="csv, json, jsonl or excel (default: the file's, else csv)"),  # noqa: A002
        mode: Mode = Form("llm", description="mock = offline test data; llm = the models configured in .env"),
        compress: Compress = Form("none", description="none, zip or gzip"),
    ) -> JSONResponse:
        """Upload a schema file and start generating. Returns the job id at once."""
        payload = await read_upload(file)
        name = Path(file.filename or "").stem or None
        engine_request, deliver = prepare_job(payload, records, format, mode, name)
        model, _report = await run_in_threadpool(validate_request, engine_request, limits)
        record = jobs().submit(model, compress, deliver, mode)
        return JSONResponse(status_code=202, content=with_links(request, record.public()))

    @app.get("/jobs", tags=["jobs"])
    async def list_jobs() -> list[dict[str, Any]]:
        """All jobs, newest first."""
        return [record.public() for record in jobs().list()]

    @app.get("/jobs/{job_id}", tags=["jobs"])
    async def job_status(request: Request, job_id: str) -> dict[str, Any]:
        """Status and progress (records done, percent, phase); the error and a hint when it failed."""
        job_or_404(job_id)
        return with_links(request, jobs().status(job_id))

    @app.post("/jobs/{job_id}/stop", tags=["jobs"])
    async def stop_job(job_id: str) -> dict[str, Any]:
        """Stop a queued or running job. Finished batches are kept, so it can be resumed."""
        job_or_404(job_id)
        return jobs().cancel(job_id).public()

    @app.post("/jobs/{job_id}/resume", status_code=202, tags=["jobs"])
    async def resume_job(request: Request, job_id: str) -> dict[str, Any]:
        """Continue a stopped, failed or interrupted job (only unfinished batches are generated)."""
        job_or_404(job_id)
        try:
            record = jobs().resume(job_id)
        except ConfigurationError as exc:
            raise HTTPException(409, str(exc)) from exc
        return with_links(request, record.public())

    @app.delete("/jobs/{job_id}", tags=["jobs"])
    async def delete_job(job_id: str) -> dict[str, str]:
        """Delete a job that is not running, with all its files."""
        job_or_404(job_id)
        try:
            await run_in_threadpool(jobs().delete, job_id)
        except ConfigurationError as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"deleted": job_id}

    def finished(job_id: str) -> Any:
        record = job_or_404(job_id)
        if record.status != "completed":
            raise HTTPException(409, f"job {job_id} is {record.status}; files are available once it is completed")
        return record

    @app.get("/jobs/{job_id}/files", tags=["files"])
    async def list_files(request: Request, job_id: str) -> dict[str, Any]:
        """Generated files of a finished job: the dataset and its quality report, with download URLs."""
        record = finished(job_id)
        return {"job_id": job_id, "records": record.records, "files": file_entries(request, job_id)}

    @app.get("/jobs/{job_id}/files/{name}", tags=["files"], response_class=FileResponse)
    async def download_file(job_id: str, name: str) -> FileResponse:
        """Download one file (streamed from disk; supports HTTP Range to continue a broken download)."""
        finished(job_id)
        try:
            path = jobs().file_path(job_id, name)
        except ConfigurationError as exc:
            raise HTTPException(404, str(exc)) from exc
        return FileResponse(path, media_type=MEDIA_TYPES.get(path.suffix, "application/octet-stream"),
                            filename=f"{job_id}_{name}" if not name.startswith(job_id) else name)

    @app.get("/jobs/{job_id}/preview", tags=["files"])
    async def preview(job_id: str, rows: int = Query(10, ge=1, le=1000),
                      table: str | None = Query(None, description="multi-table jobs: which table "
                                                                  "(default: the first)")) -> dict[str, Any]:
        """First rows of the generated dataset (or of one table), to look at it without downloading."""
        finished(job_id)
        try:
            return await run_in_threadpool(jobs().preview, job_id, rows, table)
        except ConfigurationError as exc:
            raise HTTPException(404, str(exc)) from exc

    return app
