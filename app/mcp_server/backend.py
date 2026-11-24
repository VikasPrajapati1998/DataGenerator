"""Where the MCP tools get their work done: in this process, or on a running API server.

* :class:`LocalBackend` (default) generates in the MCP server process itself.
* :class:`HttpBackend` (``--api-url`` given) sends jobs to the FastAPI backend, so long jobs keep running when
  the MCP client (Claude Desktop, an IDE...) is closed, and several clients share one job queue.

Schema checks and samples are always answered locally: they only read files of this project.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Protocol

from app.core.exceptions import ConfigurationError
from app.service.config import ServiceSettings
from app.service.jobs import JobManager, hint_for
from app.service.requests import RequestError, list_samples, load_sample, prepare_job, validate_request
from app.storage.formats import iter_records


class Backend(Protocol):
    """Operations behind the MCP tools (all return JSON-friendly dictionaries).

    ``payload`` is a request / schema object; ``choices`` are ``records``, ``output_format``, ``mode``.
    """

    def generate_inline(self, payload: dict[str, Any], choices: dict[str, Any], max_rows: int
                        ) -> dict[str, Any]: ...  # noqa: D102
    def submit(self, payload: dict[str, Any], choices: dict[str, Any], compress: str) -> dict[str, Any]: ...  # noqa: D102
    def status(self, job_id: str) -> dict[str, Any]: ...  # noqa: D102
    def jobs(self) -> list[dict[str, Any]]: ...  # noqa: D102
    def stop(self, job_id: str) -> dict[str, Any]: ...  # noqa: D102
    def delete(self, job_id: str) -> dict[str, Any]: ...  # noqa: D102
    def files(self, job_id: str) -> dict[str, Any]: ...  # noqa: D102
    def preview(self, job_id: str, rows: int, table: str | None = None) -> dict[str, Any]: ...  # noqa: D102


def check(payload: dict[str, Any], choices: dict[str, Any], limits: ServiceSettings) -> dict[str, Any]:
    """Validate a request locally and return the report."""
    request, deliver = prepare_job(payload, choices.get("records"), choices.get("output_format"),
                                   choices.get("mode") or "llm", choices.get("schema_name"))
    _model, report = validate_request(request, limits)
    return {**report, "output_format": deliver}


def samples() -> list[dict[str, Any]]:
    """Bundled sample schemas."""
    return list_samples()


def sample(name: str) -> dict[str, Any]:
    """One sample's request JSON."""
    return load_sample(name)


class LocalBackend:
    """Runs requests with a :class:`JobManager` in this process."""

    def __init__(self, limits: ServiceSettings, manager: JobManager | None = None) -> None:
        self.limits = limits
        self._manager = manager

    @property
    def manager(self) -> JobManager:
        """Created on first use, so listing tools never loads the model registry."""
        if self._manager is None:
            self._manager = JobManager(self.limits)
        return self._manager

    def generate_inline(self, payload: dict[str, Any], choices: dict[str, Any], max_rows: int) -> dict[str, Any]:
        request, _deliver = prepare_job(payload, choices.get("records"), "jsonl", choices.get("mode") or "llm",
                                        choices.get("schema_name"))
        model, _report = validate_request(request, self.limits, max_rows)
        result = self.manager.run_sync(model)
        try:
            if not result.ok or not result.final_output:
                return {"status": result.status, "error": result.error, "error_type": result.error_type,
                        "hint": hint_for(result)}
            rows = list(iter_records(Path(result.final_output), "jsonl"))
            return {"status": "completed", "mode": "inline", "records": len(rows), "rows": rows}
        finally:
            self.manager.discard_sync_output(result.job_dir)

    def submit(self, payload: dict[str, Any], choices: dict[str, Any], compress: str) -> dict[str, Any]:
        mode = choices.get("mode") or "llm"
        request, deliver = prepare_job(payload, choices.get("records"), choices.get("output_format"), mode,
                                       choices.get("schema_name"))
        model, _report = validate_request(request, self.limits)
        return self.manager.submit(model, compress, deliver, mode).public()

    def status(self, job_id: str) -> dict[str, Any]:
        return self.manager.status(job_id)

    def jobs(self) -> list[dict[str, Any]]:
        return [record.public() for record in self.manager.list()]

    def stop(self, job_id: str) -> dict[str, Any]:
        return self.manager.cancel(job_id).public()

    def delete(self, job_id: str) -> dict[str, Any]:
        self.manager.delete(job_id)
        return {"deleted": job_id}

    def _finished(self, job_id: str) -> Any:
        record = self.manager.get(job_id)
        if record.status != "completed" or not record.download_file:
            raise ConfigurationError(f"job {job_id} is {record.status}; no files yet (check job_status)")
        return record

    def files(self, job_id: str) -> dict[str, Any]:
        record = self._finished(job_id)
        items = self.manager.files(job_id)
        return {"job_id": job_id, "records": record.records, "files": items}

    def preview(self, job_id: str, rows: int, table: str | None = None) -> dict[str, Any]:
        self._finished(job_id)
        return self.manager.preview(job_id, rows, table)


class HttpBackend:
    """Sends jobs to the FastAPI backend at ``base_url``."""

    def __init__(self, base_url: str, limits: ServiceSettings, timeout: float = 600.0, client: Any = None) -> None:
        import logging  # pylint: disable=import-outside-toplevel

        import httpx  # pylint: disable=import-outside-toplevel

        logging.getLogger("httpx").setLevel(logging.WARNING)  # one INFO line per request is noise here
        self.base_url = base_url.rstrip("/")
        self.client = client or httpx.Client(base_url=self.base_url, timeout=timeout)

    def _call(self, method: str, path: str, **kwargs: Any) -> Any:
        import httpx  # pylint: disable=import-outside-toplevel

        try:
            response = self.client.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise ConfigurationError(f"cannot reach the API at {self.base_url}: {exc}") from exc
        if response.status_code == 422:
            body = response.json()
            if "problems" in body:
                raise RequestError(body.get("error", "invalid request"), body.get("problems"))
            raise RequestError(json.dumps(body.get("detail", body))[:800])
        if response.status_code >= 400:
            try:
                detail = response.json()
            except ValueError:
                detail = response.text[:500]
            detail = detail.get("detail", detail) if isinstance(detail, dict) else detail
            raise ConfigurationError(f"API {method} {path} -> {response.status_code}: {json.dumps(detail)[:800]}")
        return response.json()

    def submit(self, payload: dict[str, Any], choices: dict[str, Any], compress: str) -> dict[str, Any]:
        name = choices.get("schema_name") or "schema"
        form = {"mode": choices.get("mode") or "llm", "compress": compress}
        if choices.get("records"):
            form["records"] = str(choices["records"])
        if choices.get("output_format"):
            form["format"] = choices["output_format"]
        files = {"file": (f"{name}.json", json.dumps(payload).encode("utf-8"), "application/json")}
        return self._call("POST", "/jobs", files=files, data=form)

    def generate_inline(self, payload: dict[str, Any], choices: dict[str, Any], max_rows: int) -> dict[str, Any]:
        job = self.submit(payload, {**choices, "output_format": "jsonl"}, "none")
        deadline = time.monotonic() + 120
        while job.get("status") in ("queued", "running") and time.monotonic() < deadline:
            time.sleep(0.5)
            job = self.status(job["job_id"])
        if job.get("status") != "completed":
            return {"status": job.get("status"), "error": job.get("error"), "hint": job.get("hint"),
                    "job_id": job["job_id"]}
        rows = self.preview(job["job_id"], max_rows)["rows"]
        self.delete(job["job_id"])  # the rows are returned; no need to keep a job for a tiny sample
        return {"status": "completed", "mode": "inline", "records": len(rows), "rows": rows}

    def status(self, job_id: str) -> dict[str, Any]:
        return self._call("GET", f"/jobs/{job_id}")

    def jobs(self) -> list[dict[str, Any]]:
        return self._call("GET", "/jobs")

    def stop(self, job_id: str) -> dict[str, Any]:
        return self._call("POST", f"/jobs/{job_id}/stop")

    def delete(self, job_id: str) -> dict[str, Any]:
        return self._call("DELETE", f"/jobs/{job_id}")

    def files(self, job_id: str) -> dict[str, Any]:
        listing = self._call("GET", f"/jobs/{job_id}/files")
        listing["note"] = "use the url values to download; paths are on the API server's machine"
        return listing

    def preview(self, job_id: str, rows: int, table: str | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {"rows": rows}
        if table:
            params["table"] = table
        return self._call("GET", f"/jobs/{job_id}/preview", params=params)


def make_backend(limits: ServiceSettings) -> Backend:
    """HTTP backend when ``api_url`` is set (``--api-url``), else the in-process backend."""
    return HttpBackend(limits.api_url, limits) if limits.api_url else LocalBackend(limits)
