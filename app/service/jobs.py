"""Synchronous runs and background jobs on top of :class:`SyntheticDataEngine`.

Records never pass through memory here: the engine writes batches to disk, merges them into one file,
and callers stream that file (optionally gzip-compressed in chunks).
"""

from __future__ import annotations

import gzip
import json
import shutil
import threading
import time
import zipfile
import zlib
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from app.config.model_registry import ModelRegistry, load_registry
from app.config.settings import Settings, get_settings
from app.core.exceptions import ConfigurationError, EngineError, ResourceError
from app.monitoring.logger import get_logger, redact
from app.schemas.input_schema import GenerationRequest
from app.service.config import ServiceSettings
from app.storage.file_manager import atomic_writer, find_job_dir, read_json, write_json_atomic
from app.storage.formats import iter_records
from app.utils.helpers import new_job_id, safe_name, utc_now_iso

logger = get_logger(__name__)

ACTIVE = ("queued", "running")
RESUMABLE = ("failed", "cancelled", "interrupted")
MEDIA_TYPES = {".csv": "text/csv", ".json": "application/json", ".jsonl": "application/x-ndjson",
               ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
               ".gz": "application/gzip", ".zip": "application/zip"}
OUTPUT_FORMATS = ("csv", "json", "jsonl", "excel")
COMPRESSION = ("none", "zip", "gzip")
EXCEL_MAX_ROWS = 1_048_575  # data rows per sheet (Excel's limit is 1,048,576 rows including the header)

# HTTP status for a failed synchronous run, by engine error type.
_STATUS_BY_ERROR = {
    "SchemaError": 422, "ConfigurationError": 422, "ValidationError": 422, "RequestError": 422,
    "FeasibilityError": 422, "ModelUnavailableError": 503, "QuotaExceededError": 503, "ResourceError": 503,
    "CancelledError": 409,
}


def status_for_error(error_type: str | None) -> int:
    """HTTP status code for a failed run (500 when the error is not the caller's to fix)."""
    return _STATUS_BY_ERROR.get(error_type or "", 500)


def hint_for(result: Any) -> str | None:
    """The CLI's hint for a failed result (what to change)."""
    from app.main import _hint, exit_code_for  # pylint: disable=import-outside-toplevel

    return _hint(exit_code_for(result), result.error_type)


class BusyError(ResourceError):
    """No free slot for a synchronous request within ``sync_wait_seconds``."""


@dataclass
class JobRecord:
    """Persistent description of one background job (stored as ``<state_dir>/<job_id>.json``)."""

    job_id: str
    status: str
    schema_name: str
    total_records: int
    output_format: str
    compress: str
    created_at: str
    mode: str = "llm"
    started_at: str | None = None
    finished_at: str | None = None
    job_dir: str | None = None
    final_output: str | None = None
    download_file: str | None = None
    size_bytes: int | None = None
    records: int | None = None
    quality_score: float | None = None
    error: str | None = None
    error_type: str | None = None
    hint: str | None = None
    files: list[dict[str, Any]] = field(default_factory=list)
    tables: list[dict[str, Any]] = field(default_factory=list)
    request: dict[str, Any] = field(default_factory=dict)

    def public(self) -> dict[str, Any]:
        """The record without the stored request (what status calls return)."""
        data = asdict(self)
        data.pop("request")
        return data


def iter_file(path: Path, chunk_size: int) -> Iterator[bytes]:
    """Yield a file in chunks (constant memory)."""
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            yield chunk


def iter_gzip(path: Path, chunk_size: int, level: int = 6) -> Iterator[bytes]:
    """Yield a file gzip-compressed on the fly, chunk by chunk (constant memory)."""
    compressor = zlib.compressobj(level, zlib.DEFLATED, 31)  # wbits 31 = gzip container
    for chunk in iter_file(path, chunk_size):
        if out := compressor.compress(chunk):
            yield out
    yield compressor.flush()


def zip_file(source: Path) -> Path:
    """Pack ``source`` into ``source.zip`` (deflate, ZIP64 for large files; atomic write)."""
    target = source.with_name(source.name + ".zip")
    with atomic_writer(target, mode="wb") as raw, zipfile.ZipFile(raw, "w", zipfile.ZIP_DEFLATED,
                                                                   allowZip64=True) as archive:
        archive.write(source, arcname=source.name)
    return target


def zip_files(sources: list[Path], target: Path) -> Path:
    """Pack several files into one ``.zip`` archive (deflate, ZIP64; atomic write)."""
    with atomic_writer(target, mode="wb") as raw, zipfile.ZipFile(raw, "w", zipfile.ZIP_DEFLATED,
                                                                   allowZip64=True) as archive:
        for source in sources:
            archive.write(source, arcname=source.name)
    return target


def _excel_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value


def to_excel(source: Path, fmt: str) -> Path:
    """Convert a CSV/JSON/JSONL dataset to ``.xlsx`` row by row (write-only mode, low memory).

    More than 1,048,575 records continue on further sheets (``data``, ``data_2``, ...).
    """
    from openpyxl import Workbook  # pylint: disable=import-outside-toplevel

    target = source.with_suffix(".xlsx")
    book = Workbook(write_only=True)
    sheet, columns, rows_in_sheet, sheets = None, None, EXCEL_MAX_ROWS, 0
    for record in iter_records(source, fmt):
        if columns is None:
            columns = list(record)
        if rows_in_sheet >= EXCEL_MAX_ROWS:
            sheets += 1
            sheet = book.create_sheet("data" if sheets == 1 else f"data_{sheets}")
            sheet.append(columns)
            rows_in_sheet = 0
        sheet.append([_excel_value(record.get(c)) for c in columns])  # type: ignore[union-attr]
        rows_in_sheet += 1
    if sheet is None:
        book.create_sheet("data")
    with atomic_writer(target, mode="wb") as handle:
        book.save(handle)
    return target


def gzip_file(source: Path, chunk_size: int) -> Path:
    """Compress ``source`` to ``source.gz`` in chunks (atomic write) and return the new path."""
    target = source.with_name(source.name + ".gz")
    with atomic_writer(target, mode="wb") as raw, gzip.GzipFile(filename=source.name, mode="wb", fileobj=raw,
                                                                 compresslevel=6) as packed:
        with source.open("rb") as handle:
            shutil.copyfileobj(handle, packed, chunk_size)
    return target


class JobManager:
    """Runs generation requests synchronously or as background jobs.

    Args:
        limits: Service limits and locations.
        settings: Engine settings (default: environment).
        registry: Model registry (default: ``models.json``).
    """

    def __init__(self, limits: ServiceSettings, settings: Settings | None = None,
                 registry: ModelRegistry | None = None) -> None:
        self.limits = limits
        self.settings = settings or get_settings()
        self.registry = registry or load_registry()
        self.state_dir = Path(limits.state_dir or Path(self.settings.output_dir) / "api_jobs")
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.sync_dir = Path(self.settings.output_dir) / "_api_sync"
        self._lock = threading.RLock()
        self._jobs: dict[str, JobRecord] = {}
        self._futures: dict[str, Future[None]] = {}
        self._engines: dict[str, Any] = {}
        self._sync_slots = threading.BoundedSemaphore(limits.sync_concurrency)
        self._executor = ThreadPoolExecutor(max_workers=limits.max_concurrent_jobs, thread_name_prefix="api-job")
        self._load()
        self._sweep_sync_output()

    # -- persistence ------------------------------------------------------------------
    def _load(self) -> None:
        for path in sorted(self.state_dir.glob("*.json")):
            data = read_json(path, None)
            if not isinstance(data, dict):
                continue
            known = set(JobRecord.__dataclass_fields__)
            if isinstance(data.get("compress"), bool):  # records written before compress became a choice
                data["compress"] = "gzip" if data["compress"] else "none"
            try:
                record = JobRecord(**{k: v for k, v in data.items() if k in known})
            except TypeError:
                logger.warning("Ignoring unreadable job record %s", path)
                continue
            if record.status in ACTIVE:  # the server stopped while the job was queued or running
                record.status = "interrupted"
                record.hint = "The server restarted; POST /jobs/{id}/resume continues it (finished batches are kept)"
                self._save(record)
            self._jobs[record.job_id] = record

    def _save(self, record: JobRecord) -> None:
        write_json_atomic(self.state_dir / f"{record.job_id}.json", asdict(record))

    def _engine(self) -> Any:
        from app.engine import SyntheticDataEngine  # pylint: disable=import-outside-toplevel

        return SyntheticDataEngine(settings=self.settings, registry=self.registry)

    # -- synchronous ------------------------------------------------------------------
    def run_sync(self, model: GenerationRequest) -> Any:
        """Generate ``model`` now and return the :class:`JobResult` (the file stays on disk).

        Raises:
            BusyError: If no slot frees up within ``sync_wait_seconds``.
        """
        if not self._sync_slots.acquire(timeout=self.limits.sync_wait_seconds):
            raise BusyError("the server is busy with other synchronous requests; retry later or use POST /jobs")
        try:
            request = model.model_dump(by_alias=True)
            request["output_dir"] = str(self.sync_dir)
            return self._engine().run(request)
        finally:
            self._sync_slots.release()

    def discard_sync_output(self, job_dir: str | None) -> None:
        """Delete a synchronous run's folder.

        Windows refuses to delete a file that is still open somewhere (an antivirus scan, a just-closed
        handle), so deletion is retried briefly; anything left is removed by the next startup sweep.
        """
        if not job_dir:
            return
        path = Path(job_dir)
        for attempt in range(10):
            shutil.rmtree(path, ignore_errors=True)
            if not path.exists():
                return
            time.sleep(0.1 * (attempt + 1))
        logger.warning("Could not delete sync output %s yet (files in use); it is removed at the next start", path)

    def _sweep_sync_output(self) -> None:
        """Delete synchronous-run folders left over from an earlier server run."""
        if not self.sync_dir.is_dir():
            return
        for day in self.sync_dir.glob("Jobs_*"):
            shutil.rmtree(day, ignore_errors=True)

    # -- background jobs --------------------------------------------------------------
    def submit(self, model: GenerationRequest, compress: str = "none", output_format: str | None = None,
               mode: str = "llm") -> JobRecord:
        """Queue ``model`` as a background job and return its record.

        Args:
            model: The engine request (its ``output_format`` is the engine's file format).
            compress: ``none``, ``zip`` or ``gzip`` for the finished file.
            output_format: Format delivered to the user (``excel`` converts the engine's JSONL file).
            mode: ``mock`` or ``llm`` (informational; the request already carries the model choice).
        """
        if compress not in COMPRESSION:
            raise ConfigurationError(f"compress must be one of {', '.join(COMPRESSION)}")
        request = model.model_dump(by_alias=True, exclude_none=True)
        schema_name = safe_name(model.schema_name or (model.data_schema.get("entity")
                                                      if isinstance(model.data_schema.get("fields"), dict) else None))
        job_id = new_job_id(schema_name)
        request.update(job_id=job_id, schema_name=schema_name)
        request.pop("output_dir", None)
        record = JobRecord(job_id=job_id, status="queued", schema_name=schema_name,
                           total_records=model.total_records, output_format=output_format or model.output_format,
                           compress=compress, mode=mode, created_at=utc_now_iso(), request=request)
        with self._lock:
            self._jobs[job_id] = record
            self._save(record)
            self._futures[job_id] = self._executor.submit(self._run, job_id, False)
        logger.info("API job %s queued: %d records", job_id, model.total_records)
        return record

    def resume(self, job_id: str) -> JobRecord:
        """Continue a failed, cancelled or interrupted job (finished batches are reused).

        Raises:
            ConfigurationError: If the job is unknown or not resumable.
        """
        with self._lock:
            record = self.get(job_id)
            if record.status not in RESUMABLE:
                raise ConfigurationError(f"job {job_id} is {record.status}; only {', '.join(RESUMABLE)} jobs resume")
            has_dir = find_job_dir(self.settings.output_dir, job_id) is not None
            record.status, record.error, record.error_type, record.hint = "queued", None, None, None
            self._save(record)
            self._futures[job_id] = self._executor.submit(self._run, job_id, has_dir)
        return record

    def _run(self, job_id: str, resume: bool) -> None:
        record = self._jobs[job_id]
        engine = self._engine()
        with self._lock:
            if record.status == "cancelled":
                return
            record.status, record.started_at = "running", utc_now_iso()
            self._engines[job_id] = engine
            self._save(record)
        try:
            result = engine.resume(job_id) if resume else engine.run(record.request)
            self._finish(record, result)
        except EngineError as exc:
            self._fail(record, type(exc).__name__, redact(f"{type(exc).__name__}: {exc}"))
        except Exception as exc:  # noqa: BLE001 - a job thread must always leave a final status
            logger.critical("API job %s crashed", job_id, exc_info=True)
            self._fail(record, type(exc).__name__, f"internal error: {type(exc).__name__}: {redact(str(exc))[:300]}")
        finally:
            with self._lock:
                self._engines.pop(job_id, None)
                self._futures.pop(job_id, None)

    def _finish(self, record: JobRecord, result: Any) -> None:
        execution = result.summary.get("execution", {}) if result.summary else {}
        with self._lock:
            record.job_dir = result.job_dir
            record.final_output = result.final_output
            record.records = execution.get("valid_records_generated") or (
                result.summary.get("record_count") if result.summary else None)
            record.quality_score = ((result.summary.get("validation") or {}).get("quality") or {}).get("score") \
                if result.summary else None
        if not result.ok:
            self._fail(record, result.error_type, result.error, hint_for(result))
            return
        source = Path(result.final_output)
        if source.is_dir():  # a multi-table run: one file per table
            files = self._package_tables(source, record, result.summary.get("tables") or {})
            download = Path(files[0]["path"])
        else:
            download = self._package(source, record)
            files = [self._describe(download, "data")]
        job_dir = Path(result.job_dir) if result.job_dir else source.parent.parent
        for report in (job_dir / "metadata" / "validation_summary.json", job_dir / "metadata" / "split_summary.json",
                       job_dir / "metadata" / "multi_table_summary.json"):
            if report.exists():
                files.append(self._describe(report, "report"))
                break
        with self._lock:
            record.records = record.records or record.total_records
            record.download_file = str(download)
            record.size_bytes = download.stat().st_size
            record.files = files
            record.status, record.finished_at = "completed", utc_now_iso()
            self._save(record)
        logger.info("API job %s completed: %s (%s bytes)", record.job_id, download, f"{record.size_bytes:,}")

    def _package(self, source: Path, record: JobRecord) -> Path:
        """Convert (Excel) and compress (zip/gzip) the engine's file; then drop the intermediate files."""
        steps: list[Path] = [source]
        if record.output_format == "excel":
            steps.append(to_excel(source, record.request.get("output_format", "jsonl")))
        if record.compress == "gzip":
            steps.append(gzip_file(steps[-1], self.limits.chunk_size))
        elif record.compress == "zip":
            steps.append(zip_file(steps[-1]))
        for intermediate in steps[:-1]:
            intermediate.unlink(missing_ok=True)
        return steps[-1]

    def _package_tables(self, final_dir: Path, record: JobRecord, tables: dict[str, Any]) -> list[dict[str, Any]]:
        """Convert and compress the table files: zip packs all tables in one archive, gzip each file."""
        engine_format = record.request.get("output_format", "jsonl")
        items: list[tuple[str, Path]] = []
        for name, info in tables.items():
            path = Path(info["file"])
            if record.output_format == "excel":
                converted = to_excel(path, engine_format)
                path.unlink(missing_ok=True)
                path = converted
            items.append((name, path))
        record.tables = [{"name": name, "records": (tables[name] or {}).get("records")} for name, _ in items]
        if record.compress == "zip":
            archive = zip_files([path for _, path in items], final_dir / f"{record.schema_name}_tables.zip")
            for _, path in items:
                path.unlink(missing_ok=True)
            return [{**self._describe(archive, "data"), "tables": [name for name, _ in items]}]
        out = []
        for name, path in items:
            if record.compress == "gzip":
                packed = gzip_file(path, self.limits.chunk_size)
                path.unlink(missing_ok=True)
                path = packed
            out.append({**self._describe(path, "data"), "table": name,
                        "records": (tables[name] or {}).get("records")})
        return out

    def preview(self, job_id: str, rows: int, table: str | None = None) -> dict[str, Any]:
        """First rows of a finished job's dataset (of one table for a multi-table job).

        Raises:
            ConfigurationError: If the job is not finished or the table is unknown.
        """
        record = self.get(job_id)
        if record.status != "completed" or not record.download_file:
            raise ConfigurationError(f"job {job_id} is {record.status}; no files yet")
        path, member, total = Path(record.download_file), None, record.records
        if record.tables:
            names = [t["name"] for t in record.tables]
            table = table or names[0]
            if table not in names:
                raise ConfigurationError(f"job {job_id} has no table {table!r}; tables: {', '.join(names)}")
            total = next(t["records"] for t in record.tables if t["name"] == table)
            entry = next((f for f in self.files(job_id) if f.get("table") == table or table in f.get("tables", [])),
                         None)
            if entry is None:
                raise ConfigurationError(f"the file of table {table!r} is gone")
            path, member = Path(entry["path"]), table if entry.get("tables") else None
        data = preview_rows(path, rows, member)
        return {"job_id": job_id, **({"table": table} if record.tables else {}), "rows": data,
                "returned": len(data), "total_records": total}

    @staticmethod
    def _describe(path: Path, kind: str) -> dict[str, Any]:
        return {"name": path.name, "kind": kind, "size_bytes": path.stat().st_size, "path": str(path.resolve())}

    def files(self, job_id: str) -> list[dict[str, Any]]:
        """Files of a finished job (the dataset and its quality report) that still exist."""
        return [f for f in self.get(job_id).files if Path(f["path"]).exists()]

    def file_path(self, job_id: str, name: str) -> Path:
        """Path of one listed file of ``job_id``.

        Raises:
            ConfigurationError: If the job has no file of that name.
        """
        for item in self.files(job_id):
            if item["name"] == name:
                return Path(item["path"])
        raise ConfigurationError(f"job {job_id} has no file {name!r}")

    def _fail(self, record: JobRecord, error_type: str | None, error: str | None, hint: str | None = None) -> None:
        with self._lock:
            record.status = "cancelled" if error_type == "CancelledError" else "failed"
            record.error, record.error_type, record.finished_at = error, error_type, utc_now_iso()
            record.hint = hint or "POST /jobs/{id}/resume retries only the unfinished batches"
            self._save(record)

    # -- queries ----------------------------------------------------------------------
    def get(self, job_id: str) -> JobRecord:
        """Return the record of ``job_id``.

        Raises:
            ConfigurationError: If the job is unknown.
        """
        record = self._jobs.get(job_id)
        if record is None:
            raise ConfigurationError(f"job {job_id} not found")
        return record

    def list(self) -> list[JobRecord]:
        """All jobs, newest first."""
        return sorted(self._jobs.values(), key=lambda r: r.created_at, reverse=True)

    def status(self, job_id: str) -> dict[str, Any]:
        """Record plus live progress (records done, percent, phase, sub-job)."""
        record = self.get(job_id)
        data = record.public()
        data["progress"] = self._progress(record)
        return data

    def _progress(self, record: JobRecord) -> dict[str, Any]:
        total = record.total_records
        if record.status == "completed":
            return {"records_done": record.records or total, "total_records": record.records or total, "percent": 100.0}
        done, phase, sub_job = 0, None, None
        run_dir = find_job_dir(self.settings.output_dir, record.job_id) if record.status != "queued" else None
        manifest = read_json(run_dir / "split_run.json", None) if run_dir else None
        if isinstance(manifest, dict):
            parts = manifest.get("parts", [])
            done = sum(p.get("record_count", 0) for p in parts if p.get("status") == "completed")
            current = next((p["part"] for p in parts if p.get("status") != "completed"), None)
            sub_job = {"current": current, "total": len(parts),
                       "completed": sum(p.get("status") == "completed" for p in parts)}
        multi = read_json(run_dir / "multi_table.json", None) if run_dir else None
        if isinstance(multi, dict):
            tables = multi.get("tables", [])
            done = sum(t.get("records") or 0 for t in tables if t.get("status") == "completed")
            total = max(sum(t.get("records") or t.get("estimate") or 0 for t in tables), 1)
            current = next((t["name"] for t in tables if t.get("status") != "completed"), None)
            sub_job = {"current_table": current, "tables": len(tables),
                       "completed": sum(t.get("status") == "completed" for t in tables)}
        engine = self._engines.get(record.job_id)
        ctx = getattr(engine, "_context", None) if engine is not None else None
        tracker = getattr(ctx, "tracker", None)
        if tracker is not None:
            done += max(tracker.valid_records, min(tracker.live_records, tracker.requested_records))
            phase = tracker.phase or "generating"
        elif record.status == "running":
            phase = "preparing (model checks, calibration) or merging"
        done = min(done, total)
        return {"records_done": done, "total_records": total, "percent": round(100.0 * done / total, 1),
                "phase": phase, **({("tables" if isinstance(multi, dict) else "sub_jobs"): sub_job} if sub_job else {})}

    def cancel(self, job_id: str) -> JobRecord:
        """Cancel a queued job, or ask a running one to stop gracefully (finished batches are kept)."""
        with self._lock:
            record = self.get(job_id)
            if record.status == "queued":
                future = self._futures.get(job_id)
                if future is not None:
                    future.cancel()
                record.status, record.finished_at = "cancelled", utc_now_iso()
                self._save(record)
            elif record.status == "running" and (engine := self._engines.get(job_id)) is not None:
                engine.cancel()
                record.hint = "cancellation requested; the job stops after in-flight work"
        return record

    def delete(self, job_id: str) -> None:
        """Delete a finished job: its record and its output folder.

        Raises:
            ConfigurationError: If the job is unknown or still active.
        """
        with self._lock:
            record = self.get(job_id)
            if record.status in ACTIVE:
                raise ConfigurationError(f"job {job_id} is {record.status}; cancel it first")
            job_dir = find_job_dir(self.settings.output_dir, job_id)
            if job_dir is not None:
                shutil.rmtree(job_dir, ignore_errors=True)
            (self.state_dir / f"{job_id}.json").unlink(missing_ok=True)
            del self._jobs[job_id]

    def wait(self, job_id: str, timeout: float) -> JobRecord:
        """Wait up to ``timeout`` seconds for a job to leave the queued/running state."""
        deadline = time.monotonic() + max(timeout, 0)
        while self.get(job_id).status in ACTIVE and time.monotonic() < deadline:
            time.sleep(0.2)
        return self.get(job_id)

    def shutdown(self) -> None:
        """Stop running jobs gracefully (they become ``cancelled`` and can be resumed)."""
        with self._lock:
            engines = list(self._engines.values())
        for engine in engines:
            engine.cancel()
        self._executor.shutdown(wait=False, cancel_futures=True)


def preview_rows(path: Path, rows: int, member: str | None = None) -> list[dict[str, Any]]:
    """Read the first ``rows`` records of a dataset file without loading the whole file.

    Handles ``.csv``, ``.json``, ``.jsonl`` and ``.xlsx``, plain, gzipped (``.gz``) or zipped (``.zip``).
    ``member`` picks the file of a zip archive by name without extension (a table of a multi-table job).
    """
    import io  # pylint: disable=import-outside-toplevel

    archive = None
    try:
        if path.suffix == ".zip":
            archive = zipfile.ZipFile(path)
            names = archive.namelist()
            chosen = next((n for n in names if member and n.split(".")[0] == member), names[0])
            stream, inner = archive.open(chosen), chosen
        elif path.suffix == ".gz":
            stream, inner = gzip.open(path, "rb"), path.name[:-3]
        else:
            stream, inner = path.open("rb"), path.name
        with stream:
            ext = Path(inner).suffix.lower()
            if ext == ".xlsx":
                return _preview_excel(stream if stream.seekable() else io.BytesIO(stream.read()), rows)
            with io.TextIOWrapper(stream, encoding="utf-8", newline="") as text:
                return _parse_text(text, ext.lstrip(".") or "csv", rows)
    finally:
        if archive is not None:
            archive.close()


def _preview_excel(handle: Any, rows: int) -> list[dict[str, Any]]:
    from openpyxl import load_workbook  # pylint: disable=import-outside-toplevel

    book = load_workbook(handle, read_only=True)
    try:
        values = book.worksheets[0].iter_rows(values_only=True)
        header = [str(c) for c in next(values, ())]
        out = []
        for row in values:
            out.append(dict(zip(header, row)))
            if len(out) >= rows:
                break
        return out
    finally:
        book.close()


def _parse_text(handle: Any, fmt: str, rows: int) -> list[dict[str, Any]]:
    import csv  # pylint: disable=import-outside-toplevel

    out: list[dict[str, Any]] = []
    if fmt == "csv":
        for row in csv.DictReader(handle):
            out.append({k: (None if v == "" else v) for k, v in row.items()})
            if len(out) >= rows:
                break
    elif fmt == "jsonl":
        for line in handle:
            if line.strip():
                out.append(json.loads(line))
            if len(out) >= rows:
                break
    else:  # a JSON array: decode objects one by one from a growing buffer
        decoder, buffer, pos, started = json.JSONDecoder(), "", 0, False
        while len(out) < rows:
            chunk = handle.read(65536)
            buffer = buffer[pos:] + chunk
            pos = 0
            while len(out) < rows:
                while pos < len(buffer) and buffer[pos] in " \t\r\n,":
                    pos += 1
                if not started and pos < len(buffer) and buffer[pos] == "[":
                    started, pos = True, pos + 1
                    continue
                if pos >= len(buffer) or buffer[pos] == "]":
                    break
                try:
                    item, end = decoder.raw_decode(buffer, pos)
                except json.JSONDecodeError:
                    break  # incomplete object: read more
                out.append(item)
                pos = end
            if not chunk or (pos < len(buffer) and buffer[pos] == "]"):
                break
    return out
