"""Public engine API: run, resume and cancel generation jobs."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError as PydanticValidationError

from app.config.model_registry import ModelRegistry, load_registry
from app.config.settings import Settings, get_settings
from app.core.exceptions import (
    CancelledError,
    ConfigurationError,
    EngineError,
    ModelUnavailableError,
    ResourceError,
    SchemaError,
    StorageError,
)
from app.graph.graph_builder import build_graph
from app.graph.state import EngineContext
from app.monitoring.logger import configure_logging, get_logger, redact
from app.multi_table import MANIFEST as MULTI_MANIFEST
from app.multi_table import MultiTableRunner, is_multi_table
from app.schemas.input_schema import GenerationRequest
from app.split_runner import MANIFEST as SPLIT_MANIFEST
from app.split_runner import SplitRunner
from app.storage.file_manager import find_job_dir, write_json_atomic
from app.utils.helpers import utc_now_iso

logger = get_logger(__name__)
_RECURSION_LIMIT = 400
_JOIN_POLL_SECONDS = 0.5
_GRAPH_ERROR_TYPES = {"failed": "TaskExecutionError", "cancelled": "CancelledError"}
"""``error_type`` of a graph that finished without an exception (batches failed or the job was stopped)."""
_USER_FIXABLE = (SchemaError, ConfigurationError, ModelUnavailableError, ResourceError, CancelledError)


@dataclass
class JobResult:
    """Outcome of one engine run."""

    status: str
    job_id: str | None
    job_dir: str | None
    final_output: str | None
    summary: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    error_type: str | None = None

    @property
    def ok(self) -> bool:
        """Whether the dataset was delivered."""
        return self.status == "completed"


class SyntheticDataEngine:
    """Entry point that wires settings, registry and the LangGraph workflow.

    Args:
        settings: Application settings (defaults to the environment).
        registry: Model registry (defaults to ``models.json``).
    """

    def __init__(self, settings: Settings | None = None, registry: ModelRegistry | None = None) -> None:
        self.settings = settings or get_settings()
        configure_logging(self.settings.log_dir, self.settings.log_level)
        self.registry = registry or load_registry()
        self.graph = build_graph()
        self._context: EngineContext | None = None
        self.cancel_requested = False
        # Model availability results, reused by the sub-jobs of one split run (cleared for every new run).
        self.availability_cache: dict[str, Any] = {}
        self.in_split_run = False

    def run(self, request: dict[str, Any] | GenerationRequest) -> JobResult:
        """Generate a dataset for ``request``.

        Args:
            request: Request dictionary (JSON shape) or model.

        Returns:
            The job result (``status`` is ``completed``, ``failed`` or ``cancelled``).
        """
        model = request if isinstance(request, GenerationRequest) else GenerationRequest(**request)
        if model.links is None and is_multi_table(model.data_schema):
            self.cancel_requested = False
            return MultiTableRunner(self).start(model)
        split_size = model.split_size if model.split_size is not None else self.settings.max_records_per_job
        if model.sub_job is None and split_size and model.total_records > split_size:
            self.cancel_requested = False
            return SplitRunner(self).start(model, split_size)
        return self._invoke({"request": model.model_dump(by_alias=True), "resume": False})

    def resume(self, job: str | Path) -> JobResult:
        """Resume an interrupted or failed job (or split run); completed batches are reused.

        Args:
            job: Job id, split-run id, or a job / split-run directory.

        Returns:
            The job result.

        Raises:
            ConfigurationError: If the job directory does not exist.
        """
        job_dir = find_job_dir(self.settings.output_dir, job)
        if job_dir is None:
            raise ConfigurationError(f"Job {job} not found under {self.settings.output_dir}")
        if (job_dir / MULTI_MANIFEST).exists():
            self.cancel_requested = False
            return MultiTableRunner(self).resume(job_dir)
        if (job_dir / SPLIT_MANIFEST).exists():
            self.cancel_requested = False
            return SplitRunner(self).resume(job_dir)
        return self._invoke({"resume": True, "job_dir": str(job_dir)})

    def cancel(self) -> None:
        """Request graceful cancellation of the running job (and of the remaining sub-jobs of a split run)."""
        self.cancel_requested = True
        if self._context is not None:
            logger.warning("Cancellation requested; finishing in-flight work and stopping")
            self._context.cancel.cancel()

    def shutdown_now(self) -> None:
        """Stop the running job immediately and release its processes (signal and ``atexit`` handlers)."""
        ctx = self._context
        if ctx is not None:
            self.cancel_requested = True
            ctx.cancel.cancel()
            ctx.close(force=True)

    def _invoke(self, initial: dict[str, Any]) -> JobResult:
        if not self.in_split_run:
            self.availability_cache.clear()
        ctx = EngineContext(settings=self.settings, registry=self.registry)
        ctx.options["availability_cache"] = self.availability_cache
        self._context = ctx
        outcome: dict[str, Any] = {}

        def run_graph() -> None:
            try:
                outcome["state"] = self.graph.invoke(
                    initial, context=ctx,
                    config={"max_concurrency": max(os.cpu_count() or 1, 1), "recursion_limit": _RECURSION_LIMIT},
                )
            except BaseException as exc:  # noqa: BLE001 - handed to the calling thread and re-raised there
                outcome["error"] = exc

        # The graph runs in a helper thread so the calling (main) thread waits with a timeout. A blocked
        # main thread would never see Ctrl+C / SIGTERM on Windows, so the job could not be stopped.
        runner = threading.Thread(target=run_graph, name="engine-graph", daemon=True)
        clean = False
        try:
            runner.start()
            while runner.is_alive():
                runner.join(_JOIN_POLL_SECONDS)
            if "error" in outcome:
                raise outcome["error"]
            state = outcome["state"]
            clean = True
            status = state.get("status", "failed")
            return JobResult(status=status, job_id=state.get("job_id"),
                             job_dir=str(ctx.paths.root) if ctx.paths else None,
                             final_output=state.get("final_output"), summary=state.get("summary", {}),
                             error=redact("; ".join(state.get("errors", []))) or None,
                             error_type=_GRAPH_ERROR_TYPES.get(status))
        except KeyboardInterrupt:
            logger.warning("Interrupted: stopping workers and releasing resources")
            ctx.cancel.cancel()
            raise
        except _USER_FIXABLE as exc:
            # Bad schema, no model, not enough resources...: the message says what to do. The console shows
            # just this line; the stack trace goes to the log file only (see ConsoleFormatter).
            logger.error("Job failed: %s", redact(str(exc)), exc_info=True)
            return self._failed(ctx, exc)
        except EngineError as exc:
            # Other expected failures (disk full, merge refused...): report with details, don't crash.
            logger.error("Job failed: %s", redact(f"{type(exc).__name__}: {exc}"), exc_info=True)
            return self._failed(ctx, exc)
        except PydanticValidationError as exc:
            # Typically a hand-edited or corrupt job_config.json on resume.
            logger.error("Job failed: invalid stored configuration: %s", exc, exc_info=True)
            return self._failed(ctx, ConfigurationError(f"invalid job configuration: {exc.errors()[0]['msg']}"))
        except OSError as exc:
            logger.error("Job failed: file system error: %s", exc, exc_info=True)
            return self._failed(ctx, StorageError(f"file system error: {exc}"))
        except Exception as exc:
            # A bug: record it in the job folder for diagnosis, then let it propagate.
            logger.critical("Unexpected engine error: %s", exc, exc_info=True)
            self._write_failure(ctx, f"unexpected {type(exc).__name__}: {exc}")
            raise
        finally:
            # A graph that ended normally has already stopped its workers; anything else is torn down
            # forcefully so no process, thread or model stays busy after the job.
            if not clean and ctx.paths is not None:
                ctx.cancel.cancel()  # tells any still-running queue thread or worker process to stop
            ctx.close(force=not clean)
            self._context = None

    def _failed(self, ctx: EngineContext, exc: Exception) -> JobResult:
        message = redact(f"{type(exc).__name__}: {exc}")
        self._write_failure(ctx, message)
        return JobResult(status="failed", job_id=ctx.paths.root.name if ctx.paths else None,
                         job_dir=str(ctx.paths.root) if ctx.paths else None, final_output=None, error=message,
                         error_type=type(exc).__name__)

    @staticmethod
    def _write_failure(ctx: EngineContext, message: str) -> None:
        if ctx.paths is None:
            return
        try:
            write_json_atomic(ctx.paths.metadata / "execution_summary.json",
                              {"status": "failed", "error": redact(message), "finished_at": utc_now_iso()})
        except StorageError as exc:
            logger.error("Could not write the failure summary: %s", exc)
