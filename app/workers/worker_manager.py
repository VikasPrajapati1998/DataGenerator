"""Drives logical core queues and executes their tasks in threads or processes.

``thread`` mode runs each task inside the LangGraph worker thread that owns
the queue (best for LLM-bound work: inference happens in the Ollama server,
so the GIL is not the bottleneck). ``process`` mode keeps the scheduling
thread but hands each task to a dedicated worker process (one slot per
queue), giving true multi-core parallelism for CPU-heavy Faker/validation
work. In both modes a queue never has more than one task in flight.
"""

from __future__ import annotations

import multiprocessing
import signal
import threading
import time
import traceback
from collections.abc import Callable
from concurrent.futures import CancelledError as FuturesCancelledError
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from pickle import PicklingError
from typing import Any

from app.core.cancellation import CancellationToken
from app.core.exceptions import EngineError, WorkerCrashError
from app.models.rate_limiter import Limiter
from app.monitoring.execution_tracker import ExecutionTracker
from app.monitoring.logger import configure_logging, current_log_file, get_logger, redact
from app.scheduler.affinity import core_index, pin_current_thread, unpin_current_thread
from app.scheduler.work_stealing import CoreQueueScheduler
from app.storage.batch_writer import STATUS_CANCELLED, STATUS_COMPLETED, STATUS_FAILED
from app.workers.generation_worker import WorkItem, run_task
from app.workers.services import JobSpec, WorkerServices

logger = get_logger(__name__)
_MAX_POOL_RESTARTS = 2
_POLL_SECONDS = 1.0
_CANCEL_GRACE_SECONDS = 60.0
_SHUTDOWN_GRACE_SECONDS = 5.0
_KILL_WAIT_SECONDS = 3.0

_PROCESS_SERVICES: dict[str, WorkerServices] = {}
_PROCESS_LIMITERS: dict[str, Limiter] | None = None
_PROCESS_RATE_DIVISOR = 1


def init_worker_process(log_file: str | None, limiters: dict[str, Limiter] | None, rate_divisor: int) -> None:
    """Initialise a worker process (logging and shared LLM limiters).

    Args:
        log_file: Parent's log file to append to.
        limiters: Cross-process semaphores per model key.
        rate_divisor: Number of processes sharing provider rate limits.
    """
    global _PROCESS_LIMITERS, _PROCESS_RATE_DIVISOR  # pylint: disable=global-statement
    # Ctrl+C reaches every process of the console; only the parent decides how to stop (cancel marker,
    # then terminate), so a worker never dies half-way through writing a batch because of it.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    configure_logging(log_file=Path(log_file) if log_file else None, console=False)
    _PROCESS_LIMITERS = limiters
    _PROCESS_RATE_DIVISOR = rate_divisor


def process_entry(job_config_path: str, item: dict[str, Any]) -> dict[str, Any]:
    """Run one task inside a worker process.

    Args:
        job_config_path: Path of the job's ``job_config.json``.
        item: Serialised :class:`WorkItem`.

    Returns:
        The task result.
    """
    services = _PROCESS_SERVICES.get(job_config_path)
    if services is None:
        services = WorkerServices(JobSpec.load(Path(job_config_path)), _PROCESS_LIMITERS, _PROCESS_RATE_DIVISOR)
        _PROCESS_SERVICES[job_config_path] = services
    return run_task(WorkItem(**item), services)


class TaskExecutor:
    """Executes work items either in the calling thread or in a process pool.

    Args:
        mode: ``thread`` or ``process``.
        services: Parent-process services (used in thread mode).
        job_config_path: Frozen job configuration (used by worker processes).
        workers: Process-pool size (number of queues).
        llm_slots: Per-model LLM concurrency for process mode.
    """

    def __init__(self, mode: str, services: WorkerServices, job_config_path: Path, workers: int,
                 llm_slots: dict[str, int], task_timeout: float | None = None,
                 cancel: CancellationToken | None = None, entry: Callable[..., dict[str, Any]] | None = None) -> None:
        self.mode = mode
        self.services = services
        self.job_config_path = str(job_config_path)
        self.workers = workers
        self.task_timeout = task_timeout
        self.cancel = cancel
        self._entry = entry or process_entry
        self._pool: ProcessPoolExecutor | None = None
        self._manager = None
        self._limiters: dict[str, Limiter] = {}
        self._lock = threading.Lock()
        self.restarts = 0
        if mode == "process":
            try:
                self._manager = multiprocessing.Manager()
                self._limiters = {key: self._manager.BoundedSemaphore(slots) for key, slots in llm_slots.items()}
            except (OSError, EOFError) as exc:
                raise WorkerCrashError(f"Cannot start the process manager: {exc}") from exc
            self._pool = self._new_pool()
            logger.info("Started %d worker processes", workers)

    def _new_pool(self) -> ProcessPoolExecutor:
        log_file = current_log_file()
        try:
            return ProcessPoolExecutor(max_workers=self.workers, initializer=init_worker_process,
                                       initargs=(str(log_file) if log_file else None, self._limiters, self.workers))
        except (OSError, ValueError) as exc:
            raise WorkerCrashError(f"Cannot start {self.workers} worker processes: {exc}") from exc

    def run(self, item: WorkItem) -> dict[str, Any]:
        """Execute ``item`` and return its result.

        Raises:
            WorkerCrashError: If the worker process died. The pool is replaced
                (up to ``_MAX_POOL_RESTARTS`` times) so later tasks can still run.
        """
        if self.mode != "process":
            return run_task(item, self.services)
        with self._lock:
            pool = self._pool
        if pool is None:
            raise WorkerCrashError("Worker processes are not running (executor shut down)")
        try:
            future = pool.submit(self._entry, self.job_config_path, item.to_dict())
            return self._wait(future, pool, item)
        except BrokenProcessPool as exc:
            self._restart(pool)
            raise WorkerCrashError(f"Worker process died while running {item.label}: {exc}") from exc
        except FuturesCancelledError as exc:
            raise WorkerCrashError(f"{item.label} was cancelled by a pool shutdown") from exc
        except (PicklingError, AttributeError, ImportError) as exc:
            raise WorkerCrashError(f"{item.label} could not be sent to a worker process: {exc}") from exc

    def _wait(self, future: Future, pool: ProcessPoolExecutor, item: WorkItem) -> dict[str, Any]:
        """Wait for a worker-process result without ever blocking forever.

        Polls once per second so that a task timeout or a cancellation that the
        worker does not honour (hung native code) ends in killed processes and
        a failed task instead of a hang.
        """
        started = time.monotonic()
        cancelled_at: float | None = None
        while True:
            try:
                return future.result(timeout=_POLL_SECONDS)
            except FuturesTimeoutError:
                now = time.monotonic()
                if self.task_timeout and now - started > self.task_timeout:
                    self._kill(pool, f"{item.label} exceeded the task timeout of {self.task_timeout:.0f}s")
                    raise WorkerCrashError(f"{item.label} timed out after {self.task_timeout:.0f}s; "
                                           "its worker processes were terminated") from None
                if self.cancel is not None and self.cancel.is_cancelled():
                    cancelled_at = cancelled_at or now
                    if now - cancelled_at > _CANCEL_GRACE_SECONDS:
                        self._kill(pool, f"{item.label} ignored cancellation for {_CANCEL_GRACE_SECONDS:.0f}s")
                        raise WorkerCrashError(f"{item.label} did not stop after cancellation; "
                                               "its worker processes were terminated") from None

    def _kill(self, pool: ProcessPoolExecutor, reason: str) -> None:
        logger.error("Terminating worker processes: %s", reason)
        _terminate_pool(pool, grace=0)
        self._restart(pool)

    def _restart(self, broken: ProcessPoolExecutor) -> None:
        with self._lock:
            if self._pool is not broken:  # another queue already replaced it
                return
            broken.shutdown(wait=False, cancel_futures=True)
            if self.cancel is not None and self.cancel.is_cancelled():
                self._pool = None  # the job is stopping: do not start new processes
                return
            if self.restarts >= _MAX_POOL_RESTARTS:
                logger.error("Process pool broke %d times; not restarting it again", self.restarts + 1)
                self._pool = None
                return
            self.restarts += 1
            logger.warning("Process pool broke; starting a new one (restart %d/%d)", self.restarts,
                           _MAX_POOL_RESTARTS)
            self._pool = self._new_pool()

    def shutdown(self, force: bool = False) -> None:
        """Stop worker processes and the manager process (no-op in thread mode).

        Never blocks indefinitely: idle workers get a short grace period, then any
        process still alive (hung or busy) is terminated, then killed.

        Args:
            force: Skip the grace period (failure, cancellation, interpreter exit).
        """
        with self._lock:
            pool, self._pool = self._pool, None
        if pool is not None:
            _terminate_pool(pool, grace=0 if force else _SHUTDOWN_GRACE_SECONDS)
        if self._manager is not None:
            try:
                self._manager.shutdown()
            except (OSError, EOFError) as exc:
                logger.warning("Process manager did not shut down cleanly: %s", exc)
            self._manager = None


def _terminate_pool(pool: ProcessPoolExecutor, grace: float) -> None:
    """Shut a process pool down without waiting on hung workers; terminate what remains."""
    processes = list(getattr(pool, "_processes", {}).values())  # snapshot before shutdown clears it
    pool.shutdown(wait=False, cancel_futures=True)
    deadline = time.monotonic() + grace
    for process in processes:
        process.join(timeout=max(0.0, deadline - time.monotonic()))
    alive = [p for p in processes if p.is_alive()]
    for process in alive:
        process.terminate()
    for process in alive:
        process.join(timeout=_KILL_WAIT_SECONDS)
        if process.is_alive():
            process.kill()
            process.join(timeout=_KILL_WAIT_SECONDS)
    if alive:
        logger.warning("Terminated %d worker process(es) that did not exit on their own", len(alive))


def run_core_queue(core_id: str, scheduler: CoreQueueScheduler, executor: TaskExecutor,
                   tracker: ExecutionTracker, cancel: CancellationToken, pin: bool = False) -> list[dict[str, Any]]:
    """Run one logical core queue to completion (one task at a time).

    Args:
        core_id: Queue to drive.
        scheduler: Scheduler owning all queues.
        executor: Task executor.
        tracker: Execution tracker.
        cancel: Cancellation token.
        pin: Pin this thread to the matching logical CPU.

    Returns:
        Results of every task this queue executed (including stolen ones and
        cancelled placeholders).
    """
    pinned = pin and executor.mode == "thread" and pin_current_thread(core_index(core_id))

    def placeholder(item: WorkItem, status: str, error: str) -> dict[str, Any]:
        return {"task_id": item.task_id, "part": item.part, "kind": item.kind, "attempt": item.attempt,
                "model_key": item.model_key, "status": status, "record_count": 0, "core_id": core_id,
                "error": error}

    def execute(item: WorkItem) -> tuple[dict[str, Any], bool]:
        tracker.task_started(item.task_id)
        try:
            result = executor.run(item)
        except EngineError as exc:
            # Worker crash or a failure the task itself could not record: fail this item, keep the queue going.
            logger.error("%s failed on %s: %s", item.label, core_id, redact(str(exc)), exc_info=True)
            result = placeholder(item, STATUS_FAILED, redact(f"{type(exc).__name__}: {exc}"))
        except (OSError, EOFError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            logger.error("%s crashed on %s: %s\n%s", item.label, core_id, exc, traceback.format_exc())
            result = placeholder(item, STATUS_FAILED, redact(f"worker crash: {type(exc).__name__}: {exc}"))
        result["core_id"] = core_id
        tracker.task_finished(result)
        return result, result.get("status") == STATUS_COMPLETED

    try:
        results, skipped = scheduler.run_queue(core_id, execute, cancel)
    finally:
        if pinned:
            unpin_current_thread()
    results.extend(placeholder(item, STATUS_CANCELLED, "cancelled") for item in skipped)
    return results
