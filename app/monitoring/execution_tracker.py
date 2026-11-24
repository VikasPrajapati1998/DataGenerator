"""Thread-safe execution statistics, resource sampling and progress reporting."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import psutil

from app.models.model_router import LLMUsage
from app.monitoring.logger import get_logger
from app.scheduler.cpu_detector import query_gpus
from app.utils.helpers import format_duration

logger = get_logger(__name__)
_LLM_SILENCE_WARNING_SECONDS = 60.0


@dataclass
class _Samples:
    cpu: list[float] = field(default_factory=list)
    ram_mb: list[float] = field(default_factory=list)
    process_ram_mb: list[float] = field(default_factory=list)
    gpu: list[float] = field(default_factory=list)


def _duration(seconds: float | None) -> str:
    """Readable duration (``45s``, ``12m05s``, ``3h20m``, ``4d07h``), or ``n/a`` when unknown."""
    return "n/a" if seconds is None else format_duration(seconds)


class ExecutionTracker:
    """Aggregates task outcomes and resource usage for one job.

    Args:
        job_id: Job identifier (for log lines).
        total_tasks: Planned tasks.
        requested_records: Requested records.
        sample_interval: Seconds between resource samples / progress lines.
    """

    def __init__(self, job_id: str, total_tasks: int, requested_records: int, sample_interval: float = 5.0,
                 stall_timeout: float = 0.0, on_stall: Callable[[float], None] | None = None) -> None:
        self.job_id = job_id
        self.stall_timeout = stall_timeout
        self.on_stall = on_stall
        self.stalled = False
        self.last_activity = time.perf_counter()
        self.total_tasks = total_tasks
        self.requested_records = requested_records
        self.sample_interval = sample_interval
        self._lock = threading.Lock()
        self.started = time.perf_counter()
        self.finished: float | None = None
        self.active: dict[int, float] = {}
        self.completed: set[int] = set()
        self.failed: set[int] = set()
        self.skipped: set[int] = set()
        self.split_tasks: set[int] = set()
        self.valid_records = 0
        self.rejected_records = 0
        self.duplicates_detected = 0
        self.semantic_duplicates = 0
        self.regeneration_rounds = 0
        self.task_retries = 0
        self.task_durations: list[float] = []
        self.llm = LLMUsage()
        # Live LLM counters (thread mode): updated per call, not per finished task.
        self.live_calls = 0
        self.live_failures = 0
        self.live_latency = 0.0
        self.live_records = 0
        self.llm_waiting = 0
        self.llm_required = False
        self.last_llm_success: float | None = None
        self.phase = ""  # e.g. "retry round 1/2", shown in the progress line
        self._samples = _Samples()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._process = psutil.Process()
        self._gpu_available = bool(query_gpus())

    # -- task lifecycle --------------------------------------------------------------
    def task_started(self, task_id: int) -> None:
        """Record that ``task_id`` started."""
        with self._lock:
            self.active[task_id] = self.last_activity = time.perf_counter()

    def on_llm_event(self, event: str, value: float = 0.0) -> None:
        """Live LLM event from a worker thread (see ``LLMUsage.listener``)."""
        with self._lock:
            if event != "waiting":
                self.last_activity = time.perf_counter()
            if event == "call":
                self.live_calls += 1
                self.live_latency += value
                self.last_llm_success = self.last_activity
            elif event == "failure":
                self.live_failures += 1
            elif event == "waiting":
                self.llm_waiting += int(value)
            elif event == "records":
                self.live_records += int(value)

    def task_finished(self, result: dict[str, Any]) -> None:
        """Record a task result (from a thread or a worker process).

        Args:
            result: The task result dictionary produced by the worker.
        """
        task_id = int(result["task_id"])
        with self._lock:
            self.active.pop(task_id, None)
            self.last_activity = time.perf_counter()
            status = result.get("status")
            if result.get("part"):
                # Sub-batch of a split task: count its records; the task completes on consolidation.
                if status == "completed" and not result.get("skipped"):
                    self.valid_records += int(result.get("record_count", 0))
                self.rejected_records += int(result.get("rejected_count", 0))
                self.duplicates_detected += int(result.get("duplicate_rejections", 0))
                self.semantic_duplicates += int(result.get("semantic_rejections", 0))
                self.llm.merge(result.get("llm_usage", {}))
                return
            if result.get("skipped"):
                self.skipped.add(task_id)
                self.completed.add(task_id)
                self.failed.discard(task_id)
                return
            if status == "completed":
                self.completed.add(task_id)
                self.failed.discard(task_id)
                self.valid_records += int(result.get("record_count", 0))
            elif status == "failed":
                self.failed.add(task_id)
            self.rejected_records += int(result.get("rejected_count", 0))
            self.duplicates_detected += int(result.get("duplicate_rejections", 0))
            self.semantic_duplicates += int(result.get("semantic_rejections", 0))
            self.regeneration_rounds += int(result.get("regeneration_rounds", 0))
            if result.get("duration"):
                self.task_durations.append(float(result["duration"]))
            self.llm.merge(result.get("llm_usage", {}))

    def task_consolidated(self, task_id: int) -> None:
        """Mark a split task as completed once its sub-batches were merged."""
        with self._lock:
            self.completed.add(task_id)
            self.failed.discard(task_id)
            self.split_tasks.add(task_id)

    def add_llm_usage(self, usage: LLMUsage) -> None:
        """Merge job-level LLM usage (e.g. profile calibration)."""
        with self._lock:
            self.llm.merge(usage)

    def note_retry(self, count: int = 1) -> None:
        """Count graph-level task retries."""
        with self._lock:
            self.task_retries += count

    # -- monitoring thread -------------------------------------------------------------
    def start_monitoring(self) -> None:
        """Start the background sampling/progress thread (idempotent)."""
        if self._thread is not None:
            return
        psutil.cpu_percent(interval=None)
        self._thread = threading.Thread(target=self._loop, name="progress-monitor", daemon=True)
        self._thread.start()

    def stop_monitoring(self) -> None:
        """Stop the background thread and freeze the end time."""
        if self._thread is not None and not self._stop.is_set():
            self.sample()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.sample_interval + 1)
            self._thread = None
        if self.finished is None:
            self.finished = time.perf_counter()

    def _loop(self) -> None:
        while not self._stop.wait(self.sample_interval):
            self.sample()
            logger.info(self.progress_line())
            self._check_stall()

    def _check_stall(self) -> None:
        """Watchdog: fire ``on_stall`` once when nothing has happened for ``stall_timeout`` seconds."""
        if not self.stall_timeout or self.stalled or self.on_stall is None:
            return
        with self._lock:
            idle = time.perf_counter() - self.last_activity
            busy = bool(self.active)
        if busy and idle > self.stall_timeout:
            self.stalled = True
            logger.error("Watchdog: no progress for %.0fs with %d task(s) running; stopping the job "
                         "(SDE_STALL_TIMEOUT=%.0f)", idle, len(self.active), self.stall_timeout)
            self.on_stall(idle)

    def set_phase(self, phase: str) -> None:
        """Name the current stage for the progress line (``""`` = first generation round)."""
        with self._lock:
            self.phase = phase
            self.last_activity = time.perf_counter()

    def touch(self) -> None:
        """Record activity that is not a task or LLM event (e.g. a new dispatch round)."""
        with self._lock:
            self.last_activity = time.perf_counter()

    def sample(self) -> None:
        """Take one resource sample."""
        cpu = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        try:
            rss = self._process.memory_info().rss + sum(
                c.memory_info().rss for c in self._process.children(recursive=True))
        except psutil.Error:
            rss = 0
        gpu = query_gpus() if self._gpu_available else []
        with self._lock:
            self._samples.cpu.append(cpu)
            self._samples.ram_mb.append((memory.total - memory.available) / 1048576)
            self._samples.process_ram_mb.append(rss / 1048576)
            if gpu:
                self._samples.gpu.append(sum(g.utilization_pct for g in gpu) / len(gpu))

    def progress_line(self) -> str:
        """Return a one-line progress summary."""
        with self._lock:
            now = time.perf_counter()
            elapsed = now - self.started
            done = len(self.completed)
            # Records finished so far: whole batches, or (while batches are still running) records whose
            # LLM fields are already filled - so long LLM-bound tasks still show movement.
            progressed = max(self.valid_records, min(self.live_records, self.requested_records))
            rate = progressed / elapsed if elapsed else 0.0
            eta = (self.requested_records - progressed) / rate if rate > 0 else None
            avg_task = sum(self.task_durations) / len(self.task_durations) if self.task_durations else None
            oldest = max((now - started for started in self.active.values()), default=0.0)
            # During retry rounds the failed tasks are being redone, so "failed" would read as final.
            failed = f"{len(self.failed)} being retried" if self.phase else f"{len(self.failed)} failed"
            stage = f" {self.phase}:" if self.phase else ""
            parts = [
                f"[{self.job_id}]{stage} tasks {done}/{self.total_tasks} done, "
                f"{len(self.active)} active, {failed}",
                f"records {progressed:,}/{self.requested_records:,} ({rate:,.1f}/s)",
                f"task avg {_duration(avg_task)}, oldest running {_duration(oldest)}",
                f"rejected {self.rejected_records:,}",
            ]
            if self.llm_required or self.live_calls:
                avg_call = self.live_latency / self.live_calls if self.live_calls else None
                parts.append(f"LLM calls {self.live_calls or self.llm.calls} (avg {_duration(avg_call)}, "
                             f"{self.live_failures} failed, {self.llm_waiting} waiting for a slot)")
            parts.append(f"ETA {_duration(eta)}")
            silent = now - (self.last_llm_success or self.started)
            if self.llm_required and self.active and silent > _LLM_SILENCE_WARNING_SECONDS:
                parts.append(f"WARNING: no LLM answer for {_duration(silent)}")
            return " | ".join(parts)

    # -- reporting -------------------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        """Return the execution statistics as a JSON-serialisable dict."""
        with self._lock:
            end = self.finished or time.perf_counter()
            elapsed = end - self.started
            samples = self._samples

            def stats(values: list[float]) -> dict[str, float] | None:
                if not values:
                    return None
                return {"avg": round(sum(values) / len(values), 1), "peak": round(max(values), 1)}

            durations = sorted(self.task_durations)
            return {
                "requested_records": self.requested_records,
                "dynamic_tasks": self.total_tasks,
                "completed_tasks": len(self.completed),
                "failed_tasks": sorted(self.failed),
                "skipped_tasks_already_complete": len(self.skipped),
                "split_tasks": sorted(self.split_tasks),
                "valid_records_generated": self.valid_records,
                "rejected_records": self.rejected_records,
                "duplicate_rejections": self.duplicates_detected,
                "semantic_rejections": self.semantic_duplicates,
                "regeneration_rounds": self.regeneration_rounds,
                "task_retries": self.task_retries,
                "execution_time_seconds": round(elapsed, 3),
                "records_per_second": round(self.valid_records / elapsed, 1) if elapsed else 0.0,
                "task_duration_seconds": {
                    "min": round(durations[0], 3), "median": round(durations[len(durations) // 2], 3),
                    "max": round(durations[-1], 3)} if durations else None,
                "llm_usage": self.llm.to_dict(),
                "llm_live": {"calls": self.live_calls, "failures": self.live_failures,
                             "avg_latency_seconds": round(self.live_latency / self.live_calls, 3)
                             if self.live_calls else None},
                "avg_llm_latency_seconds": round(self.llm.latency_total / self.llm.calls, 3) if self.llm.calls else None,
                "cpu_utilization_pct": stats(samples.cpu),
                "system_ram_used_mb": stats(samples.ram_mb),
                "engine_ram_mb": stats(samples.process_ram_mb),
                "gpu_utilization_pct": stats(samples.gpu) if self._gpu_available else "no GPU detected",
            }
