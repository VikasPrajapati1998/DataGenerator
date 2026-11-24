"""Nothing may block forever, and nothing may be left running after a job."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import psutil
import pytest

from app.config.model_registry import ModelConfig
from app.config.settings import Settings
from app.core.cancellation import CancellationToken
from app.core.exceptions import CancelledError, ProviderTimeoutError, WorkerCrashError
from app.models.mock_adapter import MockBehavior, MockLLMAdapter
from app.models.model_router import LLMUsage, ModelRouter
from app.monitoring.execution_tracker import ExecutionTracker
from app.workers.generation_worker import WorkItem
from app.workers.worker_manager import TaskExecutor
from tests.test_integration import BIO_SCHEMA

_SCHEMA = {"type": "object", "properties": {"a": {"type": "number"}}}


def _crash_entry(job_config_path: str, item: dict) -> dict:  # runs in a worker process
    os._exit(3)


def _hang_entry(job_config_path: str, item: dict) -> dict:  # runs in a worker process
    time.sleep(120)
    return {}


def _leftovers() -> tuple[list[str], list[int]]:
    threads = [t.name for t in threading.enumerate()
               if t.is_alive() and t.name in ("progress-monitor", "engine-graph")]
    children = [p.pid for p in psutil.Process().children(recursive=True) if p.is_running()]
    return threads, children


def _mock(name: str = "mock-llm") -> ModelConfig:
    return ModelConfig(provider="mock", model_name=name, model_id=name, retry={"max_attempts": 1})


def test_waiting_for_a_model_slot_is_bounded() -> None:
    slot = threading.BoundedSemaphore(1)
    slot.acquire()  # held forever, as if by a worker that died
    router = ModelRouter([_mock()], Settings(llm_slot_timeout=1), limiters={"mock/mock-llm": slot})
    started = time.perf_counter()
    with pytest.raises(ProviderTimeoutError, match="no free request slot"):
        router.generate_json("s", "u", _SCHEMA, LLMUsage())
    assert time.perf_counter() - started < 10


def test_waiting_for_a_slot_stops_on_cancellation() -> None:
    slot = threading.BoundedSemaphore(1)
    slot.acquire()
    cancel = CancellationToken()
    router = ModelRouter([_mock()], Settings(llm_slot_timeout=600), limiters={"mock/mock-llm": slot},
                         cancel=cancel)
    threading.Timer(0.5, cancel.cancel).start()
    started = time.perf_counter()
    with pytest.raises(CancelledError):
        router.generate_json("s", "u", _SCHEMA, LLMUsage())
    assert time.perf_counter() - started < 5


def test_watchdog_fires_when_nothing_progresses() -> None:
    fired = threading.Event()
    tracker = ExecutionTracker("job", 1, 10, sample_interval=0.1, stall_timeout=0.3,
                               on_stall=lambda idle: fired.set())
    tracker.task_started(1)
    tracker.start_monitoring()
    try:
        assert fired.wait(5)
    finally:
        tracker.stop_monitoring()
    assert tracker.stalled


def test_crashed_and_hung_worker_processes_are_replaced_and_cleaned_up(tmp_path: Path) -> None:
    item = WorkItem(task_id=1, start_index=0, record_count=1)
    crashing = TaskExecutor("process", None, tmp_path / "job_config.json", 1, {}, entry=_crash_entry)
    try:
        with pytest.raises(WorkerCrashError, match="died"):
            crashing.run(item)
        assert crashing.restarts == 1  # the pool was replaced, later tasks can still run
    finally:
        crashing.shutdown(force=True)

    hanging = TaskExecutor("process", None, tmp_path / "job_config.json", 1, {}, task_timeout=2, entry=_hang_entry)
    started = time.perf_counter()
    try:
        with pytest.raises(WorkerCrashError, match="timed out"):
            hanging.run(item)
    finally:
        hanging.shutdown(force=True)
    assert time.perf_counter() - started < 60
    assert _leftovers()[1] == []  # no worker or manager process survives


def test_stalled_job_is_stopped_by_the_watchdog(settings, registry) -> None:
    from app.engine import SyntheticDataEngine  # pylint: disable=import-outside-toplevel

    MockLLMAdapter.behaviors["mock-llm"] = MockBehavior(latency=8.0)  # every model call "hangs" for 8 s
    engine = SyntheticDataEngine(settings=settings.model_copy(update={"stall_timeout": 1}), registry=registry)
    request = {"schema": BIO_SCHEMA, "total_records": 16, "batch_size": 16, "model_name": "mock-llm",
               "model_provider": "mock", "execution_mode": "thread"}
    started = time.perf_counter()
    result = engine.run(request)
    assert result.status == "failed" and "stalled" in (result.error or "")
    assert time.perf_counter() - started < 60
    assert _leftovers() == ([], [])


def test_finished_and_cancelled_jobs_leave_nothing_running(engine, employee_request) -> None:
    assert engine.run(employee_request).ok
    assert _leftovers() == ([], [])

    request = {**employee_request, "total_records": 90_000, "batch_size": 500, "execution_mode": "process"}
    threading.Timer(3.0, engine.cancel).start()
    result = engine.run(request)
    assert result.status in ("cancelled", "completed")
    assert _leftovers() == ([], [])
