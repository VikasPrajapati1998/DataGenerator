"""Exception hierarchy and error handling at each layer."""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np
import pytest

from app.config.model_registry import ModelConfig
from app.config.settings import Settings
from app.core import exceptions
from app.core.cancellation import CancellationToken
from app.core.exceptions import (
    ContextLengthError,
    EngineError,
    InvalidModelOutputError,
    ModelUnavailableError,
    ProviderError,
    ProviderTimeoutError,
    QuotaExceededError,
    RateLimitError,
    StorageError,
    WorkerCrashError,
)
from app.main import main
from app.models.ollama_local import OllamaLocalAdapter
from app.monitoring.execution_tracker import ExecutionTracker
from app.monitoring.logger import redact
from app.scheduler.work_stealing import CoreQueueScheduler
from app.guardrails.semantic_deduplication import load_signatures, save_signatures
from app.storage.batch_writer import verified_completion
from app.storage.file_manager import JobPaths, read_json
from app.utils.hashing import file_sha256
from app.workers.worker_manager import run_core_queue


def test_every_engine_exception_derives_from_engine_error() -> None:
    classes = [c for _, c in inspect.getmembers(exceptions, inspect.isclass) if issubclass(c, Exception)]
    assert all(issubclass(c, EngineError) for c in classes)
    assert issubclass(WorkerCrashError, exceptions.TaskExecutionError)
    assert issubclass(exceptions.RecoveryError, StorageError)


def _adapter(handler) -> OllamaLocalAdapter:
    config = ModelConfig(provider="ollama_local", model_name="m", model_id="m", timeout=5)
    adapter = OllamaLocalAdapter(config, Settings())
    adapter._client = httpx.Client(base_url="http://ollama.test", transport=httpx.MockTransport(handler))
    return adapter


@pytest.mark.parametrize(("response", "error"), [
    (httpx.Response(429, headers={"retry-after": "7"}, text="slow down"), RateLimitError),
    (httpx.Response(402, json={"error": "Add usage credits or subscribe"}), QuotaExceededError),
    (httpx.Response(429, text="insufficient_quota: you exceeded your plan"), QuotaExceededError),
    (httpx.Response(400, json={"error": "This model's maximum context length is 8192 tokens"}), ContextLengthError),
    (httpx.Response(413, text="prompt is too long"), ContextLengthError),
    (httpx.Response(500, text="boom"), ProviderError),
    (httpx.Response(503, text="queue full"), ProviderError),
    (httpx.Response(404, json={"error": "model 'm' not found"}), ModelUnavailableError),
    (httpx.Response(401, text="unauthorized"), ModelUnavailableError),
    (httpx.Response(504, text="gateway"), ProviderTimeoutError),
    (httpx.Response(200, text="<html>not json</html>"), ProviderError),
    (httpx.Response(200, json={"message": {"content": ""}}), InvalidModelOutputError),
])
def test_http_failures_are_normalised(response: httpx.Response, error: type[Exception]) -> None:
    adapter = _adapter(lambda request: response)
    with pytest.raises(error) as caught:
        adapter.generate_json("s", "u", {"type": "object"})
    if error is RateLimitError:
        assert caught.value.retry_after == 7.0


def test_timeouts_and_connection_errors_are_normalised() -> None:
    def timeout(request):
        raise httpx.ReadTimeout("slow", request=request)

    def refused(request):
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(ProviderTimeoutError):
        _adapter(timeout).generate_json("s", "u", {"type": "object"})
    with pytest.raises(ProviderError):
        _adapter(refused).generate_json("s", "u", {"type": "object"})
    assert not _adapter(refused).check_availability().available  # probe reports, never raises


def test_error_bodies_do_not_leak_keys() -> None:
    adapter = _adapter(lambda r: httpx.Response(500, text="echo Authorization: Bearer nvapi-SECRETSECRET123"))
    with pytest.raises(ProviderError) as caught:
        adapter.generate_json("s", "u", {"type": "object"})
    assert "SECRETSECRET123" not in str(caught.value)


def test_redaction_patterns() -> None:
    text = redact("Bearer abc.def-123 api_key=sk-abcdefghijk token: xyz nvapi-0123456789abcdef")
    for secret in ("abc.def-123", "sk-abcdefghijk", "xyz", "nvapi-0123456789abcdef"):
        assert secret not in text


def test_storage_errors_are_wrapped(tmp_path: Path) -> None:
    with pytest.raises(StorageError):
        file_sha256(tmp_path / "missing.csv")
    (tmp_path / "bad.json").write_text("{", encoding="utf-8")
    with pytest.raises(StorageError):
        read_json(tmp_path / "bad.json")
    with pytest.raises(StorageError):
        load_signatures(tmp_path / "missing.npy")
    blocker = tmp_path / "blocker"
    blocker.write_text("x", encoding="utf-8")
    with pytest.raises(StorageError):
        save_signatures(blocker / "sig.npy", np.zeros(3, dtype=np.uint64))
    with pytest.raises(StorageError):
        JobPaths.create(blocker, "job_x")


def test_corrupt_status_counts_as_not_completed(tmp_path: Path) -> None:
    paths = JobPaths.create(tmp_path, "job")
    paths.task_status(1).write_text("{ broken", encoding="utf-8")
    assert verified_completion(paths, 1, 10, "csv") is None


@dataclass
class _Item:
    task_id: int
    part: int = 0
    kind: str = "generate"
    attempt: int = 0
    model_key: str | None = None

    @property
    def label(self) -> str:
        return f"task {self.task_id}"


class _CrashingExecutor:
    mode = "thread"

    def run(self, item: _Item) -> dict:
        if item.task_id == 1:
            raise WorkerCrashError("worker process died")
        return {"task_id": item.task_id, "status": "completed", "record_count": 5}


def test_crashed_task_fails_alone_and_the_queue_continues() -> None:
    scheduler = CoreQueueScheduler({"core_1": [_Item(1), _Item(2)]})
    tracker = ExecutionTracker("t", 2, 10)
    results = run_core_queue("core_1", scheduler, _CrashingExecutor(), tracker, CancellationToken())
    assert [(r["task_id"], r["status"]) for r in results] == [(1, "failed"), (2, "completed")]
    assert "WorkerCrashError" in results[0]["error"]
    assert tracker.failed == {1} and tracker.completed == {2}


def test_cli_reports_bad_input_with_exit_code_2(tmp_path: Path, capsys) -> None:
    assert main(["generate", "--config", str(tmp_path / "nope.json")]) == 2
    assert "file not found" in capsys.readouterr().err
    bad = tmp_path / "bad.json"
    bad.write_text("{ oops", encoding="utf-8")
    assert main(["generate", "--config", str(bad)]) == 2
    assert "invalid JSON" in capsys.readouterr().err
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps({"schema": {"id": "unique integer"}, "total_records": -5}), encoding="utf-8")
    assert main(["generate", "--config", str(invalid)]) == 2
    assert "total_records" in capsys.readouterr().err
    assert main(["resume", "does_not_exist_20990101_000000_abcdef"]) == 2
