"""End-to-end orchestration tests through the LangGraph workflow (mock LLM)."""

from __future__ import annotations

import csv
import json
import sqlite3
import threading
import time
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from app.models.mock_adapter import MockBehavior, MockLLMAdapter
from app.storage.batch_writer import load_task_status, write_batch, write_task_status
from app.storage.file_manager import JobPaths
from app.storage.formats import iter_records

BIO_SCHEMA = {
    "customer_id": "unique integer",
    "name": "realistic full name",
    "email": "valid email",
    "segment": ["retail", "business", "enterprise"],
    "bio": "two-sentence profile of the customer's buying habits",
}


def _summary(result) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(result.job_dir) / "metadata"
    execution = json.loads((root / "execution_summary.json").read_text(encoding="utf-8"))
    validation = json.loads((root / "validation_summary.json").read_text(encoding="utf-8"))
    return execution, validation


def _final_rows(result, fmt: str) -> list[dict[str, Any]]:
    return list(iter_records(Path(result.final_output), fmt))


def test_e2e_1000_records_csv(engine, employee_request) -> None:
    result = engine.run(employee_request)
    assert result.ok, result.error
    rows = _final_rows(result, "csv")
    assert len(rows) == 1000
    assert sorted(int(r["employee_id"]) for r in rows) == list(range(1, 1001))
    assert len({r["email"] for r in rows}) == 1000
    assert {r["department"] for r in rows} == {"IT", "HR", "Finance", "Sales"}
    execution, validation = _summary(result)
    assert execution["dynamic_tasks"] == 4 and execution["completed_tasks"] == 4
    assert all(q["max_active"] <= 1 for q in execution["queue_reports"])
    assert validation["final_validation"] == {
        "final_record_count": 1000, "requested_record_count": 1000, "record_count_matches": True,
        "schema_validation_passed": True, "duplicate_count": 0, "passed": True}
    assert validation["quality"]["score"] >= 85
    batches = sorted((Path(result.job_dir) / "batches").glob("batch_*.csv"))
    assert len(batches) == 4
    with Path(result.final_output).open(encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle))
    assert header == list(employee_request["schema"])  # column order preserved


def test_output_layout_uses_date_and_schema_name(engine, settings, employee_request) -> None:
    first = engine.run({**employee_request, "schema_name": "employees"})
    second = engine.run({**employee_request, "schema_name": "employees"})
    assert first.ok and second.ok
    day_dir = settings.output_dir.resolve() / f"Jobs_{date.today().isoformat()}" / "employees"
    for result in (first, second):
        assert result.job_id.startswith("employees_")
        assert Path(result.job_dir) == day_dir / result.job_id
        assert Path(result.final_output).parent == day_dir / result.job_id / "final"
    assert first.job_dir != second.job_dir  # repeated runs never overwrite each other
    unnamed = engine.run({**employee_request, "total_records": 10, "batch_size": 10})
    assert Path(unnamed.job_dir).parent.name == "dataset"


def test_schema_without_llm_needs_no_model_and_sizes_batches_itself(engine) -> None:
    MockLLMAdapter.behaviors["mock-llm"] = MockBehavior(available=False)  # would fail if it were checked
    request = {"schema": {"fields": {
        "id": {"type": "integer", "unique": True},
        "tier": {"type": "string", "allowed_values": ["a", "b"]},
        "score": {"type": "integer", "generator": "profile", "depends_on": "tier",
                  "profile": {"a": {"min": 1, "max": 10}, "b": {"min": 11, "max": 20}}}}},
        "total_records": 3000, "execution_mode": "thread"}
    result = engine.run(request)
    assert result.ok, result.error
    execution, _ = _summary(result)
    assert execution["model_pool"] == [] and execution["model_status"] == {}
    config = json.loads((Path(result.job_dir) / "metadata" / "job_config.json").read_text(encoding="utf-8"))
    assert config["batch_size"] >= 250  # chosen at run time, not hardcoded
    assert MockLLMAdapter.calls == {}


def test_e2e_10000_records_jsonl_with_work_stealing(engine, employee_request) -> None:
    request = {**employee_request, "total_records": 10_000, "batch_size": 1000, "output_format": "jsonl",
               "work_stealing": True}
    result = engine.run(request)
    assert result.ok, result.error
    rows = _final_rows(result, "jsonl")
    assert len(rows) == 10_000 and len({r["employee_id"] for r in rows}) == 10_000
    execution, _ = _summary(result)
    assert execution["dynamic_tasks"] == 16
    assert execution["valid_records_generated"] == 10_000


def test_llm_fields_semantic_dedup_and_json_output(engine) -> None:
    request = {"schema": BIO_SCHEMA, "total_records": 300, "output_format": "json", "model_provider": "mock",
               "model_name": "mock-llm", "batch_size": 100, "execution_mode": "thread", "seed": 3}
    result = engine.run(request)
    assert result.ok, result.error
    rows = _final_rows(result, "json")
    assert len(rows) == 300
    assert len({r["bio"] for r in rows}) == 300
    execution, validation = _summary(result)
    assert execution["llm_usage"]["calls"] >= 300 // 8
    assert (Path(result.job_dir) / "metadata" / "signatures" / "batch_0001.npy").exists()
    assert any(c["check"] == "distinct_text_ratio" and c["passed"]
               for c in validation["global_guardrails"]["distribution_checks"])


def test_model_fallback_inside_a_job(engine) -> None:
    MockLLMAdapter.behaviors["mock-flaky"] = MockBehavior(fail_rate=1.0)
    request = {"schema": BIO_SCHEMA, "total_records": 40, "model_provider": "mock", "model_name": "mock-flaky",
               "fallback_models": ["mock/mock-backup"], "batch_size": 20, "execution_mode": "thread"}
    result = engine.run(request)
    assert result.ok, result.error
    execution, _ = _summary(result)
    assert execution["llm_usage"]["by_model"].get("mock/mock-backup", 0) > 0


def test_model_pool_spreads_one_model_per_task(engine) -> None:
    pool = ["mock/mock-llm", "mock/mock-backup", "mock/mock-flaky"]
    request = {"schema": BIO_SCHEMA, "total_records": 120, "batch_size": 20, "model_pool": pool,
               "execution_mode": "thread", "seed": 5}
    result = engine.run(request)
    assert result.ok, result.error
    execution, _ = _summary(result)
    assert execution["model_pool"] == pool
    assert execution["model_assignment"] == {"mock/mock-llm": 3, "mock/mock-backup": 3, "mock/mock-flaky": 2}
    assert set(execution["llm_usage"]["by_model"]) == set(pool)
    paths = JobPaths(Path(result.job_dir))
    used = [load_task_status(paths, t)["model_key"] for t in range(1, 9)]
    assert used == [pool[(t - 1) % 3] for t in range(1, 9)]


def test_unavailable_pool_model_is_dropped_and_others_cover(engine) -> None:
    MockLLMAdapter.behaviors["mock-flaky"] = MockBehavior(available=False)
    request = {"schema": BIO_SCHEMA, "total_records": 80, "batch_size": 20,
               "model_pool": ["mock/mock-llm", "mock/mock-flaky", "mock/mock-backup"],
               "model_assignment": "weighted", "model_weights": [3, 5, 1], "execution_mode": "thread"}
    result = engine.run(request)
    assert result.ok, result.error
    execution, _ = _summary(result)
    assert execution["model_pool"] == ["mock/mock-llm", "mock/mock-backup"]
    assert execution["model_assignment"] == {"mock/mock-llm": 3, "mock/mock-backup": 1}


def test_cloud_model_without_credits_is_excluded_before_the_run(settings, registry) -> None:
    from app.config.model_registry import ModelConfig  # pylint: disable=import-outside-toplevel
    from app.engine import SyntheticDataEngine  # pylint: disable=import-outside-toplevel

    registry.register(ModelConfig(provider="mock", model_name="mock-cloud", model_id="mock-cloud",
                                  execution="cloud", max_concurrency=4))
    MockLLMAdapter.behaviors["mock-cloud"] = MockBehavior(quota_exhausted=True)
    engine = SyntheticDataEngine(settings=settings, registry=registry)
    request = {"schema": BIO_SCHEMA, "total_records": 40, "batch_size": 10, "execution_mode": "thread",
               "model_pool": ["mock/mock-cloud", "mock/mock-llm"]}
    result = engine.run(request)
    assert result.ok, result.error
    execution, _ = _summary(result)
    assert execution["model_pool"] == ["mock/mock-llm"]
    assert execution["model_status"]["mock/mock-cloud"]["available"] is False
    assert "quota" in execution["model_status"]["mock/mock-cloud"]["detail"]
    assert MockLLMAdapter.calls["mock-cloud"] == 1  # only the probe; no task ever called it


def test_failed_batches_recover_without_rerunning_successful_ones(engine) -> None:
    request = {"schema": BIO_SCHEMA, "total_records": 80, "model_provider": "mock", "model_name": "mock-flaky",
               "batch_size": 20, "execution_mode": "thread", "seed": 11,
               "quality_requirements": {"max_task_retries": 0}}
    MockLLMAdapter.behaviors["mock-flaky"] = MockBehavior(fail_rate=1.0)
    failed = engine.run(request)
    assert failed.status == "failed"
    paths = JobPaths(Path(failed.job_dir))
    assert load_task_status(paths, 1)["status"] == "failed"
    MockLLMAdapter.behaviors.clear()
    recovered = engine.resume(failed.job_id)
    assert recovered.ok, recovered.error
    assert len(_final_rows(recovered, "csv")) == 80

    # A second resume must not regenerate any completed batch.
    mtimes = {p.name: p.stat().st_mtime_ns for p in paths.batches.glob("*.csv")}
    again = engine.resume(failed.job_id)
    assert again.ok
    assert {p.name: p.stat().st_mtime_ns for p in paths.batches.glob("*.csv")} == mtimes
    execution, _ = _summary(again)
    assert execution["skipped_tasks_already_complete"] == 4


def test_failed_tasks_are_split_across_queues_and_consolidated(engine) -> None:
    # The first 4 LLM calls fail: both tasks exhaust their 2 attempts in round 1 and fail.
    MockLLMAdapter.behaviors["mock-flaky"] = MockBehavior(fail_until_call=4)
    request = {"schema": BIO_SCHEMA, "total_records": 200, "batch_size": 100, "model_pool": ["mock/mock-flaky"],
               "execution_mode": "thread", "max_workers": 2, "seed": 9,
               "quality_requirements": {"min_split_records": 25}}
    result = engine.run(request)
    assert result.ok, result.error
    execution, _ = _summary(result)
    assert execution["split_tasks"] == [1, 2]
    paths = JobPaths(Path(result.job_dir))
    assert sorted(p.name for p in paths.batches.iterdir()) == ["batch_0001.csv", "batch_0002.csv"]
    assert not list(paths.tasks.glob("*_p*.json")) and not list(paths.tasks.glob("*_split.json"))
    status = load_task_status(paths, 1)
    assert status["split_parts"] == 2 and status["record_count"] == 100
    rows = _final_rows(result, "csv")
    assert [int(r["customer_id"]) for r in rows] == list(range(1, 201))  # ID ranges and order preserved
    with sqlite3.connect(paths.registry_db) as conn:
        owners = {row[0] for row in conn.execute("SELECT DISTINCT task_id FROM claims")}
    assert owners == {1, 2}  # sub-batch claims were handed back to their tasks
    rounds = {q["round"] for q in execution["queue_reports"]}
    assert len(rounds) >= 2


def test_corrupt_job_config_fails_cleanly_on_resume(engine, employee_request) -> None:
    result = engine.run({**employee_request, "total_records": 20, "batch_size": 10})
    assert result.ok
    (Path(result.job_dir) / "metadata" / "job_config.json").write_text("{ not json", encoding="utf-8")
    resumed = engine.resume(result.job_id)
    assert resumed.status == "failed" and "StorageError" in (resumed.error or "")


def test_missing_batch_is_regenerated_on_resume(engine, employee_request) -> None:
    result = engine.run(employee_request)
    assert result.ok
    paths = JobPaths(Path(result.job_dir))
    untouched = {p.name: p.stat().st_mtime_ns for p in paths.batches.glob("*.csv") if p.name != "batch_0002.csv"}
    paths.batch_file(2, "csv").unlink()
    resumed = engine.resume(result.job_id)
    assert resumed.ok, resumed.error
    assert paths.batch_file(2, "csv").exists()
    assert {p.name: p.stat().st_mtime_ns for p in paths.batches.glob("*.csv")
            if p.name != "batch_0002.csv"} == untouched
    assert len(_final_rows(resumed, "csv")) == 1000


def test_global_guardrails_repair_cross_batch_duplicates(engine, employee_request) -> None:
    result = engine.run(employee_request)
    assert result.ok
    paths = JobPaths(Path(result.job_dir))
    columns = list(employee_request["schema"])
    first = list(iter_records(paths.batch_file(1, "csv"), "csv"))
    second = list(iter_records(paths.batch_file(2, "csv"), "csv"))
    second[0]["email"] = first[0]["email"]  # inject a cross-batch duplicate behind the node guardrails' back
    _, checksum = write_batch(paths, 2, second, columns, "csv")
    status = load_task_status(paths, 2)
    write_task_status(paths, {**status, "checksum": checksum})

    resumed = engine.resume(result.job_id)
    assert resumed.ok, resumed.error
    rows = _final_rows(resumed, "csv")
    assert len(rows) == 1000 and len({r["email"] for r in rows}) == 1000
    execution, _ = _summary(resumed)
    assert execution["repair_rounds"] >= 1


def test_graceful_cancellation_then_resume(engine) -> None:
    MockLLMAdapter.behaviors["mock-llm"] = MockBehavior(latency=0.05)
    request = {"schema": BIO_SCHEMA, "total_records": 640, "model_provider": "mock", "model_name": "mock-llm",
               "batch_size": 20, "execution_mode": "thread", "max_workers": 2}
    timer = threading.Timer(1.0, engine.cancel)
    timer.start()
    cancelled = engine.run(request)
    timer.cancel()
    assert cancelled.status == "cancelled"
    MockLLMAdapter.behaviors.clear()
    resumed = engine.resume(cancelled.job_id)
    assert resumed.ok, resumed.error
    assert len(_final_rows(resumed, "csv")) == 640


def test_invalid_schema_fails_cleanly(engine) -> None:
    result = engine.run({"schema": {"a": {"type": "integer", "min": 5, "max": 1}}, "total_records": 10,
                         "model_provider": "mock", "model_name": "mock-llm"})
    assert result.status == "failed" and "SchemaError" in (result.error or "")


def test_unavailable_model_is_reported(engine, employee_request) -> None:
    MockLLMAdapter.behaviors["mock-flaky"] = MockBehavior(available=False)
    result = engine.run({**employee_request, "model_name": "mock-flaky"})
    assert result.status == "failed" and "ModelUnavailableError" in (result.error or "")


def test_process_mode_generates_with_real_parallelism(engine, employee_request) -> None:
    request = {**employee_request, "total_records": 20_000, "batch_size": 2500, "execution_mode": "process"}
    started = time.perf_counter()
    result = engine.run(request)
    assert result.ok, result.error
    execution, _ = _summary(result)
    assert execution["execution_mode"] == "process" and execution["dynamic_tasks"] == 8
    assert len(_final_rows(result, "csv")) == 20_000
    assert time.perf_counter() - started < 300


@pytest.mark.slow
def test_e2e_100000_records(engine, employee_request) -> None:
    request = {**employee_request, "total_records": 100_000, "batch_size": 1000, "execution_mode": "auto"}
    result = engine.run(request)
    assert result.ok, result.error
    execution, _ = _summary(result)
    assert execution["dynamic_tasks"] == 128 and execution["valid_records_generated"] == 100_000
