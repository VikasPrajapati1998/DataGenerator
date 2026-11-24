"""Splitting large requests into sequential sub-jobs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.core.exceptions import ConfigurationError
from app.guardrails.uniqueness import UniquenessRegistry
from app.split_runner import MANIFEST, plan_parts
from app.storage.formats import iter_records


def test_plan_parts_is_balanced_and_complete() -> None:
    parts = plan_parts(2500, 1000)
    assert [p["record_count"] for p in parts] == [834, 833, 833]
    assert [p["record_offset"] for p in parts] == [0, 834, 1667]
    assert plan_parts(1000, 1000) == [{"part": 1, "record_offset": 0, "record_count": 1000}]
    assert sum(p["record_count"] for p in plan_parts(1_000_000, 100_000)) == 1_000_000
    with pytest.raises(ConfigurationError):
        plan_parts(10, 0)


def test_registry_namespaces_keep_sub_jobs_apart(tmp_path: Path) -> None:
    first = UniquenessRegistry(tmp_path / "shared.sqlite", namespace=1)
    second = UniquenessRegistry(tmp_path / "shared.sqlite", namespace=2)
    assert first.claim(1, "email", [11, 12]) == set()
    assert second.claim(1, "email", [12, 13]) == {12}  # same task id, different sub-job: still a conflict
    second.release_task(1)
    assert second.claim(7, "email", [11]) == {11}  # sub-job 2 must not have released sub-job 1's claims


def _manifest(result) -> dict:
    return json.loads((Path(result.job_dir) / MANIFEST).read_text(encoding="utf-8"))


def test_large_request_runs_as_sub_jobs_and_merges(engine, employee_request) -> None:
    result = engine.run({**employee_request, "total_records": 2500, "split_size": 1000, "schema_name": "employees"})
    assert result.ok, result.error
    manifest = _manifest(result)
    assert [p["status"] for p in manifest["parts"]] == ["completed"] * 3
    rows = list(iter_records(Path(result.final_output), "csv"))
    assert [int(r["employee_id"]) for r in rows] == list(range(1, 2501))  # IDs continue across sub-jobs
    assert len({r["email"] for r in rows}) == 2500  # unique across sub-jobs (shared registry)
    profiles = [json.loads((Path(p["job_dir"]) / "metadata" / "job_config.json").read_text(encoding="utf-8"))
                ["profiles"] for p in manifest["parts"]]
    assert profiles[0] == profiles[1] == profiles[2]  # calibrated once, reused by every sub-job
    assert result.summary["execution"]["sub_jobs"] == 3
    assert result.summary["validation"]["cross_sub_job"]["passed"] is True
    assert Path(result.job_dir).parent.name == "employees"


def test_split_run_resumes_only_unfinished_sub_jobs(engine, employee_request) -> None:
    result = engine.run({**employee_request, "total_records": 2100, "split_size": 700})
    assert result.ok
    manifest = _manifest(result)
    kept = {p["part"]: Path(p["final_output"]).stat().st_mtime_ns for p in manifest["parts"][:2]}
    manifest["parts"][2]["status"] = "failed"
    for path in sorted(Path(manifest["parts"][2]["job_dir"]).rglob("*"), reverse=True):
        path.unlink() if path.is_file() else path.rmdir()
    (Path(result.job_dir) / MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")

    resumed = engine.resume(result.job_id)
    assert resumed.ok, resumed.error
    after = _manifest(resumed)
    assert {p["part"]: Path(p["final_output"]).stat().st_mtime_ns for p in after["parts"][:2]} == kept
    rows = list(iter_records(Path(resumed.final_output), "csv"))
    assert [int(r["employee_id"]) for r in rows] == list(range(1, 2101))


def test_split_run_resumes_an_interrupted_sub_job(engine, employee_request) -> None:
    """Like pressing Ctrl+C during sub-job 3: sub-jobs 1-2 are done, sub-job 3 is half-written."""
    result = engine.run({**employee_request, "total_records": 2100, "split_size": 700})
    assert result.ok
    manifest = _manifest(result)
    kept = {p["part"]: Path(p["final_output"]).stat().st_mtime_ns for p in manifest["parts"][:2]}
    third = manifest["parts"][2]
    Path(third["final_output"]).unlink()
    batches = sorted((Path(third["job_dir"]) / "batches").glob("batch_*.csv"))
    untouched = {b.name: b.stat().st_mtime_ns for b in batches[1:]}
    batches[0].unlink()  # one batch of the interrupted sub-job never got written
    third.update(status="pending", final_output=None)
    manifest["status"] = "running"
    (Path(result.job_dir) / MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")

    resumed = engine.resume(result.job_id)
    assert resumed.ok, resumed.error
    after = _manifest(resumed)
    assert {p["part"]: Path(p["final_output"]).stat().st_mtime_ns for p in after["parts"][:2]} == kept
    assert {b.name: b.stat().st_mtime_ns for b in sorted((Path(third["job_dir"]) / "batches").glob("batch_*.csv"))
            if b.name in untouched} == untouched  # only the missing batch was regenerated
    rows = list(iter_records(Path(resumed.final_output), "csv"))
    assert [int(r["employee_id"]) for r in rows] == list(range(1, 2101))


def test_requests_within_the_limit_are_not_split(engine, employee_request) -> None:
    result = engine.run({**employee_request, "total_records": 500, "split_size": 1000})
    assert result.ok and not (Path(result.job_dir) / MANIFEST).exists()
