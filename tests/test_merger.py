"""Atomic writes, batch persistence and the deterministic merger."""

from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path

import pytest

from app.core.exceptions import MergeError, StorageError
from app.storage.batch_writer import verified_completion, write_batch, write_task_status
from app.storage.file_manager import JobPaths, atomic_writer, find_job_dir
from app.storage.formats import count_records, iter_records
from app.storage.merger import merge_batches

COLUMNS = ["id", "name", "score", "note"]


def _batch(paths: JobPaths, task_id: int, start: int, count: int, fmt: str) -> dict:
    records = [{"id": start + i, "name": f"Person {start + i}", "score": round(0.5 * (start + i), 1),
                "note": None if i % 3 == 0 else f"line, with comma {i}"} for i in range(count)]
    path, checksum = write_batch(paths, task_id, records, COLUMNS, fmt)
    write_task_status(paths, {"task_id": task_id, "status": "completed", "record_count": count,
                              "checksum": checksum, "batch_file": str(path)})
    return {"task_id": task_id, "record_count": count, "start_index": start - 1}


@pytest.mark.parametrize("fmt", ["csv", "json", "jsonl"])
def test_merge_all_formats(tmp_path: Path, fmt: str) -> None:
    paths = JobPaths.create(tmp_path, "merge")
    tasks = [_batch(paths, 1, 1, 5, fmt), _batch(paths, 2, 6, 4, fmt), _batch(paths, 3, 10, 3, fmt)]
    result = merge_batches(paths, tasks, COLUMNS, fmt, ["id"], ["name", "score", "note"], requested=12)
    assert result.record_count == 12 == count_records(result.output_path, fmt)
    rows = list(iter_records(result.output_path, fmt))
    assert [int(r["id"]) for r in rows] == list(range(1, 13))  # order preserved
    assert rows[0]["note"] is None
    if fmt == "csv":
        text = result.output_path.read_text(encoding="utf-8")
        assert text.count("id,name,score,note") == 1  # single header
    if fmt == "json":
        assert isinstance(json.loads(result.output_path.read_text(encoding="utf-8")), list)


def test_merge_refuses_duplicates(tmp_path: Path) -> None:
    paths = JobPaths.create(tmp_path, "dup")
    tasks = [_batch(paths, 1, 1, 5, "csv"), _batch(paths, 2, 5, 5, "csv")]  # id 5 appears twice
    with pytest.raises(MergeError, match="duplicate"):
        merge_batches(paths, tasks, COLUMNS, "csv", ["id"], ["name", "score", "note"], requested=10)
    assert not paths.final_file("csv").exists()  # nothing partial was published


def test_merge_refuses_wrong_counts_and_corruption(tmp_path: Path) -> None:
    paths = JobPaths.create(tmp_path, "bad")
    tasks = [_batch(paths, 1, 1, 5, "csv")]
    with pytest.raises(MergeError):
        merge_batches(paths, tasks, COLUMNS, "csv", ["id"], ["name"], requested=6)
    with paths.batch_file(1, "csv").open("a", encoding="utf-8") as handle:
        handle.write("99,Tampered,1.0,x\n")
    assert verified_completion(paths, 1, 5, "csv") is None
    with pytest.raises(MergeError, match="checksum"):
        merge_batches(paths, tasks, COLUMNS, "csv", ["id"], ["name"], requested=5)


def test_merge_refuses_schema_mismatch(tmp_path: Path) -> None:
    paths = JobPaths.create(tmp_path, "cols")
    tasks = [_batch(paths, 1, 1, 3, "csv")]
    with pytest.raises(MergeError, match="columns"):
        merge_batches(paths, tasks, ["id", "name"], "csv", ["id"], ["name"], requested=3)


def test_atomic_writer_leaves_target_untouched_on_failure(tmp_path: Path) -> None:
    target = tmp_path / "data.csv"
    target.write_text("original", encoding="utf-8")
    with pytest.raises(RuntimeError):
        with atomic_writer(target) as handle:
            handle.write("partial")
            raise RuntimeError("crash mid-write")
    assert target.read_text(encoding="utf-8") == "original"
    assert list(tmp_path.glob("*.tmp")) == []


def test_csv_round_trip_handles_quotes_and_newlines(tmp_path: Path) -> None:
    paths = JobPaths.create(tmp_path, "rt")
    records = [{"id": 1, "name": 'He said "hi"', "score": 1.5, "note": "multi\nline"}]
    path, _ = write_batch(paths, 1, records, COLUMNS, "csv")
    with path.open(encoding="utf-8", newline="") as handle:
        assert list(csv.DictReader(handle))[0]["note"] == "multi\nline"
    assert count_records(path, "csv") == 1


def test_job_layout_groups_by_date_and_schema(tmp_path: Path) -> None:
    first = JobPaths.create(tmp_path, "employees_20261120_185312_84be8c", "employees", date(2026, 11, 20))
    second = JobPaths.create(tmp_path, "employees_20261120_191000_aa11bb", "employees", date(2026, 11, 20))
    assert first.root == tmp_path / "Jobs_2026-11-20" / "employees" / "employees_20261120_185312_84be8c"
    assert first.root.parent == second.root.parent and first.root != second.root
    assert find_job_dir(tmp_path, "employees_20261120_185312_84be8c") == first.root
    assert find_job_dir(tmp_path, str(second.root)) == second.root
    assert find_job_dir(tmp_path, "unknown_20261120_000000_000000") is None
    legacy = tmp_path / "jobs" / "job_20261120_185148_f0630b"
    (legacy / "metadata").mkdir(parents=True)
    assert find_job_dir(tmp_path, "20261120_185148_f0630b") == legacy


def test_unknown_format_raises(tmp_path: Path) -> None:
    with pytest.raises(StorageError):
        list(iter_records(tmp_path / "x.xml", "xml"))
