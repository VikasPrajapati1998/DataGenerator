"""Deterministic Python merger that consolidates batch files.

No LLM and no LangGraph worker is involved. The merger streams every batch in
task order, verifies integrity (checksum), schema consistency (columns),
per-batch record counts and global uniqueness, then writes the final dataset
atomically. It refuses to merge rather than silently dropping records.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.core.exceptions import MergeError
from app.guardrails.uniqueness import RECORD_KEY
from app.storage.batch_writer import verified_completion
from app.storage.file_manager import JobPaths, atomic_writer
from app.storage.formats import count_records, iter_records, read_header
from app.utils.hashing import file_sha256, record_hash, value_hash


@dataclass
class MergeResult:
    """Outcome of a successful merge."""

    output_path: Path
    record_count: int
    batch_count: int
    duplicate_count: int
    file_size: int
    checksum: str
    per_batch_counts: dict[int, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable copy."""
        return {"output_path": str(self.output_path), "record_count": self.record_count,
                "batch_count": self.batch_count, "duplicate_count": self.duplicate_count,
                "file_size": self.file_size, "checksum": self.checksum}


def merge_batches(paths: JobPaths, tasks: list[dict[str, Any]], columns: list[str], fmt: str,
                  unique_fields: list[str], record_fields: list[str], requested: int) -> MergeResult:
    """Merge all verified batch files into the final dataset.

    Args:
        paths: Job layout.
        tasks: Planned tasks (``task_id``, ``record_count``) in merge order.
        columns: Expected column order.
        fmt: Output format.
        unique_fields: Fields that must be globally unique.
        record_fields: Fields used for whole-record duplicate detection.
        requested: Requested record count.

    Returns:
        The merge result.

    Raises:
        MergeError: On any integrity, schema, count or uniqueness violation.
    """
    batches: list[tuple[int, Path, int]] = []
    for task in tasks:
        task_id, expected = int(task["task_id"]), int(task["record_count"])
        status = verified_completion(paths, task_id, expected, fmt)
        if status is None:
            raise MergeError(f"Batch {task_id} is missing, incomplete or fails its checksum")
        batch = paths.batch_file(task_id, fmt)
        header = read_header(batch, fmt)
        if header != columns:
            raise MergeError(f"Batch {task_id} columns {header} differ from schema {columns}")
        batches.append((task_id, batch, expected))

    seen: dict[str, set[int]] = {name: set() for name in [*unique_fields, RECORD_KEY]}
    duplicates = 0
    per_batch: dict[int, int] = {}
    target = paths.final_file(fmt)
    total = 0
    with atomic_writer(target, newline="" if fmt == "csv" else None) as out:
        writer = csv.writer(out) if fmt == "csv" else None
        if writer is not None:
            writer.writerow(columns)
        elif fmt == "json":
            out.write("[")
        for task_id, batch, expected in batches:
            rows = 0
            for record in iter_records(batch, fmt):
                for name in unique_fields:
                    digest = value_hash(record.get(name))
                    if digest in seen[name]:
                        duplicates += 1
                    seen[name].add(digest)
                if record_fields:  # empty: whole-record duplicates are legitimate for this schema
                    digest = record_hash(record, record_fields)
                    if digest in seen[RECORD_KEY]:
                        duplicates += 1
                    seen[RECORD_KEY].add(digest)
                if writer is not None:
                    writer.writerow(["" if record.get(c) is None else record.get(c) for c in columns])
                elif fmt == "jsonl":
                    out.write(json.dumps({c: record.get(c) for c in columns}, ensure_ascii=False) + "\n")
                else:
                    out.write(",\n" if total else "\n")
                    out.write(json.dumps({c: record.get(c) for c in columns}, ensure_ascii=False))
                rows += 1
                total += 1
            if rows != expected:
                raise MergeError(f"Batch {task_id} contains {rows} records, expected {expected}")
            per_batch[task_id] = rows
        if fmt == "json":
            out.write("\n]\n" if total else "]\n")
        if duplicates:
            raise MergeError(f"Merge found {duplicates} duplicate values across batches")
        if total != requested:
            raise MergeError(f"Merged {total} records but {requested} were requested")

    final_count = count_records(target, fmt)
    if final_count != requested:
        raise MergeError(f"Final file contains {final_count} records, expected {requested}")
    return MergeResult(output_path=target, record_count=final_count, batch_count=len(batches),
                       duplicate_count=duplicates, file_size=target.stat().st_size,
                       checksum=file_sha256(target), per_batch_counts=per_batch)
