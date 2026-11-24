"""Redistribution of failed tasks as parallel sub-batches.

When a task fails, its retry can be split into ``k`` sub-batches ("parts")
with contiguous, deterministic slices of the task's record range. The parts
are queued like any other work item, so several core queues generate them in
parallel, each part with a different pool model.

Invariants that keep the rest of the engine unchanged:

* the record range (and therefore every sequence ID) of the task never changes;
* each part owns its uniqueness claims (``task_id * 1000 + part``), so retrying
  one part never releases a sibling's claims;
* when every part is complete, :func:`consolidate` concatenates the part
  files in order into the task's normal ``batch_NNNN`` file, transfers the
  claims to the task and writes the task's status with a fresh checksum.
  Global guardrails, repairs, recovery and the merger only ever see whole
  batches;
* a split manifest records the plan, so an interrupted job resumes the
  unfinished parts instead of starting the task over.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from app.config.guardrail_config import GuardrailConfig
from app.core.exceptions import RecoveryError, StorageError
from app.guardrails.semantic_deduplication import load_signatures, save_signatures
from app.monitoring.logger import get_logger
from app.storage.batch_writer import (
    STATUS_COMPLETED,
    verified_completion,
    write_batch,
    write_rejected,
    write_task_status,
)
from app.storage.file_manager import JobPaths, read_json, write_json_atomic
from app.storage.formats import iter_records
from app.utils.helpers import utc_now_iso
from app.workers.generation_worker import WorkItem
from app.workers.services import WorkerServices

logger = get_logger(__name__)


def plan_split(task: dict[str, Any], queues: int, config: GuardrailConfig) -> list[dict[str, int]]:
    """Return the sub-batch layout for a failed task, or ``[]`` if it should not be split.

    Args:
        task: Planned task (``task_id``, ``start_index``, ``record_count``).
        queues: Logical core queues available for the retry.
        config: Guardrail settings (split switches and limits).

    Returns:
        ``[{"part", "start_index", "record_count"}, ...]`` covering the task's
        range exactly, in order, with sizes differing by at most one.
    """
    count = int(task["record_count"])
    parts = min(queues, config.max_split_parts, count // config.min_split_records)
    if not config.split_failed_tasks or parts < 2:
        return []
    base, remainder = divmod(count, parts)
    layout, start = [], int(task["start_index"])
    for index in range(parts):
        size = base + (1 if index < remainder else 0)
        layout.append({"part": index + 1, "start_index": start, "record_count": size})
        start += size
    return layout


def write_manifest(paths: JobPaths, task: dict[str, Any], layout: list[dict[str, int]]) -> dict[str, Any]:
    """Persist the split plan of a task (atomically) and return it."""
    manifest = {"task_id": task["task_id"], "start_index": task["start_index"],
                "record_count": task["record_count"], "parts": layout, "created_at": utc_now_iso()}
    write_json_atomic(paths.split_manifest(int(task["task_id"])), manifest)
    return manifest


def load_manifest(paths: JobPaths, task_id: int) -> dict[str, Any] | None:
    """Return the split manifest of ``task_id``, or ``None`` if the task is not split.

    Raises:
        RecoveryError: If the manifest exists but is corrupt or inconsistent.
    """
    try:
        manifest = read_json(paths.split_manifest(task_id))
    except StorageError as exc:
        raise RecoveryError(f"Split manifest of task {task_id} is unreadable: {exc}") from exc
    if manifest is None:
        return None
    parts = manifest.get("parts") or []
    if sum(int(p["record_count"]) for p in parts) != int(manifest["record_count"]):
        raise RecoveryError(f"Split manifest of task {task_id} does not cover the task's records")
    return manifest


def part_items(manifest: dict[str, Any], parts: list[int], attempt: int, pool: list[str],
               base_model: str | None) -> list[dict[str, Any]]:
    """Return work items for the given sub-batches, each on a different pool model.

    Args:
        manifest: Split manifest.
        parts: Part numbers to (re)run.
        attempt: Retry round, used for seeds and model rotation.
        pool: Available pool models.
        base_model: Model the task originally used.

    Returns:
        Work item dictionaries.
    """
    start = pool.index(base_model) if base_model in pool else 0
    items = []
    for spec in manifest["parts"]:
        if spec["part"] not in parts:
            continue
        model = pool[(start + attempt + spec["part"] - 1) % len(pool)] if pool else base_model
        items.append(WorkItem(task_id=manifest["task_id"], start_index=spec["start_index"],
                              record_count=spec["record_count"], attempt=attempt, model_key=model,
                              part=spec["part"]).to_dict())
    return items


def finished_parts(paths: JobPaths, manifest: dict[str, Any], fmt: str) -> list[int]:
    """Return the part numbers whose sub-batch is completed and verified."""
    return [p["part"] for p in manifest["parts"]
            if verified_completion(paths, manifest["task_id"], p["record_count"], fmt, p["part"])]


def consolidate(services: WorkerServices, manifest: dict[str, Any]) -> dict[str, Any]:
    """Merge the finished sub-batches of a split task into its normal batch.

    Args:
        services: Job services (paths, schema, registry).
        manifest: Split manifest of the task.

    Returns:
        The task's new status.

    Raises:
        RecoveryError: If a part is missing, corrupt or the merged count is wrong.
    """
    paths, fmt = services.paths, services.spec.output_format
    task_id = int(manifest["task_id"])
    statuses = []
    records: list[dict[str, Any]] = []
    for spec in manifest["parts"]:
        status = verified_completion(paths, task_id, spec["record_count"], fmt, spec["part"])
        if status is None:
            raise RecoveryError(f"Task {task_id} part {spec['part']} is not complete; cannot consolidate")
        statuses.append(status)
        records.extend(iter_records(paths.batch_file(task_id, fmt, spec["part"]), fmt))
    if len(records) != int(manifest["record_count"]):
        raise RecoveryError(f"Task {task_id}: parts hold {len(records)} records, "
                            f"expected {manifest['record_count']}")

    batch_path, checksum = write_batch(paths, task_id, records, services.schema.columns, fmt)
    signature_files = [paths.signature_file(task_id, p["part"]) for p in manifest["parts"]]
    if all(f.exists() for f in signature_files):
        try:
            save_signatures(paths.signature_file(task_id),
                            np.concatenate([load_signatures(f) for f in signature_files]))
        except StorageError as exc:
            raise RecoveryError(f"Cannot merge semantic signatures of task {task_id}: {exc}") from exc
    rejected: list[dict[str, Any]] = []
    for path in [paths.rejected_file(task_id), *(paths.rejected_file(task_id, p["part"]) for p in manifest["parts"])]:
        report = read_json(path, {}) or {}
        rejected.extend(report.get("rejected", []))
    write_rejected(paths, task_id, rejected, unresolved=0)
    services.registry.adopt_parts(task_id)
    status = {
        "task_id": task_id, "kind": "generate", "status": STATUS_COMPLETED, "record_count": len(records),
        "start_index": manifest["start_index"], "batch_file": str(batch_path), "checksum": checksum,
        "split_parts": len(manifest["parts"]), "model_key": None,
        "part_models": {str(s.get("part")): s.get("model_key") for s in statuses},
        "rejected_count": sum(int(s.get("rejected_count", 0)) for s in statuses),
        "duplicate_rejections": sum(int(s.get("duplicate_rejections", 0)) for s in statuses),
        "semantic_rejections": sum(int(s.get("semantic_rejections", 0)) for s in statuses),
    }
    write_task_status(paths, status)
    remove_split_files(paths, manifest, fmt)
    logger.info("Task %d: consolidated %d sub-batches into %s", task_id, len(manifest["parts"]), batch_path.name)
    return status


def remove_split_files(paths: JobPaths, manifest: dict[str, Any], fmt: str) -> None:
    """Delete part files and the manifest of a consolidated task (idempotent)."""
    task_id = int(manifest["task_id"])
    files: list[Path] = []
    for spec in manifest["parts"]:
        part = spec["part"]
        files += [paths.batch_file(task_id, fmt, part), paths.task_status(task_id, part),
                  paths.rejected_file(task_id, part), paths.signature_file(task_id, part)]
    files.append(paths.split_manifest(task_id))  # last: its presence means "clean-up pending"
    for path in files:
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("Could not remove %s after consolidation: %s", path, exc)
