"""Atomic persistence of batch files, rejected-record reports and task status.

The task status file is written *after* the batch file and records its
SHA-256 checksum and record count. A task counts as completed only when the
status says so and the batch file still matches the checksum, which makes
recovery safe against crashes between the two writes.

Every function accepts ``part``: ``0`` addresses the task's own batch, ``p > 0``
addresses sub-batch ``p`` created when a failed task was split.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.core.exceptions import StorageError
from app.monitoring.logger import get_logger
from app.storage.file_manager import JobPaths, atomic_writer, read_json, write_json_atomic
from app.storage.formats import write_records
from app.utils.hashing import file_sha256
from app.utils.helpers import utc_now_iso

logger = get_logger(__name__)

STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"


def write_batch(paths: JobPaths, task_id: int, records: list[dict[str, Any]], columns: list[str],
                fmt: str, part: int = 0) -> tuple[Path, str]:
    """Atomically write a validated batch.

    Args:
        paths: Job layout.
        task_id: Task id.
        records: Validated records in final order.
        columns: Column order.
        fmt: Output format.
        part: Sub-batch number (0 for the whole task).

    Returns:
        The batch path and its SHA-256 checksum.

    Raises:
        StorageError: If the file cannot be written or hashed.
    """
    target = paths.batch_file(task_id, fmt, part)
    with atomic_writer(target, newline="" if fmt == "csv" else None) as handle:
        write_records(handle, records, columns, fmt)
    return target, file_sha256(target)


def write_rejected(paths: JobPaths, task_id: int, rejected: list[dict[str, Any]], unresolved: int,
                   part: int = 0) -> Path:
    """Atomically write the rejected-record report of a task or sub-batch.

    Args:
        paths: Job layout.
        task_id: Task id.
        rejected: Rejection entries (record id, round, reason, record).
        unresolved: Records still invalid after all regeneration rounds.
        part: Sub-batch number (0 for the whole task).

    Returns:
        Path of the report.
    """
    target = paths.rejected_file(task_id, part)
    reasons: dict[str, int] = {}
    for item in rejected:
        key = str(item.get("reason", "")).split(":")[0]
        reasons[key] = reasons.get(key, 0) + 1
    write_json_atomic(target, {"task_id": task_id, "part": part, "rejected_count": len(rejected),
                               "unresolved": unresolved, "reason_counts": reasons, "written_at": utc_now_iso(),
                               "rejected": rejected})
    return target


def write_task_status(paths: JobPaths, status: dict[str, Any]) -> None:
    """Atomically write a task (or sub-batch) status file."""
    write_json_atomic(paths.task_status(int(status["task_id"]), int(status.get("part", 0))),
                      {**status, "updated_at": utc_now_iso()})


def load_task_status(paths: JobPaths, task_id: int, part: int = 0) -> dict[str, Any] | None:
    """Return the stored status of ``task_id`` (or sub-batch ``part``), if any."""
    return read_json(paths.task_status(task_id, part))


def verified_completion(paths: JobPaths, task_id: int, expected_count: int, fmt: str,
                        part: int = 0) -> dict[str, Any] | None:
    """Return the status of a task only if its completed batch is intact.

    Args:
        paths: Job layout.
        task_id: Task id.
        expected_count: Records the task (or sub-batch) must contain.
        fmt: Output format.
        part: Sub-batch number (0 for the whole task).

    Returns:
        The status dict when completed and verified, else ``None``. A corrupt
        status file counts as "not completed" so the batch is regenerated.
    """
    try:
        status = load_task_status(paths, task_id, part)
        if not status or status.get("status") != STATUS_COMPLETED:
            return None
        batch = paths.batch_file(task_id, fmt, part)
        if int(status.get("record_count", -1)) != expected_count or not batch.exists():
            return None
        if file_sha256(batch) != status.get("checksum"):
            logger.warning("Batch %s fails its checksum; it will be regenerated", batch.name)
            return None
    except (StorageError, TypeError, ValueError) as exc:
        logger.warning("Status of task %d part %d is unreadable (%s); treating it as not completed",
                       task_id, part, exc)
        return None
    return status
