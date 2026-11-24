"""Job directory layout and atomic file operations.

Atomic writes go to a temporary file in the destination directory, are
flushed and fsynced, and then moved into place with :func:`os.replace`, which
is atomic on both Windows and POSIX for same-volume moves. Readers therefore
never observe partially written files. On Windows a replace can transiently
fail while antivirus or indexers hold the target open, so it is retried.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import IO, Any, Iterator

from app.core.exceptions import ResourceError, StorageError

_REPLACE_ATTEMPTS = 6


@dataclass(frozen=True)
class JobPaths:
    """Well-known locations inside one job directory.

    Attributes:
        root: ``output/Jobs_<YYYY-MM-DD>/<schema_name>/<job_id>``.
    """

    root: Path

    @classmethod
    def at(cls, root: Path) -> JobPaths:
        """Create (if needed) the layout at an explicit job directory (sub-jobs of a split run).

        Raises:
            StorageError: If the directories cannot be created.
        """
        paths = cls(root)
        try:
            for directory in (paths.metadata, paths.tasks, paths.batches, paths.rejected, paths.final,
                              paths.signatures):
                directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise StorageError(f"Cannot create job directory {root}: {exc}") from exc
        return paths

    @classmethod
    def create(cls, output_dir: Path, job_id: str, schema_name: str = "dataset",
               day: date | None = None) -> JobPaths:
        """Create (if needed) and return the directory layout for ``job_id``.

        Raises ``StorageError`` if the directories cannot be created.

        All jobs of one day share ``Jobs_<YYYY-MM-DD>``; inside it each schema
        has its own folder, and each run of that schema its own ``<job_id>``
        folder, so repeated runs never overwrite each other.

        Args:
            output_dir: Output root.
            job_id: Job identifier (``<schema_name>_<timestamp>_<hex>``).
            schema_name: Schema folder name.
            day: Date folder; defaults to today.

        Returns:
            The job layout.
        """
        day_dir = output_dir / f"Jobs_{(day or date.today()).isoformat()}"
        return cls.at(day_dir / schema_name / job_id)

    @property
    def metadata(self) -> Path:
        """Job-level metadata directory."""
        return self.root / "metadata"

    @property
    def tasks(self) -> Path:
        """Per-task status files (recovery manifest)."""
        return self.metadata / "tasks"

    @property
    def signatures(self) -> Path:
        """Per-batch semantic signatures for global LSH."""
        return self.metadata / "signatures"

    @property
    def batches(self) -> Path:
        """Validated batch files."""
        return self.root / "batches"

    @property
    def rejected(self) -> Path:
        """Rejected-record reports."""
        return self.root / "rejected"

    @property
    def final(self) -> Path:
        """Final merged dataset."""
        return self.root / "final"

    @property
    def job_config(self) -> Path:
        """Frozen job configuration (request, schema, plan, profiles)."""
        return self.metadata / "job_config.json"

    @property
    def registry_db(self) -> Path:
        """SQLite uniqueness registry."""
        return self.metadata / "uniqueness.sqlite"

    @staticmethod
    def _suffix(part: int) -> str:
        return f"_p{part:02d}" if part else ""

    def batch_file(self, task_id: int, fmt: str, part: int = 0) -> Path:
        """Path of the batch file for ``task_id`` (or of its sub-batch ``part``)."""
        return self.batches / f"batch_{task_id:04d}{self._suffix(part)}.{fmt}"

    def rejected_file(self, task_id: int, part: int = 0) -> Path:
        """Path of the rejected-record report for ``task_id`` (or sub-batch ``part``)."""
        return self.rejected / f"batch_{task_id:04d}{self._suffix(part)}_rejected.json"

    def task_status(self, task_id: int, part: int = 0) -> Path:
        """Path of the status file for ``task_id`` (or sub-batch ``part``)."""
        return self.tasks / f"task_{task_id:04d}{self._suffix(part)}.json"

    def signature_file(self, task_id: int, part: int = 0) -> Path:
        """Path of the signature file for ``task_id`` (or sub-batch ``part``)."""
        return self.signatures / f"batch_{task_id:04d}{self._suffix(part)}.npy"

    def split_manifest(self, task_id: int) -> Path:
        """Path of the manifest describing how a failed task was split into sub-batches."""
        return self.tasks / f"task_{task_id:04d}_split.json"

    def final_file(self, fmt: str) -> Path:
        """Path of the merged dataset."""
        return self.final / f"generated_dataset.{fmt}"


_JOB_DATE = re.compile(r"(\d{4})(\d{2})(\d{2})_\d{6}_[0-9a-f]{6}$")


def find_job_dir(output_dir: Path, job: str | Path) -> Path | None:
    """Locate a job directory from a job id or a path.

    Looks in ``Jobs_<date>/<schema>/<job_id>`` (the date is read from the id),
    then searches every date folder, then the legacy ``jobs/job_<id>`` layout.

    Args:
        output_dir: Output root.
        job: Job id or an existing job directory.

    Returns:
        The job directory, or ``None`` if it does not exist.
    """
    candidate = Path(job)
    if candidate.is_dir() and ((candidate / "metadata").is_dir() or (candidate / "split_run.json").exists()
                                or (candidate / "multi_table.json").exists()):
        return candidate
    job_id = str(job)
    if match := _JOB_DATE.search(job_id):
        day_dir = output_dir / f"Jobs_{match.group(1)}-{match.group(2)}-{match.group(3)}"
        hits = sorted(day_dir.glob(f"*/{job_id}")) if day_dir.is_dir() else []
        if hits:
            return hits[0]
    hits = sorted(output_dir.glob(f"Jobs_*/*/{job_id}"))
    if hits:
        return hits[0]
    legacy = output_dir / "jobs" / f"job_{job_id}"
    return legacy if legacy.is_dir() else None


def _replace_with_retry(source: Path, target: Path) -> None:
    for attempt in range(1, _REPLACE_ATTEMPTS + 1):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt == _REPLACE_ATTEMPTS:
                raise
            time.sleep(0.05 * 2 ** attempt)


@contextmanager
def atomic_writer(target: Path, mode: str = "w", newline: str | None = None) -> Iterator[IO[Any]]:
    """Yield a file handle whose content atomically replaces ``target`` on success.

    Args:
        target: Final destination.
        mode: ``"w"`` for text or ``"wb"`` for binary.
        newline: Passed to :func:`open` for text mode (``""`` for CSV).

    Yields:
        An open file handle on a temporary file.

    Raises:
        StorageError: If writing or replacing fails; the target is untouched.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    tmp_path = Path(tmp_name)
    try:
        kwargs: dict[str, Any] = {} if "b" in mode else {"encoding": "utf-8", "newline": newline}
        with os.fdopen(fd, mode, **kwargs) as handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        _replace_with_retry(tmp_path, target)
    except OSError as exc:
        tmp_path.unlink(missing_ok=True)
        raise StorageError(f"Atomic write to {target} failed: {exc}") from exc
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def write_json_atomic(target: Path, payload: Any) -> None:
    """Atomically write ``payload`` as pretty-printed JSON."""
    with atomic_writer(target) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)


def read_json(path: Path, default: Any = None) -> Any:
    """Read a JSON file, returning ``default`` when it does not exist.

    Raises:
        StorageError: If the file exists but is corrupt.
    """
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise StorageError(f"Cannot read {path}: {exc}") from exc


def ensure_disk_space(directory: Path, required_bytes: int) -> int:
    """Check that ``directory``'s volume has at least ``required_bytes`` free.

    Args:
        directory: Any path on the target volume.
        required_bytes: Bytes needed.

    Returns:
        Free bytes on the volume.

    Raises:
        ResourceError: If there is not enough free space.
    """
    try:
        directory.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(directory).free
    except OSError as exc:
        raise StorageError(f"Cannot inspect free disk space in {directory}: {exc}") from exc
    if free < required_bytes:
        raise ResourceError(f"Insufficient disk space in {directory}: need {required_bytes:,} bytes, "
                            f"have {free:,}")
    return free


def cleanup_temp_files(directory: Path, predicate: Callable[[Path], bool] | None = None) -> int:
    """Remove orphaned ``.tmp`` files left by an interrupted run.

    Args:
        directory: Directory to scan recursively.
        predicate: Optional extra filter.

    Returns:
        Number of files removed.
    """
    removed = 0
    for path in directory.rglob("*.tmp"):
        if predicate is None or predicate(path):
            path.unlink(missing_ok=True)
            removed += 1
    return removed
