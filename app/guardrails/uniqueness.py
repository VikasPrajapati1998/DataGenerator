"""Shared, persistent uniqueness registry backed by SQLite.

Every worker (thread or process) claims the hashes of its unique-field values
and of its full record before a batch is accepted. A ``PRIMARY KEY`` on
``(field, value_hash)`` makes claims atomic across threads and processes, so
two tasks can never both own the same value. Because the registry lives in
the job directory, it survives crashes and is reused when a job resumes.

Claims are tagged with the owning task id; a retried task first releases its
own stale claims, which keeps task execution idempotent.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterable, Sequence
from pathlib import Path

from app.core.exceptions import StorageError
from app.utils.helpers import chunked

RECORD_KEY = "__record__"
_SQL_CHUNK = 400
PART_FACTOR = 1000
"""Claims of sub-batch ``p`` of task ``t`` are owned by ``t * PART_FACTOR + p``."""


def owner_id(task_id: int, part: int = 0) -> int:
    """Return the claim owner of a task (``part == 0``) or of one of its sub-batches.

    Args:
        task_id: Task id.
        part: Sub-batch number (1-based) or 0 for the whole task.

    Returns:
        The owner id stored with each claim.
    """
    return task_id if part == 0 else task_id * PART_FACTOR + part


NAMESPACE_FACTOR = 10 ** 12
"""Owner ids of sub-job ``n`` of a split run are offset by ``n * NAMESPACE_FACTOR``."""


class UniquenessRegistry:
    """Cross-task registry of claimed unique values.

    Args:
        db_path: SQLite database file (created when missing).
        namespace: Owner namespace. Sub-jobs of a split run share one database,
            and each uses its own namespace so that their task ids never alias.
    """

    def __init__(self, db_path: Path, namespace: int = 0) -> None:
        self.db_path = db_path
        self._ns = namespace * NAMESPACE_FACTOR
        self._local = threading.local()
        # Every thread opens its own connection; all are kept here so close() releases them all. A connection
        # left to the garbage collector keeps the .sqlite/-wal/-shm files locked on Windows, so the job
        # folder could not be deleted.
        self._all: list[sqlite3.Connection] = []
        self._all_lock = threading.Lock()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS claims (field TEXT NOT NULL, value_hash INTEGER NOT NULL, "
                "task_id INTEGER NOT NULL, PRIMARY KEY (field, value_hash)) WITHOUT ROWID"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS claims_task ON claims(task_id)")

    def _connect(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            try:
                conn = sqlite3.connect(self.db_path, timeout=60, isolation_level=None, check_same_thread=False)
            except sqlite3.Error as exc:
                raise StorageError(f"Cannot open uniqueness registry {self.db_path}: {exc}") from exc
            conn.execute("PRAGMA busy_timeout=60000")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
            with self._all_lock:
                self._all.append(conn)
        return conn

    def claim(self, task_id: int, field: str, hashes: Sequence[int]) -> set[int]:
        """Atomically claim ``hashes`` for ``task_id``.

        Args:
            task_id: Owning task.
            field: Field name (or :data:`RECORD_KEY`).
            hashes: Value hashes to claim; duplicates within the call are
                reported as conflicts after the first occurrence.

        Returns:
            Hashes that are already owned by a *different* task (conflicts).
        """
        if not hashes:
            return set()
        conn = self._connect()
        conflicts: set[int] = set()
        owner = task_id + self._ns
        try:
            conn.execute("BEGIN IMMEDIATE")
            for part in chunked(list(dict.fromkeys(hashes)), _SQL_CHUNK):
                conn.executemany("INSERT OR IGNORE INTO claims(field, value_hash, task_id) VALUES (?, ?, ?)",
                                 [(field, h, owner) for h in part])
                placeholders = ",".join("?" * len(part))
                rows = conn.execute(
                    f"SELECT value_hash FROM claims WHERE field = ? AND task_id != ? "  # noqa: S608
                    f"AND value_hash IN ({placeholders})", (field, owner, *part)).fetchall()
                conflicts.update(row[0] for row in rows)
            conn.execute("COMMIT")
        except sqlite3.Error as exc:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise StorageError(f"Uniqueness claim failed for task {task_id}, field {field!r}: {exc}") from exc
        return conflicts

    def _write(self, sql: str, params: tuple[object, ...]) -> int:
        """Execute one autocommit write, normalising SQLite errors."""
        try:
            return self._connect().execute(sql, params).rowcount
        except sqlite3.Error as exc:
            raise StorageError(f"Uniqueness registry write failed: {exc}") from exc

    def release(self, task_id: int, field: str, hashes: Iterable[int]) -> None:
        """Release specific claims owned by ``task_id``.

        Args:
            task_id: Owning task.
            field: Field name.
            hashes: Hashes to release.
        """
        for part in chunked(list(hashes), _SQL_CHUNK):
            placeholders = ",".join("?" * len(part))
            self._write(f"DELETE FROM claims WHERE field = ? AND task_id = ? "  # noqa: S608
                        f"AND value_hash IN ({placeholders})", (field, task_id + self._ns, *part))

    def release_task(self, task_id: int) -> int:
        """Release every claim of ``task_id`` (before an idempotent retry).

        Args:
            task_id: Task whose claims are dropped.

        Returns:
            Number of claims removed.
        """
        return self._write("DELETE FROM claims WHERE task_id = ?", (task_id + self._ns,))

    def release_family(self, task_id: int) -> int:
        """Release the claims of a task and of all its sub-batches.

        Args:
            task_id: Task id.

        Returns:
            Number of claims removed.
        """
        ns = self._ns
        low, high = owner_id(task_id, 1) + ns, owner_id(task_id, PART_FACTOR - 1) + ns
        return self._write("DELETE FROM claims WHERE task_id = ? OR task_id BETWEEN ? AND ?",
                           (task_id + ns, low, high))

    def adopt_parts(self, task_id: int) -> int:
        """Re-own the claims of all sub-batches by the task itself (after consolidation).

        Args:
            task_id: Task id.

        Returns:
            Number of claims moved.

        Raises:
            StorageError: If the update fails.
        """
        ns = self._ns
        low, high = owner_id(task_id, 1) + ns, owner_id(task_id, PART_FACTOR - 1) + ns
        return self._write("UPDATE claims SET task_id = ? WHERE task_id BETWEEN ? AND ?", (task_id + ns, low, high))

    def owned_by(self, task_id: int, field: str) -> set[int]:
        """Return the hashes ``task_id`` owns for ``field``."""
        rows = self._connect().execute("SELECT value_hash FROM claims WHERE task_id = ? AND field = ?",
                                       (task_id + self._ns, field)).fetchall()
        return {row[0] for row in rows}

    def count(self, field: str | None = None) -> int:
        """Return the number of claims, optionally for one field."""
        conn = self._connect()
        if field is None:
            return conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0]
        return conn.execute("SELECT COUNT(*) FROM claims WHERE field = ?", (field,)).fetchone()[0]

    def close(self) -> None:
        """Close the connections of every thread (call once the job's workers are done)."""
        with self._all_lock:
            connections, self._all = self._all, []
        for conn in connections:
            try:
                conn.close()
            except sqlite3.Error:  # pragma: no cover - closing is best effort
                pass
        self._local = threading.local()
