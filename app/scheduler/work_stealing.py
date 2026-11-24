"""Core-queue scheduler with optional work stealing.

LangGraph's ``Send`` API fans work out to parallel node invocations but does
not guarantee per-core sequencing. This scheduler provides that guarantee:
one LangGraph worker invocation drives exactly one logical core queue and
asks the scheduler for its next item, so each queue runs one task at a time
while different queues run concurrently.

With work stealing enabled, a queue whose own work is exhausted takes the
*last* pending item of the queue with the most pending work (stealing from
the tail keeps the victim's next item - and its locality - intact).
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any, Protocol

from app.core.cancellation import CancellationToken
from app.core.exceptions import SchedulingError
from app.scheduler.core_queue import CoreQueue


class WorkItem(Protocol):
    """Anything schedulable must expose a ``task_id``."""

    task_id: int


class CoreQueueScheduler:
    """Thread-safe owner of all logical core queues for one dispatch round.

    Args:
        allocation: ``core_id -> work items`` in execution order.
        work_stealing: Allow idle queues to take pending items from busy ones.
    """

    def __init__(self, allocation: dict[str, list[Any]], work_stealing: bool = False) -> None:
        self.work_stealing = work_stealing
        self._lock = threading.Lock()
        self.queues: dict[str, CoreQueue] = {}
        for core_id, items in allocation.items():
            queue = CoreQueue(core_id=core_id, pending=deque(items))
            queue.assigned = [item.task_id for item in items]
            self.queues[core_id] = queue
        self._active_queues = 0
        self.max_concurrent_queues = 0
        self.events: list[tuple[float, str, str, int]] = []

    def acquire(self, core_id: str) -> Any | None:
        """Return the next item ``core_id`` should run, or ``None`` when done.

        Args:
            core_id: Requesting queue.

        Returns:
            A work item, or ``None`` if no work remains for this queue.

        Raises:
            SchedulingError: If the queue already has an active item.
        """
        with self._lock:
            queue = self.queues[core_id]
            if queue.active is not None:
                raise SchedulingError(f"{core_id} requested work while a task is still active")
            item = queue.pending.popleft() if queue.pending else None
            if item is None and self.work_stealing:
                victim = max((q for q in self.queues.values() if q.core_id != core_id and q.pending),
                             key=lambda q: len(q.pending), default=None)
                if victim is not None:
                    item = victim.pending.pop()
                    victim.stolen_out.append(item.task_id)
                    queue.stolen_in.append(item.task_id)
                    self.events.append((time.perf_counter(), core_id, f"steal_from_{victim.core_id}", item.task_id))
            if item is None:
                return None
            queue.start(item)
            self._active_queues += 1
            self.max_concurrent_queues = max(self.max_concurrent_queues, self._active_queues)
            self.events.append((time.perf_counter(), core_id, "start", item.task_id))
            return item

    def release(self, core_id: str, item: Any, success: bool) -> None:
        """Mark the active item of ``core_id`` as finished."""
        with self._lock:
            self.queues[core_id].finish(item.task_id, success)
            self._active_queues -= 1
            self.events.append((time.perf_counter(), core_id, "finish" if success else "fail", item.task_id))

    def drain(self, core_id: str) -> list[Any]:
        """Remove and return the pending items of ``core_id`` (used on cancellation)."""
        with self._lock:
            queue = self.queues[core_id]
            items = list(queue.pending)
            queue.pending.clear()
            return items

    def run_queue(self, core_id: str, execute: Callable[[Any], tuple[Any, bool]],
                  cancel: CancellationToken | None = None) -> tuple[list[Any], list[Any]]:
        """Execute ``core_id``'s queue sequentially until it (and stealable work) is exhausted.

        Args:
            core_id: Queue to drive.
            execute: Runs one item and returns ``(result, success)``.
            cancel: Optional cancellation token checked between tasks.

        Returns:
            ``(results, skipped_items)``; skipped items were never started
            because of cancellation.
        """
        results: list[Any] = []
        while True:
            if cancel is not None and cancel.is_cancelled():
                return results, self.drain(core_id)
            item = self.acquire(core_id)
            if item is None:
                return results, []
            success = False
            try:
                result, success = execute(item)
                results.append(result)
            finally:
                self.release(core_id, item, success)

    def pending_count(self) -> int:
        """Number of items not yet started across all queues."""
        with self._lock:
            return sum(len(q.pending) for q in self.queues.values())

    def active_count(self) -> int:
        """Number of queues currently executing an item."""
        with self._lock:
            return self._active_queues

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of every queue."""
        with self._lock:
            return {"work_stealing": self.work_stealing, "max_concurrent_queues": self.max_concurrent_queues,
                    "queues": {cid: q.snapshot() for cid, q in self.queues.items()}}
