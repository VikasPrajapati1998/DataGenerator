"""A single logical core queue: FIFO of work items with a one-active-task invariant.

A logical core queue is a scheduling construct, not a hardware binding: it is
executed by one thread (or one worker-process slot) at a time. Physical
pinning is optional and handled separately by :mod:`app.scheduler.affinity`.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

from app.core.exceptions import SchedulingError


@dataclass
class CoreQueue:
    """FIFO queue owned by one logical core.

    Attributes:
        core_id: Queue name such as ``core_1``.
        pending: Work items not started yet (own work at the left).
        active: The item currently executing, if any.
        completed: Task ids finished by this queue.
        failed: Task ids that failed on this queue.
        stolen_in: Task ids this queue took from other queues.
        stolen_out: Task ids other queues took from this one.
        max_active: Highest number of simultaneously active items observed.
    """

    core_id: str
    pending: deque[Any] = field(default_factory=deque)
    active: Any = None
    completed: list[int] = field(default_factory=list)
    failed: list[int] = field(default_factory=list)
    stolen_in: list[int] = field(default_factory=list)
    stolen_out: list[int] = field(default_factory=list)
    max_active: int = 0
    assigned: list[int] = field(default_factory=list)

    def start(self, item: Any) -> None:
        """Mark ``item`` as the active item.

        Raises:
            SchedulingError: If another item is already active on this queue.
        """
        if self.active is not None:
            raise SchedulingError(f"{self.core_id} attempted to run two tasks at once")
        self.active = item
        self.max_active = max(self.max_active, 1)

    def finish(self, task_id: int, success: bool) -> None:
        """Clear the active item and record its outcome.

        Raises:
            SchedulingError: If no item is active.
        """
        if self.active is None:
            raise SchedulingError(f"{self.core_id} finished a task that was not running")
        self.active = None
        (self.completed if success else self.failed).append(task_id)

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the queue."""
        return {"core_id": self.core_id, "assigned": list(self.assigned),
                "pending": [getattr(i, "task_id", i) for i in self.pending],
                "active": getattr(self.active, "task_id", self.active), "completed": list(self.completed),
                "failed": list(self.failed), "stolen_in": list(self.stolen_in), "stolen_out": list(self.stolen_out),
                "max_active": self.max_active}
