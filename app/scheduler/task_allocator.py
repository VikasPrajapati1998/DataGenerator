"""Power-of-two task calculation, record partitioning and core allocation."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from app.core.exceptions import SchedulingError


@dataclass(frozen=True)
class TaskSpec:
    """One generation task.

    Attributes:
        task_id: 1-based task identifier.
        start_index: Zero-based global index of the task's first record.
        record_count: Records this task must deliver (always >= 1).
    """

    task_id: int
    start_index: int
    record_count: int

    @property
    def end_index(self) -> int:
        """Zero-based global index of the task's last record (inclusive)."""
        return self.start_index + self.record_count - 1

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable copy."""
        return asdict(self)


def calculate_dynamic_nodes(total_records: int, batch_size: int) -> int:
    """Return the power-of-two number of generation tasks.

    Args:
        total_records: Requested records (positive).
        batch_size: Target records per task (positive).

    Returns:
        ``2 ** ceil(log2(ceil(total_records / batch_size)))``, or 1.

    Raises:
        SchedulingError: If either argument is not positive.
    """
    if total_records <= 0 or batch_size <= 0:
        raise SchedulingError("total_records and batch_size must be positive")
    required_nodes = math.ceil(total_records / batch_size)
    if required_nodes <= 1:
        return 1
    exponent = math.ceil(math.log2(required_nodes))
    return 2 ** exponent


def plan_tasks(total_records: int, batch_size: int) -> list[TaskSpec]:
    """Partition ``total_records`` across the power-of-two task count.

    Records are spread as evenly as possible (sizes differ by at most one).
    If the power-of-two count exceeds the number of records (tiny batch
    sizes), the count is capped at ``total_records`` so no task is empty;
    the cap is the only deviation and is logged by the planner.

    Args:
        total_records: Requested records.
        batch_size: Target records per task.

    Returns:
        Contiguous, non-overlapping tasks whose counts sum to ``total_records``.
    """
    nodes = min(calculate_dynamic_nodes(total_records, batch_size), total_records)
    base, remainder = divmod(total_records, nodes)
    tasks: list[TaskSpec] = []
    start = 0
    for index in range(nodes):
        count = base + (1 if index < remainder else 0)
        tasks.append(TaskSpec(task_id=index + 1, start_index=start, record_count=count))
        start += count
    if start != total_records or any(t.record_count < 1 for t in tasks):
        raise SchedulingError("Record partitioning invariant violated")
    return tasks


def allocate_to_cores(task_ids: list[int], cpu_cores: int) -> dict[str, list[int]]:
    """Distribute tasks across logical core queues in contiguous, balanced blocks.

    Args:
        task_ids: Task ids in execution order.
        cpu_cores: Number of logical core queues (positive).

    Returns:
        ``{"core_1": [...], ...}``; queue sizes differ by at most one and only
        non-empty queues are returned.

    Raises:
        SchedulingError: If ``cpu_cores`` is not positive.
    """
    if cpu_cores <= 0:
        raise SchedulingError("cpu_cores must be positive")
    base, remainder = divmod(len(task_ids), cpu_cores)
    allocation: dict[str, list[int]] = {}
    position = 0
    for core in range(cpu_cores):
        assigned = base + (1 if core < remainder else 0)
        if assigned == 0:
            continue
        allocation[f"core_{core + 1}"] = task_ids[position:position + assigned]
        position += assigned
    return allocation
