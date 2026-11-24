"""Core queue manager and dynamic ``Send`` dispatcher.

``prepare_dispatch`` distributes pending work items over the logical core
queues (balanced contiguous blocks) and creates the scheduler for the round.
``dispatch_core_queues`` then returns one ``Send`` per non-empty queue to the
single reusable ``core_worker`` node. Parallelism therefore equals the number
of queues, and sequencing inside a queue is enforced by the scheduler - not
assumed from ``Send``.
"""

from __future__ import annotations

from typing import Any

from langgraph.types import Send

from app.scheduler.task_allocator import allocate_to_cores
from app.scheduler.work_stealing import CoreQueueScheduler
from app.workers.generation_worker import WorkItem

CORE_WORKER_NODE = "core_worker"


def prepare_dispatch(pending: list[dict[str, Any]], queues: int, work_stealing: bool
                     ) -> tuple[CoreQueueScheduler, dict[str, list[int]]]:
    """Build the scheduler and core allocation for one dispatch round.

    Args:
        pending: Work item dictionaries in execution order.
        queues: Number of logical core queues.
        work_stealing: Whether idle queues may steal.

    Returns:
        The scheduler and ``core_id -> labels`` (task id, or ``"<task>.<part>"`` for sub-batches).
    """
    items = [WorkItem(**p) for p in pending]
    # Allocate by position: sub-batches of a split task share its task id.
    positions = allocate_to_cores(list(range(len(items))), max(1, min(queues, len(items) or 1)))
    allocation = {core: [items[i] for i in idx] for core, idx in positions.items()}
    labels = {core: [item.task_id if not item.part else f"{item.task_id}.{item.part}" for item in queue]
              for core, queue in allocation.items()}
    return CoreQueueScheduler(allocation, work_stealing=work_stealing), labels


def dispatch_core_queues(state: dict[str, Any]) -> list[Send] | str:
    """Fan out one ``Send`` per non-empty core queue.

    Args:
        state: Current graph state.

    Returns:
        ``Send`` objects targeting the reusable core worker node, or
        ``"collect_results"`` when there is nothing to run (e.g. a resumed job
        whose batches are all complete).
    """
    round_no = int(state.get("dispatch_round", 0))
    sends = [Send(CORE_WORKER_NODE, {"core_id": core_id, "dispatch_round": round_no})
             for core_id, task_ids in state.get("core_allocation", {}).items() if task_ids]
    return sends or "collect_results"
