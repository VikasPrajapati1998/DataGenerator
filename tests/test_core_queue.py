"""Single core queue invariants and scheduler bookkeeping."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from app.core.exceptions import SchedulingError
from app.scheduler.core_queue import CoreQueue
from app.scheduler.work_stealing import CoreQueueScheduler


@dataclass
class Item:
    task_id: int


def test_queue_rejects_second_active_task() -> None:
    queue = CoreQueue("core_1")
    queue.start(Item(1))
    with pytest.raises(SchedulingError):
        queue.start(Item(2))
    queue.finish(1, True)
    assert queue.completed == [1]


def test_finish_without_start_is_an_error() -> None:
    with pytest.raises(SchedulingError):
        CoreQueue("core_1").finish(1, True)


def test_scheduler_executes_own_queue_in_fifo_order() -> None:
    scheduler = CoreQueueScheduler({"core_1": [Item(1), Item(2), Item(3)]})
    order: list[int] = []
    results, skipped = scheduler.run_queue("core_1", lambda item: (order.append(item.task_id) or item.task_id, True))
    assert order == [1, 2, 3] and results == [1, 2, 3] and skipped == []
    assert scheduler.snapshot()["queues"]["core_1"]["completed"] == [1, 2, 3]


def test_acquire_while_active_is_rejected() -> None:
    scheduler = CoreQueueScheduler({"core_1": [Item(1), Item(2)]})
    scheduler.acquire("core_1")
    with pytest.raises(SchedulingError):
        scheduler.acquire("core_1")


def test_failure_is_recorded_and_queue_continues() -> None:
    scheduler = CoreQueueScheduler({"core_1": [Item(1), Item(2)]})
    scheduler.run_queue("core_1", lambda item: (None, item.task_id != 1))
    snap = scheduler.snapshot()["queues"]["core_1"]
    assert snap["failed"] == [1] and snap["completed"] == [2]


def test_exception_releases_the_queue() -> None:
    scheduler = CoreQueueScheduler({"core_1": [Item(1)]})

    def boom(item: Item):
        raise RuntimeError("worker crashed")

    with pytest.raises(RuntimeError):
        scheduler.run_queue("core_1", boom)
    assert scheduler.queues["core_1"].active is None
    assert scheduler.active_count() == 0


def test_no_stealing_when_disabled() -> None:
    scheduler = CoreQueueScheduler({"core_1": [], "core_2": [Item(1), Item(2)]}, work_stealing=False)
    assert scheduler.acquire("core_1") is None
