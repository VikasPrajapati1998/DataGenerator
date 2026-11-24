"""Concurrency guarantees: parallel across queues, sequential within a queue."""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from app.core.cancellation import CancellationToken
from app.scheduler.task_allocator import allocate_to_cores
from app.scheduler.work_stealing import CoreQueueScheduler


@dataclass
class Item:
    task_id: int
    duration: float = 0.02


class Probe:
    """Records overlapping executions per queue and globally."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.active_per_queue: dict[str, int] = defaultdict(int)
        self.max_per_queue: dict[str, int] = defaultdict(int)
        self.active = 0
        self.max_active = 0
        self.ran_on: dict[int, str] = {}

    def task(self, core_id: str):
        def run(item: Item):
            with self.lock:
                self.active_per_queue[core_id] += 1
                self.max_per_queue[core_id] = max(self.max_per_queue[core_id], self.active_per_queue[core_id])
                self.active += 1
                self.max_active = max(self.max_active, self.active)
                self.ran_on[item.task_id] = core_id
            time.sleep(item.duration)
            with self.lock:
                self.active_per_queue[core_id] -= 1
                self.active -= 1
            return item.task_id, True
        return run


def _run(scheduler: CoreQueueScheduler, probe: Probe, cancel: CancellationToken | None = None):
    with ThreadPoolExecutor(max_workers=len(scheduler.queues)) as pool:
        futures = {cid: pool.submit(scheduler.run_queue, cid, probe.task(cid), cancel) for cid in scheduler.queues}
        return {cid: f.result() for cid, f in futures.items()}


def _scheduler(tasks: int, cores: int, stealing: bool = False, durations: dict[int, float] | None = None):
    allocation = allocate_to_cores(list(range(1, tasks + 1)), cores)
    items = {cid: [Item(t, (durations or {}).get(t, 0.02)) for t in ids] for cid, ids in allocation.items()}
    return CoreQueueScheduler(items, work_stealing=stealing)


def test_parallel_across_queues_and_sequential_within() -> None:
    scheduler, probe = _scheduler(20, 4), Probe()
    started = time.perf_counter()
    _run(scheduler, probe)
    elapsed = time.perf_counter() - started
    assert max(probe.max_per_queue.values()) == 1  # never two tasks at once in a queue
    assert probe.max_active == 4  # all four queues ran concurrently
    assert scheduler.max_concurrent_queues == 4
    assert elapsed < 20 * 0.02 * 0.75  # clearly faster than sequential execution


def test_uneven_allocation_completes_every_task_once() -> None:
    scheduler, probe = _scheduler(19, 4), Probe()
    results = _run(scheduler, probe)
    done = sorted(t for res, _ in results.values() for t in res)
    assert done == list(range(1, 20))


def test_concurrency_override_limits_parallelism() -> None:
    scheduler, probe = _scheduler(12, 2), Probe()
    _run(scheduler, probe)
    assert probe.max_active == 2


def test_work_stealing_moves_tasks_to_idle_queue() -> None:
    # core_1 owns slow tasks 1-4, core_2 owns fast tasks 5-8.
    durations = {1: 0.15, 2: 0.15, 3: 0.15, 4: 0.15}
    scheduler, probe = _scheduler(8, 2, stealing=True, durations=durations), Probe()
    _run(scheduler, probe)
    snap = scheduler.snapshot()["queues"]
    assert snap["core_2"]["stolen_in"], "idle core_2 should steal pending work from core_1"
    assert set(snap["core_2"]["stolen_in"]) == set(snap["core_1"]["stolen_out"])
    assert max(probe.max_per_queue.values()) == 1


def test_without_work_stealing_tasks_stay_on_their_queue() -> None:
    durations = {1: 0.05, 2: 0.05}
    scheduler, probe = _scheduler(4, 2, stealing=False, durations=durations), Probe()
    _run(scheduler, probe)
    assert probe.ran_on == {1: "core_1", 2: "core_1", 3: "core_2", 4: "core_2"}


def test_graceful_cancellation_skips_unstarted_tasks() -> None:
    cancel = CancellationToken()
    scheduler, probe = _scheduler(16, 2), Probe()
    original = probe.task

    def cancelling(core_id: str):
        run = original(core_id)

        def wrapped(item: Item):
            if item.task_id == 2:
                cancel.cancel()
            return run(item)
        return wrapped

    probe.task = cancelling  # type: ignore[method-assign]
    results = _run(scheduler, probe, cancel)
    executed = [t for res, _ in results.values() for t in res]
    skipped = [i.task_id for _, sk in results.values() for i in sk]
    assert len(executed) < 16 and skipped
    assert sorted(executed + skipped) == list(range(1, 17))


def test_cancel_marker_file_is_seen_across_tokens(tmp_path) -> None:
    CancellationToken(tmp_path).cancel()
    assert CancellationToken(tmp_path).is_cancelled()
    CancellationToken(tmp_path).clear()
    assert not CancellationToken(tmp_path).is_cancelled()
