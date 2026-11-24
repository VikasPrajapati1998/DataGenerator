"""Dynamic Send dispatch, queue preparation and parallel result reducers."""

from __future__ import annotations

from langgraph.types import Send

from app.graph.dispatcher import CORE_WORKER_NODE, dispatch_core_queues, prepare_dispatch
from app.graph.planner import build_plan, generation_items
from app.graph.reducers import concat, latest_by_task, merge_dicts


def test_prepare_dispatch_balances_work() -> None:
    items = generation_items(build_plan(64_000, 1000)["tasks"])
    scheduler, allocation = prepare_dispatch(items, queues=8, work_stealing=True)
    assert len(allocation) == 8 and all(len(v) == 8 for v in allocation.values())
    assert scheduler.work_stealing and scheduler.pending_count() == 64


def test_dispatch_creates_one_send_per_queue_to_one_node() -> None:
    state = {"core_allocation": {"core_1": [1, 2], "core_2": [3], "core_3": []}, "dispatch_round": 2}
    sends = dispatch_core_queues(state)
    assert all(isinstance(s, Send) and s.node == CORE_WORKER_NODE for s in sends)
    assert [s.arg["core_id"] for s in sends] == ["core_1", "core_2"]
    assert all(s.arg["dispatch_round"] == 2 for s in sends)


def test_dispatch_with_no_work_goes_straight_to_collection() -> None:
    assert dispatch_core_queues({"core_allocation": {}}) == "collect_results"


def test_fewer_items_than_queues() -> None:
    items = generation_items(build_plan(2000, 1000)["tasks"])
    _, allocation = prepare_dispatch(items, queues=8, work_stealing=False)
    assert allocation == {"core_1": [1], "core_2": [2]}


def test_reducers() -> None:
    assert concat([1], [2, 3]) == [1, 2, 3]
    assert concat(None, [1]) == [1]
    assert merge_dicts({"a": 1}, {"b": 2, "a": 3}) == {"a": 3, "b": 2}


def test_latest_by_task_prefers_newer_round_and_success() -> None:
    results = [
        {"task_id": 1, "round": 1, "status": "failed"},
        {"task_id": 1, "round": 2, "status": "completed"},
        {"task_id": 2, "round": 1, "status": "completed"},
        {"task_id": 2, "round": 1, "status": "failed"},
        {"task_id": 3, "part": 1, "round": 2, "status": "completed"},
        {"task_id": 3, "part": 2, "round": 2, "status": "failed"},
    ]
    latest = latest_by_task(results)
    assert latest[(1, 0)]["status"] == "completed" and latest[(1, 0)]["round"] == 2
    assert latest[(2, 0)]["status"] == "completed"
    assert latest[(3, 1)]["status"] == "completed" and latest[(3, 2)]["status"] == "failed"
