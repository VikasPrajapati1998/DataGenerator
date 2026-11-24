"""Power-of-two task calculation and record partitioning."""

from __future__ import annotations

import pytest
from pydantic import ValidationError as PydanticValidationError

from app.core.exceptions import SchedulingError
from app.graph.planner import (
    assign_models,
    auto_batch_size,
    build_plan,
    generation_items,
    repair_items,
    retry_items,
)
from app.config.guardrail_config import GuardrailConfig
from app.schemas.input_schema import GenerationRequest
from app.workers.task_splitter import part_items, plan_split
from app.scheduler.task_allocator import calculate_dynamic_nodes, plan_tasks


@pytest.mark.parametrize(("records", "expected"), [
    (500, 1), (1_000, 1), (5_000, 8), (10_000, 16), (50_000, 64), (100_000, 128), (500_000, 512),
    (1_000_000, 1024),
])
def test_specification_table(records: int, expected: int) -> None:
    assert calculate_dynamic_nodes(records, 1000) == expected


@pytest.mark.parametrize("tasks", [1, 2, 8, 16, 64, 128, 1024])
def test_task_counts_are_powers_of_two(tasks: int) -> None:
    records = tasks * 1000 - 1 if tasks > 1 else 1000
    plan = plan_tasks(records, 1000)
    assert len(plan) == tasks
    assert len(plan) & (len(plan) - 1) == 0


@pytest.mark.parametrize(("records", "batch"), [(100_000, 1000), (12_345, 1000), (7, 2), (1_000_000, 1000),
                                                (999, 100), (1, 1000)])
def test_partition_invariants(records: int, batch: int) -> None:
    plan = plan_tasks(records, batch)
    assert sum(t.record_count for t in plan) == records
    assert all(t.record_count >= 1 for t in plan)
    counts = {t.record_count for t in plan}
    assert max(counts) - min(counts) <= 1
    position = 0
    for task in plan:  # contiguous, non-overlapping ranges
        assert task.start_index == position
        position += task.record_count


def test_100k_over_128_tasks_matches_spec() -> None:
    plan = plan_tasks(100_000, 1000)
    assert {t.record_count for t in plan} == {781, 782}


def test_tiny_batch_never_creates_empty_tasks() -> None:
    plan = plan_tasks(3, 1)  # power of two would be 4 -> capped to 3
    assert [t.record_count for t in plan] == [1, 1, 1]


def test_invalid_arguments_raise() -> None:
    with pytest.raises(SchedulingError):
        calculate_dynamic_nodes(0, 1000)
    with pytest.raises(SchedulingError):
        calculate_dynamic_nodes(10, 0)


def test_configurable_batch_size_changes_task_count() -> None:
    assert build_plan(10_000, 500)["total_tasks"] == 32
    assert build_plan(10_000, 5000)["total_tasks"] == 2


def test_round_robin_gives_each_task_exactly_one_pool_model() -> None:
    tasks = build_plan(16_000, 1000)["tasks"]
    pool = ["ollama_local/qwen3:0.6b", "ollama_local/llama3.2:1b", "ollama_local/gemma3:1b",
            "ollama_local/gpt-oss:120b-cloud"]
    counts = assign_models(tasks, pool)
    assert counts == {key: 4 for key in pool}
    assert [t["model_key"] for t in tasks[:5]] == [*pool, pool[0]]
    assert all(i["model_key"] for i in generation_items(tasks))


def test_weighted_assignment_is_proportional_and_interleaved() -> None:
    tasks = build_plan(64_000, 1000)["tasks"]
    counts = assign_models(tasks, ["a", "b", "c"], "weighted", {"a": 2, "b": 1, "c": 1})
    assert counts == {"a": 32, "b": 16, "c": 16}
    assert len({t["model_key"] for t in tasks[:4]}) == 3  # spread, not blocks


def test_assignment_keeps_existing_and_replaces_dropped_models() -> None:
    tasks = build_plan(4000, 1000)["tasks"]
    assign_models(tasks, ["a", "b"])
    assert [t["model_key"] for t in tasks] == ["a", "b", "a", "b"]
    assign_models(tasks, ["a", "c"])  # "b" became unavailable on resume
    assert [t["model_key"] for t in tasks] == ["a", "c", "a", "c"]


def test_failed_tasks_retry_on_a_different_pool_model() -> None:
    tasks = build_plan(4000, 1000)["tasks"]
    assign_models(tasks, ["a", "b", "c"])  # tasks 1..4 -> a, b, c, a
    first = retry_items(tasks, {1, 2}, attempt=1, pool=["a", "b", "c"])
    assert [(i["task_id"], i["model_key"]) for i in first] == [(1, "b"), (2, "c")]
    second = retry_items(tasks, {1}, attempt=2, pool=["a", "b", "c"])
    assert second[0]["model_key"] == "c"
    assert (first[0]["start_index"], first[0]["record_count"]) == (0, 1000)  # same record range
    single = retry_items(tasks, {1}, attempt=1, pool=["a"])
    assert single[0]["model_key"] == "a"


def test_failed_task_split_layout_preserves_its_range() -> None:
    task = {"task_id": 7, "start_index": 6000, "record_count": 101}
    layout = plan_split(task, queues=4, config=GuardrailConfig(min_split_records=25))
    assert [p["record_count"] for p in layout] == [26, 25, 25, 25]
    assert layout[0]["start_index"] == 6000 and layout[-1]["start_index"] + 25 == 6101
    assert [p["part"] for p in layout] == [1, 2, 3, 4]
    assert plan_split(task, 4, GuardrailConfig(min_split_records=60)) == []  # too small to split
    assert plan_split(task, 1, GuardrailConfig(min_split_records=10)) == []  # no spare queue
    assert plan_split(task, 4, GuardrailConfig(split_failed_tasks=False)) == []
    items = part_items({"task_id": 7, "parts": layout}, [2, 4], attempt=1, pool=["a", "b", "c"], base_model="a")
    assert [(i["part"], i["model_key"], i["start_index"]) for i in items] == [(2, "c", 6026), (4, "b", 6076)]


def test_request_validates_model_pool() -> None:
    base = {"schema": {"id": "unique integer"}, "total_records": 10}
    GenerationRequest(**base, model_pool=["mock/a", "mock/b"], model_assignment="weighted", model_weights=[3, 1])
    for bad in ({"model_pool": ["mock/a", "mock/a"]},
                {"model_pool": ["mock/a"], "model_weights": [1.0]},
                {"model_pool": ["mock/a", "mock/b"], "model_assignment": "weighted", "model_weights": [1]}):
        with pytest.raises(PydanticValidationError):
            GenerationRequest(**base, **bad)


def test_auto_batch_size() -> None:
    assert auto_batch_size(100_000, 8, llm_required=False) == 1563  # ~64 tasks for 8 cores
    assert auto_batch_size(1000, 8, llm_required=False) == 250       # lower bound
    assert auto_batch_size(10_000_000, 8, llm_required=False) == 5000  # upper bound
    assert auto_batch_size(100_000, 8, llm_required=True) == 100      # LLM tasks stay small
    assert auto_batch_size(40, 8, llm_required=True) == 40


def test_repair_items_cover_missing_and_flagged() -> None:
    tasks = build_plan(4000, 1000)["tasks"]
    items = repair_items(tasks, {"2": [{"row": 5, "reason": "dup"}]}, missing=[4], round_no=1)
    kinds = {(i["task_id"], i["kind"]) for i in items}
    assert kinds == {(2, "repair"), (4, "generate")}
