"""Power-of-two workload planning and construction of work items."""

from __future__ import annotations

from typing import Any

from app.scheduler.task_allocator import calculate_dynamic_nodes, plan_tasks
from app.workers.generation_worker import WorkItem


_MIN_AUTO_BATCH = 250
_MAX_AUTO_BATCH = 5000
_TASKS_PER_CORE = 8
_LLM_BATCH = 100


def auto_batch_size(total_records: int, cpu_cores: int, llm_required: bool) -> int:
    """Choose a target batch size when the request does not fix one.

    * With per-record LLM fields a task runs for minutes, so batches stay small
      (100). A failure or a resume then redoes little work, and progress is visible.
    * Without them, batches aim at about ``8 x cores`` tasks: enough for balanced queues and
      work stealing, few enough that per-task overhead stays negligible. They are
      clamped to 250-5,000 records.

    Args:
        total_records: Requested records.
        cpu_cores: Usable cores.
        llm_required: Whether records need LLM fields.

    Returns:
        Target records per task (the planner still rounds the task count to a power of two).
    """
    if llm_required:
        return min(_LLM_BATCH, max(total_records, 1))
    target = -(-total_records // max(cpu_cores * _TASKS_PER_CORE, 1))
    return max(_MIN_AUTO_BATCH, min(_MAX_AUTO_BATCH, target))


def build_plan(total_records: int, batch_size: int) -> dict[str, Any]:
    """Compute the dynamic task plan.

    Args:
        total_records: Requested records.
        batch_size: Target records per task.

    Returns:
        ``{"power_of_two_tasks", "total_tasks", "tasks", "min_records", "max_records"}``.
    """
    power = calculate_dynamic_nodes(total_records, batch_size)
    tasks = [t.to_dict() for t in plan_tasks(total_records, batch_size)]
    for task in tasks:
        task["record_count"] = int(task["record_count"])
    counts = [t["record_count"] for t in tasks]
    return {"power_of_two_tasks": power, "total_tasks": len(tasks), "tasks": tasks,
            "min_records": min(counts), "max_records": max(counts)}


def assign_models(tasks: list[dict[str, Any]], pool: list[str], strategy: str = "round_robin",
                  weights: dict[str, float] | None = None) -> dict[str, int]:
    """Give every task exactly one model from ``pool`` (in place).

    ``round_robin`` cycles through the pool in task order. ``weighted`` uses
    smooth weighted round-robin, so each model receives a share of tasks
    proportional to its weight while assignments stay interleaved (and
    therefore spread over all core queues). Assignment is deterministic, so a
    resumed job keeps the same model per task. Tasks that already carry a
    ``model_key`` from the pool keep it.

    Args:
        tasks: Planned tasks (updated with ``model_key``).
        pool: Available model keys.
        strategy: ``round_robin`` or ``weighted``.
        weights: ``model key -> weight`` for ``weighted``.

    Returns:
        Number of tasks per model key.
    """
    if not pool:
        return {}
    current = {key: 0.0 for key in pool}
    weight = {key: float((weights or {}).get(key, 1.0)) for key in pool} if strategy == "weighted" else \
        {key: 1.0 for key in pool}
    if sum(weight.values()) <= 0:
        weight = {key: 1.0 for key in pool}
    total = sum(weight.values())
    counts = {key: 0 for key in pool}
    for index, task in enumerate(tasks):
        if task.get("model_key") in counts:
            counts[task["model_key"]] += 1
            continue
        if strategy == "weighted":
            for key in pool:
                current[key] += weight[key]
            chosen = max(pool, key=lambda k: current[k])
            current[chosen] -= total
        else:
            chosen = pool[index % len(pool)]
        task["model_key"] = chosen
        counts[chosen] += 1
    return counts


def generation_items(tasks: list[dict[str, Any]], task_ids: set[int] | None = None, attempt: int = 0
                     ) -> list[dict[str, Any]]:
    """Return generation work items for ``task_ids`` (all tasks when ``None``)."""
    return [WorkItem(task_id=t["task_id"], start_index=t["start_index"], record_count=t["record_count"],
                     attempt=attempt, model_key=t.get("model_key")).to_dict()
            for t in tasks if task_ids is None or t["task_id"] in task_ids]


def retry_items(tasks: list[dict[str, Any]], failed: set[int], attempt: int, pool: list[str]
                ) -> list[dict[str, Any]]:
    """Return work items for failed tasks, each moved to a different pool model.

    A failed task keeps its record range (IDs and batch file stay the same),
    but its primary model rotates ``attempt`` steps through the pool, so a
    model that keeps failing is not retried as primary. The remaining pool
    models are still its fallbacks. The prepare_queues step then spreads the
    retried tasks over every core queue, so idle queues pick them up.

    Args:
        tasks: Planned tasks.
        failed: Ids of the tasks to retry.
        attempt: Retry round (1-based).
        pool: Available pool models.

    Returns:
        Work item dictionaries.
    """
    items = generation_items(tasks, failed, attempt=attempt)
    if len(pool) > 1:
        for item in items:
            start = pool.index(item["model_key"]) if item.get("model_key") in pool else 0
            item["model_key"] = pool[(start + attempt) % len(pool)]
    return items


def repair_items(tasks: list[dict[str, Any]], affected: dict[Any, list[dict[str, Any]]], missing: list[int],
                 round_no: int) -> list[dict[str, Any]]:
    """Return repair items for flagged rows plus regeneration items for missing batches.

    Args:
        tasks: All planned tasks.
        affected: ``task_id -> [{"row", "reason"}]`` from the global report.
        missing: Tasks whose batch is missing or corrupt.
        round_no: Repair round (used to diversify random seeds).

    Returns:
        Work item dictionaries.
    """
    by_id = {t["task_id"]: t for t in tasks}
    items = generation_items(tasks, set(missing), attempt=round_no)
    for task_id, rows in affected.items():
        task = by_id[int(task_id)]
        if int(task_id) in missing:
            continue
        items.append(WorkItem(task_id=task["task_id"], start_index=task["start_index"],
                              record_count=task["record_count"], kind="repair", attempt=round_no,
                              repair_rows=rows, model_key=task.get("model_key")).to_dict())
    return sorted(items, key=lambda i: i["task_id"])
