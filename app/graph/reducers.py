"""State reducers for results produced by parallel ``Send`` branches.

LangGraph applies the updates of all branches of a superstep together; keys
written by more than one branch must have a reducer, otherwise the update is
rejected (``INVALID_CONCURRENT_GRAPH_UPDATE``). Results are therefore
*appended* (never overwritten) and later consolidated by
:func:`latest_by_task`, which is deterministic regardless of branch order.
"""

from __future__ import annotations

from typing import Any


def concat(left: list[Any] | None, right: list[Any] | None) -> list[Any]:
    """Concatenate two lists (order-preserving, ``None``-safe)."""
    return [*(left or []), *(right or [])]


def merge_dicts(left: dict[str, Any] | None, right: dict[str, Any] | None) -> dict[str, Any]:
    """Shallow-merge two dictionaries; keys from ``right`` win."""
    return {**(left or {}), **(right or {})}


def latest_by_task(results: list[dict[str, Any]]) -> dict[tuple[int, int], dict[str, Any]]:
    """Return the most recent result per task and sub-batch.

    A later dispatch round (retry or repair) supersedes an earlier one; within
    a round, a completed result wins over a failed one.

    Args:
        results: All task results accumulated in the state.

    Returns:
        ``(task_id, part) -> result``; ``part`` is 0 for whole tasks.
    """
    latest: dict[tuple[int, int], dict[str, Any]] = {}
    for result in results:
        slot = (int(result["task_id"]), int(result.get("part", 0)))
        previous = latest.get(slot)
        key = (int(result.get("round", 0)), result.get("status") == "completed")
        if previous is None or key >= (int(previous.get("round", 0)), previous.get("status") == "completed"):
            latest[slot] = result
    return latest
