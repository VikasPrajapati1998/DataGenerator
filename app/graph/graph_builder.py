"""Builds the LangGraph state graph of the generation workflow.

::

    input -> validate_schema -> check_models -> plan_workload -> detect_resources
          -> allocate_tasks -> prepare_queues ==Send x queues==> core_worker
          -> collect_results --(failed, retries left)--> prepare_queues
                             --(cancelled / failed)----> output
                             --(all done)--------------> global_guardrails
          -> quality_validation --(repairable)--> prepare_queues
                                --(failed)------> output
                                --(passed)------> merge -> final_validation -> output
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from app.graph import nodes
from app.graph.dispatcher import CORE_WORKER_NODE, dispatch_core_queues
from app.graph.state import EngineContext, GraphState


def _after_collect(state: dict[str, Any]) -> str:
    if state.get("status") in ("cancelled", "failed"):
        return "output"
    if state.get("failed_tasks"):
        return "prepare_queues"
    return "global_guardrails"


def _after_quality(state: dict[str, Any]) -> str:
    if state.get("status") == "failed":
        return "output"
    if state.get("global_report", {}).get("passed"):
        return "merge"
    return "prepare_queues"


def build_graph():  # noqa: ANN201 - returns a compiled LangGraph graph
    """Construct and compile the workflow graph.

    Returns:
        The compiled graph; invoke it with an :class:`EngineContext` as context.
    """
    graph = StateGraph(GraphState, context_schema=EngineContext)
    graph.add_node("input", nodes.input_node)
    graph.add_node("validate_schema", nodes.validate_schema_node)
    graph.add_node("check_models", nodes.check_models_node)
    graph.add_node("plan_workload", nodes.plan_workload_node)
    graph.add_node("detect_resources", nodes.detect_resources_node)
    graph.add_node("allocate_tasks", nodes.allocate_tasks_node)
    graph.add_node("prepare_queues", nodes.prepare_queues_node)
    graph.add_node(CORE_WORKER_NODE, nodes.core_worker_node)
    graph.add_node("collect_results", nodes.collect_results_node)
    graph.add_node("global_guardrails", nodes.global_guardrails_node)
    graph.add_node("quality_validation", nodes.quality_validation_node)
    graph.add_node("merge", nodes.merge_node)
    graph.add_node("final_validation", nodes.final_validation_node)
    graph.add_node("output", nodes.output_node)

    graph.add_edge(START, "input")
    graph.add_edge("input", "validate_schema")
    graph.add_edge("validate_schema", "check_models")
    graph.add_edge("check_models", "plan_workload")
    graph.add_edge("plan_workload", "detect_resources")
    graph.add_edge("detect_resources", "allocate_tasks")
    graph.add_edge("allocate_tasks", "prepare_queues")
    graph.add_conditional_edges("prepare_queues", dispatch_core_queues, [CORE_WORKER_NODE, "collect_results"])
    graph.add_edge(CORE_WORKER_NODE, "collect_results")
    graph.add_conditional_edges("collect_results", _after_collect,
                                ["prepare_queues", "global_guardrails", "output"])
    graph.add_edge("global_guardrails", "quality_validation")
    graph.add_conditional_edges("quality_validation", _after_quality, ["prepare_queues", "merge", "output"])
    graph.add_edge("merge", "final_validation")
    graph.add_edge("final_validation", "output")
    graph.add_edge("output", END)
    return graph.compile()
