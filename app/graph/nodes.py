"""LangGraph node functions (one per step of the logical workflow)."""

from __future__ import annotations

import math
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from langgraph.runtime import Runtime

from app.config.guardrail_config import GuardrailConfig
from app.config.model_registry import ModelConfig
from app.core.cancellation import CancellationToken
from app.core.exceptions import (
    ConfigurationError,
    FeasibilityError,
    LLMError,
    ModelUnavailableError,
    QuotaExceededError,
    RecoveryError,
)
from app.generation.links import resolve_counts
from app.generation.llm_pool import MIN_POOL_VALUES, PoolGrower, build_value_pool, load_pool, save_pool
from app.generation.preflight import benchmark, choose_models, feasibility_message, plan_llm
from app.generation.profile_calibrator import calibrate_profiles
from app.generation.record_builder import RecordBuilder
from app.graph.dispatcher import prepare_dispatch
from app.graph.planner import (
    assign_models,
    auto_batch_size,
    build_plan,
    generation_items,
    repair_items,
    retry_items,
)
from app.graph.reducers import latest_by_task
from app.graph.state import CoreWorkerInput, EngineContext, GraphState
from app.guardrails.global_guardrails import run_global_guardrails
from app.models.model_router import LLMUsage, ModelRouter
from app.models.base_adapter import AvailabilityResult
from app.models.ollama_local import OllamaLocalAdapter
from app.monitoring.execution_tracker import ExecutionTracker
from app.monitoring.logger import get_logger, redact
from app.scheduler.concurrency_manager import plan_concurrency
from app.scheduler.cpu_detector import detect_resources
from app.schemas.input_schema import DatasetSchema, GenerationRequest
from app.schemas.schema_parser import check_record_space, check_unique_capacity, parse_schema
from app.storage.batch_writer import verified_completion
from app.storage.file_manager import (
    JobPaths,
    cleanup_temp_files,
    ensure_disk_space,
    read_json,
    write_json_atomic,
)
from app.storage.formats import count_records, read_header
from app.storage.merger import merge_batches
from app.utils.helpers import format_duration, new_job_id, safe_name, utc_now_iso
from app.workers.services import JobSpec, WorkerServices
from app.workers.task_splitter import (
    consolidate,
    finished_parts,
    load_manifest,
    part_items,
    plan_split,
    remove_split_files,
    write_manifest,
)
from app.workers.worker_manager import TaskExecutor, run_core_queue

logger = get_logger(__name__)
_BYTES_PER_FIELD_ESTIMATE = 24
_PROCESS_MODE_MIN_RECORDS = 20_000
_CALIBRATION_SEED = 1234
_PREFLIGHT_MIN_RECORDS = 1000

Ctx = Runtime[EngineContext]


def _ctx(runtime: Ctx) -> EngineContext:
    return runtime.context


# 1. Input ------------------------------------------------------------------------------
def input_node(state: GraphState, runtime: Ctx) -> dict[str, Any]:
    """Validate the request, create the job directory or load a job to resume."""
    ctx = _ctx(runtime)
    if state.get("resume"):
        paths = JobPaths(Path(state["job_dir"]))
        config = read_json(paths.job_config)
        if not config:
            raise ConfigurationError(f"No job_config.json found in {paths.root}; cannot resume")
        ctx.paths = paths
        ctx.cancel = CancellationToken(paths.root)
        ctx.cancel.clear()
        removed = cleanup_temp_files(paths.root)
        logger.info("Resuming job %s (%d orphaned temp files removed)", config["job_id"], removed)
        request = GenerationRequest(**config["request"])
        ctx.options.update(work_stealing=request.work_stealing, cpu_affinity=request.cpu_affinity,
                           execution_mode=request.execution_mode, max_workers=request.max_workers)
        return {"job_id": config["job_id"], "request": config["request"], "schema": config["schema"],
                "output_format": config["output_format"], "total_records": config["total_records"],
                "batch_size": config["batch_size"], "seed": config["seed"], "model_chain": config["model_chain"],
                "profiles": config["profiles"], "tasks": config["tasks"], "total_tasks": len(config["tasks"]),
                "model_pool": config.get("model_pool", []),
                "retry_round": 0, "repair_round": 0, "dispatch_round": 0, "status": "running"}

    request = GenerationRequest(**state["request"])
    entity = request.data_schema.get("entity") if isinstance(request.data_schema.get("fields"), dict) else None
    schema_name = safe_name(request.schema_name or entity)
    if request.sub_job is not None:  # sub-job of a split run: directory chosen by the split runner
        paths = JobPaths.at(Path(request.sub_job.job_dir))
        job_id = paths.root.name
    else:
        job_id = request.job_id or new_job_id(schema_name)
        output_dir = Path(request.output_dir) if request.output_dir else ctx.settings.output_dir
        paths = JobPaths.create(output_dir.resolve(), job_id, schema_name)
    if read_json(paths.job_config):
        raise ConfigurationError(f"Job {job_id} already exists; use --resume to continue it")
    ctx.paths = paths
    ctx.cancel = CancellationToken(paths.root)
    ctx.cancel.clear()
    ctx.options.update(work_stealing=request.work_stealing, cpu_affinity=request.cpu_affinity,
                       execution_mode=request.execution_mode, max_workers=request.max_workers)
    logger.info("Job %s created at %s", job_id, paths.root)
    return {"job_id": job_id, "job_dir": str(paths.root), "request": request.model_dump(by_alias=True),
            "output_format": request.output_format, "total_records": request.total_records,
            "batch_size": request.batch_size or ctx.settings.default_batch_size,  # None = choose at plan time
            "seed": request.seed if request.seed is not None else random.SystemRandom().randrange(2 ** 31),
            "profiles": dict(request.sub_job.profiles) if request.sub_job else {},
            "retry_round": 0, "repair_round": 0, "dispatch_round": 0, "status": "running"}


# 2. Schema validation -----------------------------------------------------------------------
def _link_sizes(links: dict[str, Any] | None) -> dict[str, int]:
    """Parent rows behind each link of a multi-table table (bounds the values its parent fields can take)."""
    if not links:
        return {}
    sizes = {name: int(link.get("eligible_rows") or link.get("rows") or 0) for name, link in
             (links.get("links") or {}).items()}
    per_parent = links.get("per_parent")
    if per_parent:
        sizes[per_parent["link"]] = int(per_parent.get("rows") or 0)
        sizes["__child_max__"] = int(per_parent.get("max") or 0)
    return {k: v for k, v in sizes.items() if v}


def validate_schema_node(state: GraphState, runtime: Ctx) -> dict[str, Any]:  # noqa: ARG001
    """Parse and validate the schema; reject anything that cannot be generated."""
    if state.get("resume"):
        schema = DatasetSchema(**state["schema"])
    else:
        request = state["request"]
        schema = parse_schema(request["schema"], request.get("generation_instructions"))
    RecordBuilder(schema, state["total_records"])  # probes Faker providers and numeric ranges
    GuardrailConfig.from_requirements(state["request"].get("quality_requirements"))  # validates the overrides
    dataset_size = int((state["request"].get("sub_job") or {}).get("dataset_total_records") or state["total_records"])
    check_unique_capacity(schema, dataset_size)
    for name, spec in schema.fields.items():  # exact counts must add up to the whole table
        if spec.counts is not None:
            resolve_counts(name, list(spec.allowed_values or []), spec.counts, spec.weights, dataset_size)
    schema = check_record_space(schema, dataset_size, _link_sizes(state["request"].get("links")))
    llm_required = bool(schema.llm_fields)
    logger.info("Schema OK: %d fields (%d LLM, %d unique, %d rules), locale %s", len(schema.fields),
                len(schema.llm_fields), len(schema.unique_fields), len(schema.rules), schema.locale)
    return {"schema": schema.model_dump(mode="json"), "llm_required": llm_required}


# 3. Model availability ------------------------------------------------------------------------
def check_models_node(state: GraphState, runtime: Ctx) -> dict[str, Any]:
    """Resolve the model chain, check availability, apply fallbacks and calibrate profiles."""
    ctx = _ctx(runtime)
    request = state["request"]
    schema = DatasetSchema(**state["schema"])
    profiles = dict(state.get("profiles") or {})
    missing_profiles = [n for n in schema.profile_fields if schema.fields[n].profile is None and n not in profiles]
    if not state.get("llm_required") and not missing_profiles:
        # Every field is generated by Python/Faker: no model is called, so none is checked or assigned.
        logger.info("No LLM needed for this schema (no LLM fields, all ranges given); skipping model checks")
        return {"model_chain": [], "model_status": {}, "profiles": profiles, "model_pool": [], "ollama_loaded": []}

    # Precedence: the request's model_pool, then an explicit model_name/provider, then SDE_MODEL_POOL,
    # then the default model.
    if state.get("model_pool"):
        pool_keys = list(state["model_pool"])
    elif request.get("model_pool"):
        pool_keys = list(request["model_pool"])
    elif request.get("model_name") or request.get("model_provider"):
        pool_keys = []
    else:
        pool_keys = list(ctx.settings.model_pool)
    if state.get("model_chain"):
        candidates = [ModelConfig(**c) for c in state["model_chain"]]
    else:
        if pool_keys:
            candidates = [_resolve_key(ctx, key) for key in pool_keys]
            pool_keys = [c.key for c in candidates]
        else:
            candidates = [ctx.registry.resolve(request.get("model_provider"), request.get("model_name"))]
        for key in request.get("fallback_models", []):
            config = _resolve_key(ctx, key)
            if config.key not in {c.key for c in candidates}:
                candidates.append(config)
    if ctx.settings.max_retries is not None:
        candidates = [c.with_retry_cap(ctx.settings.max_retries) for c in candidates]
        logger.info("SDE_MAX_RETRIES=%d: at most %d attempt(s) per LLM call, %d regeneration round(s), "
                    "task retry round(s) and repair round(s)", ctx.settings.max_retries,
                    ctx.settings.max_retries + 1, ctx.settings.max_retries)
    router = ModelRouter(candidates, ctx.settings, cancel=ctx.cancel)
    cache: dict[str, Any] = ctx.options.setdefault("availability_cache", {})
    results = _check_availability(router, candidates, cache)
    status: dict[str, Any] = {}
    available: list[ModelConfig] = []
    for config in candidates:
        result = results[config.key]
        ctx.registry.set_availability(config.key, "available" if result.available else "unavailable")
        status[config.key] = {"available": result.available, "detail": result.detail, **result.metadata}
        if result.available:
            if result.metadata.get("remote_host"):
                config.execution = "cloud"  # served by Ollama Cloud: no local RAM
                config.approx_memory_mb = 0
            elif config.provider == "ollama_local" and result.metadata.get("size_bytes"):
                config.approx_memory_mb = int(result.metadata["size_bytes"] / 1048576 * 1.3)
            available.append(config)
            logger.info("Model %s: available (%s)", config.key, result.detail)
    for detail, keys in _group_unavailable(status).items():
        logger.warning("Unavailable: %s -> %s", ", ".join(keys), detail)
    # Cloud models can be listed yet unusable (no credits, plan limits): one tiny call finds out up front.
    for config in [c for c in available if c.execution == "cloud"]:
        problem = _cached(cache, f"probe:{config.key}", lambda c=config: _probe_cloud_model(router, c))
        if problem:
            available.remove(config)
            status[config.key].update(available=False, detail=problem)
            logger.warning("Model %s excluded before the run: %s", config.key, problem)
    loaded: set[str] = set()
    local_adapter = next((router.adapter(c) for c in available if c.provider == "ollama_local"), None)
    if isinstance(local_adapter, OllamaLocalAdapter):
        loaded = {m.get("name") or m.get("model") for m in local_adapter.loaded_models()} - {None}
    router.close()
    if not available:
        if request.get("allow_no_llm"):
            return _continue_without_llm(schema, missing_profiles, status, profiles)
        raise ModelUnavailableError(describe_unavailable(status))
    if missing_profiles:
        usage = LLMUsage()
        # Calibration needs world knowledge, not variety: strongest model first (cloud, then the largest
        # local one), deterministic decoding, fixed seed. This keeps ranges stable between runs.
        ranked = sorted(available, key=lambda c: (c.execution != "cloud", -(c.approx_memory_mb or 0)))
        calibration_chain = [c.model_copy(update={"temperature": 0.0, "seed": _CALIBRATION_SEED}) for c in ranked]
        logger.info("Calibrating profiles with %s (temperature 0, seed %d)", calibration_chain[0].key,
                    _CALIBRATION_SEED)
        calibration_router = ModelRouter(calibration_chain, ctx.settings, cancel=ctx.cancel)
        try:
            calibrated = calibrate_profiles(schema, calibration_router, request.get("generation_instructions"),
                                            usage)
        finally:
            calibration_router.close()
        profiles.update({name: {cat: rng.model_dump() for cat, rng in cats.items()}
                         for name, cats in calibrated.items()})
        ctx.options["calibration_usage"] = usage.to_dict()
    available_keys = [c.key for c in available]
    pool = [k for k in pool_keys if k in available_keys] or available_keys[:1]
    dropped = [k for k in pool_keys if k not in pool]
    if dropped:
        logger.warning("Model pool: %s unavailable and removed from the pool", dropped)
    logger.info("Model pool for generation tasks: %s", pool)
    return {"model_chain": [c.model_dump(mode="json") for c in available], "model_status": status,
            "profiles": profiles, "model_pool": pool, "ollama_loaded": sorted(loaded)}


_AVAILABILITY_TTL_SECONDS = 300.0
_SERVER_WAIT_SECONDS = 10.0
_SERVER_RETRY_SECONDS = 2.0


def _group_unavailable(status: dict[str, Any]) -> dict[str, list[str]]:
    """``reason -> model keys`` for unavailable models (one shared server failure appears once)."""
    groups: dict[str, list[str]] = {}
    for key, info in status.items():
        if not info.get("available"):
            groups.setdefault(str(info.get("detail")), []).append(key)
    return groups


def describe_unavailable(status: dict[str, Any]) -> str:
    """One short, actionable message for "no model is usable".

    Args:
        status: Per-model availability as built by ``check_models``.

    Returns:
        The message, e.g. "Ollama server not reachable at http://127.0.0.1:11434. Start it with
        `ollama serve` (or check OLLAMA_HOST). Models: a, b".
    """
    parts = []
    for detail, keys in _group_unavailable(status).items():
        if detail.startswith("Ollama server not reachable"):
            parts.append(f"{detail}. Start it with `ollama serve` (or check OLLAMA_HOST). Models: {', '.join(keys)}")
        else:
            parts.append(f"{', '.join(keys)}: {detail}")
    return "No usable model. " + " | ".join(parts) + (
        ". Or run with --allow-no-llm to fill LLM text fields with placeholder text.")


def _continue_without_llm(schema: DatasetSchema, missing_profiles: list[str], status: dict[str, Any],
                          profiles: dict[str, Any]) -> dict[str, Any]:
    """``--allow-no-llm``: replace LLM fields by Faker/range generators so the job can run offline.

    Raises:
        ModelUnavailableError: If a field cannot be generated without a model.
    """
    if missing_profiles:
        raise ModelUnavailableError(f"{describe_unavailable(status)} --allow-no-llm cannot help: "
                                    f"{missing_profiles} need a model to calibrate their ranges; put explicit "
                                    "'profile' ranges in the schema to run offline")
    fields = dict(schema.fields)
    for name in schema.llm_fields:
        spec = fields[name]
        if spec.type == "string":
            fields[name] = spec.model_copy(update={"generator": "faker", "faker_method": "paragraph",
                                                   "faker_args": {"nb_sentences": 2}})
        elif spec.type in ("integer", "float") and spec.min is not None and spec.max is not None:
            fields[name] = spec.model_copy(update={"generator": "range"})
        else:
            raise ModelUnavailableError(f"{describe_unavailable(status)} --allow-no-llm cannot fill {name!r} "
                                        f"({spec.type} without min/max) without a model")
    offline = schema.model_copy(update={"fields": fields, "semantic_fields": []})
    logger.warning("No model reachable and --allow-no-llm is set: LLM fields %s are filled with placeholder "
                   "text (Faker paragraphs). The data stays valid, but these columns are not contextual.",
                   schema.llm_fields)
    return {"schema": offline.model_dump(mode="json"), "llm_required": False, "model_chain": [],
            "model_status": status, "profiles": profiles, "model_pool": [], "ollama_loaded": []}


def _cached(cache: dict[str, Any], key: str, compute: Any) -> Any:
    """Return a cached probe result (shared by the sub-jobs of a split run) or compute it."""
    hit = cache.get(key)
    if hit is not None and time.monotonic() - hit[0] < _AVAILABILITY_TTL_SECONDS:
        return hit[1]
    value = compute()
    cache[key] = (time.monotonic(), value)
    return value


def _check_availability(router: ModelRouter, candidates: list[ModelConfig],
                        cache: dict[str, Any]) -> dict[str, AvailabilityResult]:
    """Check every candidate quickly.

    All local Ollama models share one ``/api/tags`` call (so an unreachable server
    costs one failed connection, not one per model). Other providers are probed
    in parallel. Results are cached for a few minutes, which lets later sub-jobs
    of a split run skip the checks.
    """
    results: dict[str, AvailabilityResult] = {}
    local = [c for c in candidates if c.provider == "ollama_local"]
    if local:
        adapter = router.adapter(local[0])
        assert isinstance(adapter, OllamaLocalAdapter)
        listing = _cached(cache, f"tags:{adapter.base_url}", lambda: _list_models(adapter))
        for config in local:
            results[config.key] = (router.adapter(config).availability_from(listing)  # type: ignore[attr-defined]
                                   if isinstance(listing, list) else AvailabilityResult(False, listing))
    others = [c for c in candidates if c.provider != "ollama_local"]
    if others:
        with ThreadPoolExecutor(max_workers=min(8, len(others)), thread_name_prefix="model-check") as pool:
            futures = {c.key: pool.submit(_cached, cache, f"check:{c.key}",
                                          lambda c=c: router.adapter(c).check_availability()) for c in others}
        results.update({key: future.result() for key, future in futures.items()})
    return results


def _list_models(adapter: OllamaLocalAdapter) -> list[dict[str, Any]] | str:
    """Return the server's model list, or why the server could not be reached.

    Ollama may still be starting, so an unreachable server is retried for up to
    ``_SERVER_WAIT_SECONDS`` before giving up.
    """
    deadline = time.monotonic() + _SERVER_WAIT_SECONDS
    waited = False
    while True:
        try:
            return adapter.list_models()
        except LLMError:
            if time.monotonic() + _SERVER_RETRY_SECONDS > deadline:
                return f"Ollama server not reachable at {adapter.base_url}"
            if not waited:
                logger.info("Ollama server at %s is not answering; waiting up to %.0fs in case it is starting",
                            adapter.base_url, _SERVER_WAIT_SECONDS)
                waited = True
            time.sleep(_SERVER_RETRY_SECONDS)


def _probe_cloud_model(router: ModelRouter, config: ModelConfig) -> str | None:
    """Make one minimal request; return why the model is unusable, or ``None`` if it answered.

    Transient errors (timeouts, 5xx) do not exclude the model: the run-time
    retries and fallbacks handle those.
    """
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}
    try:
        router.adapter(config).generate_json("Answer with JSON only.", 'Return {"ok": true}.', schema)
    except QuotaExceededError as exc:
        return f"quota/credits exhausted ({redact(str(exc))[:160]})"
    except ModelUnavailableError as exc:
        return f"not usable ({redact(str(exc))[:160]})"
    except LLMError as exc:
        logger.info("Probe of %s inconclusive (%s); keeping it", config.key, type(exc).__name__)
    return None


def _resolve_key(ctx: EngineContext, key: str) -> ModelConfig:
    """Resolve a ``provider/model`` registry key (``ollama/<name>`` is accepted)."""
    provider, _, name = key.partition("/")
    return ctx.registry.resolve(provider, name or None)


# 4. Power-of-two planner -------------------------------------------------------------------------
def plan_workload_node(state: GraphState, runtime: Ctx) -> dict[str, Any]:  # noqa: ARG001
    """Compute the power-of-two task count and partition the records."""
    if state.get("tasks"):
        tasks = state["tasks"]
        logger.info("Reusing stored plan: %d tasks", len(tasks))
        return {"total_tasks": len(tasks)}
    batch_size = state.get("batch_size") or auto_batch_size(state["total_records"], os.cpu_count() or 1,
                                                             bool(state.get("llm_required")))
    if not state.get("batch_size"):
        logger.info("Batch size chosen at run time: %d records per task", batch_size)
    plan = build_plan(state["total_records"], batch_size)
    offset = int((state["request"].get("sub_job") or {}).get("record_offset", 0))
    if offset:  # sub-job of a split run: its records continue the global index (and so the IDs)
        for task in plan["tasks"]:
            task["start_index"] += offset
        logger.info("Sub-job covers global records %d-%d", offset + 1, offset + state["total_records"])
    if plan["total_tasks"] != plan["power_of_two_tasks"]:
        logger.warning("Power-of-two count %d exceeds %d records; capped to avoid empty tasks",
                       plan["power_of_two_tasks"], state["total_records"])
    logger.info("Planned %d dynamic tasks for %d records (batch size %d): %d-%d records per task",
                plan["total_tasks"], state["total_records"], batch_size, plan["min_records"],
                plan["max_records"])
    return {"tasks": plan["tasks"], "total_tasks": plan["total_tasks"], "batch_size": batch_size}


# 5. CPU core detection --------------------------------------------------------------------------
def detect_resources_node(state: GraphState, runtime: Ctx) -> dict[str, Any]:
    """Detect cores, RAM, GPUs and check free disk space."""
    ctx = _ctx(runtime)
    resources = detect_resources()
    estimate = state["total_records"] * len(state["schema"]["fields"]) * _BYTES_PER_FIELD_ESTIMATE * 3
    free = ensure_disk_space(ctx.paths.root, estimate)  # type: ignore[union-attr]
    return {"resources": {**resources.to_dict(), "disk_free_bytes": free, "disk_required_estimate": estimate}}


# 6. CPU-based allocation (+ job freeze) -------------------------------------------------------------
def allocate_tasks_node(state: GraphState, runtime: Ctx) -> dict[str, Any]:
    """Compute effective concurrency, freeze the job config and start the executor."""
    ctx = _ctx(runtime)
    paths = ctx.paths
    assert paths is not None
    from app.scheduler.cpu_detector import GPUInfo, ResourceInfo  # pylint: disable=import-outside-toplevel

    res = dict(state["resources"])
    resources = ResourceInfo(
        logical_cores=res["logical_cores"], physical_cores=res["physical_cores"], usable_cores=res["usable_cores"],
        total_ram_mb=res["total_ram_mb"], available_ram_mb=res["available_ram_mb"],
        gpus=[GPUInfo(**g) for g in res["gpus"]], affinity_supported=res["affinity_supported"])
    chain = [ModelConfig(**c) for c in state["model_chain"]]
    tasks = state["tasks"]
    request = state["request"]
    plan = plan_concurrency(resources, chain[0] if chain else None, ctx.settings, len(tasks),
                            ctx.options.get("max_workers"), llm_required=bool(state.get("llm_required")), pool=chain,
                            loaded_models=set(state.get("ollama_loaded") or []))
    for reason in plan.reasons:
        if reason.startswith("Low RAM"):
            logger.warning(reason)
    pool = list(state.get("model_pool") or ([chain[0].key] if chain else []))
    weights = dict(zip(request.get("model_pool") or [], request.get("model_weights") or []))
    assignment = assign_models(tasks, pool, request.get("model_assignment", "round_robin"), weights)
    if assignment:
        logger.info("Model assignment (%s): %s", request.get("model_assignment", "round_robin"), assignment)
    guardrails = GuardrailConfig.from_requirements(request.get("quality_requirements"), ctx.settings.max_retries,
                                               ctx.settings.max_invalid_fraction)
    spec = JobSpec(job_id=state["job_id"], job_dir=str(paths.root), request=request, schema=state["schema"],
                   total_records=state["total_records"], output_format=state["output_format"],
                   instructions=request.get("generation_instructions"), model_chain=state["model_chain"],
                   guardrails=guardrails.model_dump(), profiles=state.get("profiles", {}), seed=state["seed"],
                   batch_size=state["batch_size"], tasks=tasks, model_pool=pool)
    write_json_atomic(paths.job_config, {**spec.to_dict(), "frozen_at": utc_now_iso()})

    llm_slots = {c.key: plan.llm_slots.get(c.key, plan.llm_concurrency) for c in chain}
    if ctx.services is None:
        from threading import BoundedSemaphore  # pylint: disable=import-outside-toplevel

        ctx.services = WorkerServices(spec, {k: BoundedSemaphore(v) for k, v in llm_slots.items()}, 1, ctx.cancel)
    llm_plan: dict[str, Any] = {}
    if state.get("llm_required"):
        llm_plan = _prepare_llm(state, ctx, chain, llm_slots, bool(resources.gpus))
        spec.llm_pool_path = llm_plan.get("pool_path")
        spec.disabled_models = sorted(llm_plan.get("dropped", {}))
        write_json_atomic(paths.job_config, {**spec.to_dict(), "frozen_at": utc_now_iso()})
    per_record_llm = bool(state.get("llm_required")) and llm_plan.get("mode") != "pooled"

    mode = ctx.options.get("execution_mode", "auto")
    if mode == "auto":
        mode = "process" if (state["total_records"] >= _PROCESS_MODE_MIN_RECORDS and not per_record_llm
                             and plan.queues > 1) else "thread"
    if ctx.executor is None:
        ctx.executor = TaskExecutor(mode, ctx.services, paths.job_config, plan.queues, llm_slots,
                                    task_timeout=ctx.settings.task_timeout or None, cancel=ctx.cancel)

    def on_stall(idle: float) -> None:
        ctx.options["stalled_seconds"] = round(idle)
        ctx.cancel.cancel()  # queues stop taking work; worker processes are terminated after a grace period

    ctx.tracker = ctx.tracker or ExecutionTracker(state["job_id"], len(tasks), state["total_records"],
                                                  stall_timeout=float(ctx.settings.stall_timeout),
                                                  on_stall=on_stall)
    for key in ("calibration_usage", "pool_usage"):
        if usage := ctx.options.pop(key, None):
            ctx.tracker.llm.merge(usage)
    ctx.tracker.llm_required = per_record_llm
    if mode == "thread":
        ctx.services.llm_listener = ctx.tracker  # live LLM progress; process mode reports per finished task
    if per_record_llm:
        chunk = ctx.services.enricher_for(llm_plan["models"][0]).chunk_size
        requests = math.ceil(state["total_records"] / chunk)
        projected = (format_duration(llm_plan["eta_seconds"]) if llm_plan.get("eta_seconds") is not None
                     else "not measured (small job)")
        logger.info("LLM workload: ~%d requests of %d record(s) for %d records; projected run time %s",
                    requests, chunk, state["total_records"], projected)
    ctx.tracker.start_monitoring()

    completed = [t["task_id"] for t in tasks
                 if verified_completion(paths, t["task_id"], t["record_count"], state["output_format"])]
    for task_id in completed:
        ctx.tracker.task_finished({"task_id": task_id, "status": "completed", "skipped": True})
    if completed:
        logger.info("%d task(s) already completed and verified; they will not be regenerated", len(completed))
    pending: list[dict[str, Any]] = []
    for task in tasks:
        manifest = _safe_manifest(paths, task["task_id"])
        if task["task_id"] in completed:
            if manifest is not None:  # consolidated before an interruption; finish the clean-up
                remove_split_files(paths, manifest, state["output_format"])
            continue
        if manifest is None:
            pending += generation_items([task])
            continue
        done = finished_parts(paths, manifest, state["output_format"])
        missing = [p["part"] for p in manifest["parts"] if p["part"] not in done]
        logger.info("Task %d was split earlier: %d/%d sub-batches already done", task["task_id"], len(done),
                    len(manifest["parts"]))
        pending += part_items(manifest, missing, 0, pool, task.get("model_key"))
    logger.info("Concurrency: %d logical core queues, LLM concurrency %d, mode=%s (%s)", plan.queues,
                plan.llm_concurrency, mode, "; ".join(plan.reasons))
    return {"concurrency": {**plan.to_dict(), "model_assignment": assignment, "llm_plan": llm_plan},
            "execution_mode": mode, "pending_work": pending, "already_completed": completed, "phase": "generate",
            "tasks": tasks, "model_pool": pool}


def _pool_path(ctx: EngineContext, request: dict[str, Any]) -> Path:
    """``llm_pool.json``: in the split-run root (shared by every sub-job) or the job's metadata folder."""
    sub_job = request.get("sub_job") or {}
    if sub_job.get("registry_path"):
        return Path(sub_job["registry_path"]).parent / "llm_pool.json"
    assert ctx.paths is not None
    return ctx.paths.metadata / "llm_pool.json"


def _plan_from_pool(pool_path: Path, pool_keys: list[str], total: int) -> dict[str, Any] | None:
    """The stored plan of an existing pool (resume, later sub-jobs), so the speed test is not repeated."""
    data = load_pool(pool_path)
    if not data.get("items"):
        return None
    plan = dict(data.get("plan") or {})
    # A pool from an older version has no stored plan: grow it with one model only, since local models
    # running together on a CPU-only machine slow each other down.
    plan.setdefault("models", pool_keys[:1])
    plan.setdefault("dropped", {})
    plan.update(mode="pooled", eta_seconds=round(total / 3000), reused_pool=True)
    plan.setdefault("pool_target", len(data["items"]))
    return plan


def _prepare_llm(state: GraphState, ctx: EngineContext, chain: list[ModelConfig], llm_slots: dict[str, int],
                 has_gpu: bool) -> dict[str, Any]:
    """Speed-test the models, choose the LLM mode, refuse hopeless jobs, and build the text pool if pooled.

    The plan is computed once per run: sub-jobs of a split run and resumed jobs
    find it in ``llm_pool.json`` and reuse the pool instead of rebuilding it.

    Raises:
        FeasibilityError: If the projected run time exceeds ``SDE_MAX_ETA_HOURS`` and ``force`` is not set.
        ModelUnavailableError: If no model passes the speed test or the pool cannot be built.
    """
    services, settings, request = ctx.services, ctx.settings, state["request"]
    assert services is not None
    schema = services.schema
    sub_job = request.get("sub_job") or {}
    total = int(sub_job.get("dataset_total_records") or state["total_records"])
    cache: dict[str, Any] = ctx.options.setdefault("availability_cache", {})
    requested_mode = request.get("llm_mode") or settings.llm_mode
    pool_keys = [c.key for c in chain if c.key in set(state.get("model_pool") or [])] or [c.key for c in chain]
    pool_path = _pool_path(ctx, request)
    if total < _PREFLIGHT_MIN_RECORDS and requested_mode != "pooled":
        # Small jobs finish quickly whatever the model speed; a speed test would cost more than it saves.
        return {"mode": "per_record", "models": pool_keys, "dropped": {}, "measured": False}
    info = cache.get("llm_plan")
    if info is None and requested_mode != "per_record":
        info = _plan_from_pool(pool_path, pool_keys, total)
        if info is not None:
            logger.info("Pooled mode: reusing the existing text pool %s (no speed test, no rebuild)", pool_path)
    if info is None:
        candidates = [c for c in chain if c.key in set(state.get("model_pool") or [])] or chain
        results = benchmark(services, candidates, settings, state["seed"])
        kept, dropped = choose_models(candidates, results, has_gpu)
        for key, reason in dropped.items():
            logger.warning("Model %s not used for this job: %s", key, reason)
        if not kept:
            raise ModelUnavailableError("Every model failed the speed test: " +
                                        "; ".join(f"{k}: {v}" for k, v in dropped.items()))
        text_only = all(schema.fields[f].type == "string" for f in schema.llm_fields)
        llm_plan = plan_llm(kept, results, dropped, llm_slots, total, requested_mode, settings, text_only)
        too_long = settings.max_eta_hours and llm_plan.eta_seconds > settings.max_eta_hours * 3600
        if too_long and not request.get("force"):
            raise FeasibilityError(feasibility_message(llm_plan, total, settings.max_eta_hours))
        if too_long:
            logger.warning("Projected run time %s exceeds SDE_MAX_ETA_HOURS; continuing because of --force",
                           format_duration(llm_plan.eta_seconds))
        info = llm_plan.to_dict()
    cache["llm_plan"] = info
    info = dict(info)
    services.disabled_models.update(info["dropped"])
    logger.info("LLM mode: %s for %s records using %s; projected run time %s", info["mode"].upper(),
                f"{total:,}", ", ".join(info["models"]), format_duration(info["eta_seconds"]))
    if info["mode"] != "pooled":
        return info
    if not load_pool(pool_path).get("items"):
        budget = settings.pool_build_minutes * 60
        logger.info("Pooled mode: building a pool of up to %d LLM answers (budget %s), then filling every record "
                    "from it", info["pool_target"], format_duration(budget))
        usage = LLMUsage()
        enrichers = [services.enricher_for(key) for key in info["models"]]
        items = build_value_pool(schema, services.builder, enrichers, info["pool_target"], budget,
                                 state["seed"], usage)
        if ctx.tracker is not None:
            ctx.tracker.add_llm_usage(usage)
        else:
            ctx.options["pool_usage"] = usage.to_dict()
        if len(items) < MIN_POOL_VALUES // 3:
            raise ModelUnavailableError(f"Pooled mode could only build {len(items)} LLM answers in "
                                        f"{format_duration(budget)}; raise SDE_POOL_BUILD_MINUTES or use a "
                                        "faster model")
        save_pool(pool_path, schema, items, plan={k: v for k, v in info.items() if k != "reused_pool"})
    filler = services.use_pool(pool_path)
    logger.info("Pooled mode: %d pooled answers in %s; near-duplicate text checks are off for pooled text",
                filler.size, pool_path)
    if settings.pool_grow and filler.size < settings.pool_max_values:
        services.pool_grower = PoolGrower(schema, services.builder,
                                          [services.enricher_for(key) for key in info["models"]], filler,
                                          pool_path, settings.pool_max_values, state["seed"], LLMUsage())
        services.pool_grower.start()
    info["pool_path"] = str(pool_path)
    return info


# 7. Core queue manager ----------------------------------------------------------------------------
def prepare_queues_node(state: GraphState, runtime: Ctx) -> dict[str, Any]:
    """Distribute pending work over logical core queues for the next dispatch round."""
    ctx = _ctx(runtime)
    pending = state.get("pending_work", [])
    queues = int(state["concurrency"]["queues"])
    scheduler, allocation = prepare_dispatch(pending, queues, bool(ctx.options.get("work_stealing")))
    ctx.scheduler = scheduler
    if ctx.tracker is not None:
        ctx.tracker.touch()
    round_no = int(state.get("dispatch_round", 0)) + 1
    logger.info("Dispatch round %d (%s): %d item(s) over %d queue(s): %s", round_no, state.get("phase"),
                len(pending), len(allocation), {k: len(v) for k, v in allocation.items()})
    return {"core_allocation": allocation, "dispatch_round": round_no}


# 8-12. Reusable core-queue worker (generation, node validation, guardrails, batch writing) ---------
def core_worker_node(payload: CoreWorkerInput, runtime: Ctx) -> dict[str, Any]:
    """Run one logical core queue: its tasks sequentially, one at a time."""
    ctx = _ctx(runtime)
    assert ctx.scheduler and ctx.executor and ctx.tracker
    results = run_core_queue(payload["core_id"], ctx.scheduler, ctx.executor, ctx.tracker, ctx.cancel,
                             pin=bool(ctx.options.get("cpu_affinity")))
    for result in results:
        result["round"] = payload["dispatch_round"]
    snapshot = ctx.scheduler.queues[payload["core_id"]].snapshot()
    return {"task_results": results, "queue_reports": [{**snapshot, "round": payload["dispatch_round"]}]}


# 13. Result collector --------------------------------------------------------------------------------
def collect_results_node(state: GraphState, runtime: Ctx) -> dict[str, Any]:
    """Consolidate parallel results and decide on retries."""
    ctx = _ctx(runtime)
    assert ctx.paths is not None and ctx.services is not None and ctx.tracker is not None
    latest = latest_by_task(state.get("task_results", []))
    tasks = state["tasks"]
    fmt = state["output_format"]
    completed = set(state.get("already_completed", [])) | {
        t for (t, part), r in latest.items() if part == 0 and r.get("status") == "completed"}
    split_pending: dict[int, tuple[dict[str, Any], list[int]]] = {}
    errors: list[str] = []
    for task in tasks:
        task_id = task["task_id"]
        if task_id in completed:
            continue
        manifest = _safe_manifest(ctx.paths, task_id)
        if manifest is None:
            if (task_id, 0) in latest:
                errors.append(f"task {task_id}: {latest[(task_id, 0)].get('error')}")
            continue
        done = finished_parts(ctx.paths, manifest, fmt)
        missing = [p["part"] for p in manifest["parts"] if p["part"] not in done]
        if missing:
            split_pending[task_id] = (manifest, missing)
            errors += [f"task {task_id} part {p}: {latest[(task_id, p)].get('error')}" for p in missing
                       if (task_id, p) in latest]
            continue
        try:
            consolidate(ctx.services, manifest)
        except RecoveryError as exc:
            logger.error("Task %d: consolidation failed (%s); the task will be regenerated whole", task_id, exc)
            remove_split_files(ctx.paths, manifest, fmt)
            errors.append(f"task {task_id}: {exc}")
            continue
        completed.add(task_id)
        ctx.tracker.task_consolidated(task_id)
    failed = sorted(t["task_id"] for t in tasks if t["task_id"] not in completed)
    update: dict[str, Any] = {"completed_tasks": sorted(completed), "failed_tasks": failed,
                              "batch_files": [str(ctx.paths.batch_file(t, fmt)) for t in sorted(completed)]}
    if ctx.scheduler is not None:
        logger.info("Round %d finished: max concurrent queues %d", state.get("dispatch_round", 0),
                    ctx.scheduler.max_concurrent_queues)
    if ctx.cancel.is_cancelled():
        if stalled := ctx.options.get("stalled_seconds"):
            update.update(status="failed", errors=[
                f"stalled: no progress for {stalled}s (SDE_STALL_TIMEOUT={ctx.settings.stall_timeout}). "
                "Check that the model server is responding; finished batches are kept, so 'resume' continues."])
        else:
            update["status"] = "cancelled"
        return update
    if not failed:
        return update
    guardrails = GuardrailConfig.from_requirements(state["request"].get("quality_requirements"), ctx.settings.max_retries,
                                               ctx.settings.max_invalid_fraction)
    retry_round = int(state.get("retry_round", 0))
    if retry_round >= guardrails.max_task_retries:
        logger.error("Retry limit reached: %d task(s) still failing after %d retry round(s) "
                     "(max_task_retries / SDE_MAX_RETRIES). Fix the cause and run 'resume' to continue; "
                     "completed batches are kept.", len(failed), retry_round)
        update.update(status="failed", errors=[f"{len(failed)} task(s) failed after {retry_round} retry round(s)",
                                               *errors[:20]])
        return update
    attempt = retry_round + 1
    pool = list(state.get("model_pool") or [])
    queues = int(state["concurrency"]["queues"])
    by_id = {t["task_id"]: t for t in tasks}
    items: list[dict[str, Any]] = []
    for task_id in failed:
        task = by_id[task_id]
        if task_id in split_pending:
            manifest, missing = split_pending[task_id]
        else:
            layout = plan_split(task, queues, guardrails)
            if not layout:
                items += retry_items(tasks, {task_id}, attempt, pool)
                continue
            manifest, missing = write_manifest(ctx.paths, task, layout), [p["part"] for p in layout]
            logger.warning("Task %d failed; splitting its %d records into %d sub-batches for parallel retry",
                           task_id, task["record_count"], len(layout))
        items += part_items(manifest, missing, attempt, pool, task.get("model_key"))
    ctx.tracker.note_retry(len(items))
    ctx.tracker.set_phase(f"retry round {attempt}/{guardrails.max_task_retries}")
    logger.warning("Retry round %d/%d: %d work item(s) for %d failed task(s): %s", attempt,
                   guardrails.max_task_retries, len(items), len(failed), errors[:5])
    update.update(retry_round=attempt, phase="generate", pending_work=items)
    return update


def _safe_manifest(paths: JobPaths, task_id: int) -> dict[str, Any] | None:
    """Load a split manifest; a corrupt one is discarded so the task is regenerated whole."""
    try:
        return load_manifest(paths, task_id)
    except RecoveryError as exc:
        logger.error("%s; discarding it", exc)
        paths.split_manifest(task_id).unlink(missing_ok=True)
        return None


# 14-15. Global duplicate detection and global guardrails --------------------------------------------
def global_guardrails_node(state: GraphState, runtime: Ctx) -> dict[str, Any]:
    """Stream all batches through the global checks."""
    ctx = _ctx(runtime)
    report = run_global_guardrails(ctx.services, state["tasks"], state["total_records"])  # type: ignore[arg-type]
    payload = report.to_dict()
    payload["_affected_full"] = {str(k): v for k, v in report.affected.items()}
    return {"global_report": payload}


# 16. Dataset quality validation ------------------------------------------------------------------------
def quality_validation_node(state: GraphState, runtime: Ctx) -> dict[str, Any]:
    """Decide whether the dataset is deliverable, repairable or failed."""
    ctx = _ctx(runtime)
    report = state["global_report"]
    guardrails = GuardrailConfig.from_requirements(state["request"].get("quality_requirements"), ctx.settings.max_retries,
                                               ctx.settings.max_invalid_fraction)
    update: dict[str, Any] = {"quality": report["quality"]}
    if report["passed"]:
        return update
    affected = report.get("_affected_full", {})
    repair_round = int(state.get("repair_round", 0))
    if (affected or report["missing_tasks"]) and repair_round < guardrails.max_repair_rounds:
        items = repair_items(state["tasks"], affected, report["missing_tasks"], repair_round + 1)
        logger.warning("Global guardrails flagged %d record(s) in %d batch(es) and %d missing batch(es); "
                       "repair round %d", sum(len(v) for v in affected.values()), len(affected),
                       len(report["missing_tasks"]), repair_round + 1)
        ctx.tracker.note_retry(len(items))  # type: ignore[union-attr]
        ctx.tracker.set_phase(f"repair round {repair_round + 1}")  # type: ignore[union-attr]
        update.update(repair_round=repair_round + 1, phase="repair", pending_work=items)
        return update
    if affected or report["missing_tasks"]:
        logger.error("Repair limit reached: records still flagged after %d repair round(s) "
                     "(max_repair_rounds / SDE_MAX_RETRIES)", repair_round)
    update.update(status="failed", errors=["Global guardrails failed: " + "; ".join(report["failures"][:10])
                                           or "unresolved records after repair rounds"])
    return update


# 17. Python data merger ---------------------------------------------------------------------------------
def merge_node(state: GraphState, runtime: Ctx) -> dict[str, Any]:
    """Merge verified batch files deterministically (no LLM involved)."""
    ctx = _ctx(runtime)
    services = ctx.services
    assert services is not None and ctx.paths is not None
    record_fields = services.schema.record_fields
    result = merge_batches(ctx.paths, state["tasks"], services.schema.columns, state["output_format"],
                           services.schema.unique_fields, record_fields, state["total_records"])
    logger.info("Merged %d batches into %s (%d records)", result.batch_count, result.output_path,
                result.record_count)
    return {"merge_result": result.to_dict(), "final_output": str(result.output_path)}


# 18. Final validation ------------------------------------------------------------------------------------
def final_validation_node(state: GraphState, runtime: Ctx) -> dict[str, Any]:
    """Assert record count, schema validation and duplicate count on the final file."""
    ctx = _ctx(runtime)
    merge = state["merge_result"]
    path = Path(merge["output_path"])
    final_count = count_records(path, state["output_format"])
    header_ok = read_header(path, state["output_format"]) == ctx.services.schema.columns  # type: ignore[union-attr]
    checks = {
        "final_record_count": final_count, "requested_record_count": state["total_records"],
        "record_count_matches": final_count == state["total_records"],
        "schema_validation_passed": bool(state["global_report"]["passed"]) and header_ok,
        "duplicate_count": int(merge["duplicate_count"]),
    }
    passed = checks["record_count_matches"] and checks["schema_validation_passed"] and checks["duplicate_count"] == 0
    checks["passed"] = passed
    return {"final_validation": checks, "status": "completed" if passed else "failed",
            **({} if passed else {"errors": ["Final validation failed"]})}


# 19. Output ----------------------------------------------------------------------------------------------
def output_node(state: GraphState, runtime: Ctx) -> dict[str, Any]:
    """Write execution and validation summaries and return the job summary."""
    ctx = _ctx(runtime)
    tracker, paths = ctx.tracker, ctx.paths
    status = state.get("status", "failed")
    if tracker is not None:
        tracker.stop_monitoring()
    execution = tracker.summary() if tracker else {}
    execution.update({
        "job_id": state.get("job_id"), "status": status, "cpu_cores": state.get("concurrency", {}).get("cpu_cores"),
        "effective_queues": state.get("concurrency", {}).get("queues"),
        "llm_concurrency": state.get("concurrency", {}).get("llm_concurrency"),
        "concurrency_reasons": state.get("concurrency", {}).get("reasons"),
        "execution_mode": state.get("execution_mode"), "core_allocation": state.get("core_allocation"),
        "queue_reports": state.get("queue_reports", []), "dispatch_rounds": state.get("dispatch_round"),
        "retry_rounds": state.get("retry_round"), "repair_rounds": state.get("repair_round"),
        "model_chain": [c["key"] if "key" in c else f"{c['provider']}/{c['model_name']}"
                        for c in state.get("model_chain", [])],
        "model_status": state.get("model_status"),
        "model_pool": state.get("model_pool"),
        "models_disabled_during_run": sorted(ctx.services.disabled_models) if ctx.services else [],
        "model_assignment": state.get("concurrency", {}).get("model_assignment"),
        "llm_slots": state.get("concurrency", {}).get("llm_slots"),
        "final_output": state.get("final_output"),
        "final_file_size_bytes": state.get("merge_result", {}).get("file_size"),
        "errors": state.get("errors", []),
        "finished_at": utc_now_iso(),
    })
    report = {k: v for k, v in state.get("global_report", {}).items() if k != "_affected_full"}
    validation = {"global_guardrails": report, "final_validation": state.get("final_validation"),
                  "quality": state.get("quality")}
    if paths is not None:
        write_json_atomic(paths.metadata / "execution_summary.json", execution)
        write_json_atomic(paths.metadata / "validation_summary.json", validation)
    level = logger.info if status == "completed" else logger.error
    level("Job %s finished with status %s in %.2fs", state.get("job_id"), status,
          execution.get("execution_time_seconds", math.nan))
    return {"summary": {"execution": execution, "validation": validation}, "status": status}
