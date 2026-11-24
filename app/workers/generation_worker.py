"""The single reusable generation task.

Every task - whatever its id - runs this same code:

1. skip if its batch is already completed and verified (idempotent recovery);
2. release stale uniqueness claims from earlier attempts;
3. build deterministic fields, then enrich LLM fields in small batches;
4. run node-level guardrails and regenerate *only* rejected records,
   up to ``max_regeneration_rounds``;
5. atomically write the batch, its semantic signatures, the rejected-record
   report and finally the task status file (with checksum).

A task never reports success unless all its records passed validation.
Repair tasks reuse the same pipeline to replace specific rows that global
guardrails flagged.
"""

from __future__ import annotations

import math
import random
import time
import traceback
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from app.core.exceptions import CancelledError, EngineError, LLMError, StorageError
from app.guardrails.node_guardrails import BatchGuardState
from app.guardrails.semantic_deduplication import record_text, save_signatures
from app.guardrails.uniqueness import owner_id
from app.models.model_router import LLMUsage
from app.monitoring.logger import get_logger, redact
from app.storage.batch_writer import (
    STATUS_CANCELLED,
    STATUS_COMPLETED,
    STATUS_FAILED,
    verified_completion,
    write_batch,
    write_rejected,
    write_task_status,
)
from app.storage.file_manager import read_json
from app.storage.formats import iter_records
from app.utils.hashing import stable_seed
from app.workers.services import WorkerServices

logger = get_logger(__name__)
_MAX_REJECTED_SAMPLES = 500
_REPAIR_ATTEMPT_BASE = 1000


@dataclass
class WorkItem:
    """A schedulable unit of work.

    Attributes:
        task_id: Task id (also the batch number).
        start_index: Zero-based global index of the first record.
        record_count: Records in the batch.
        kind: ``generate`` or ``repair``.
        attempt: Graph-level attempt number (retries / repair rounds).
        repair_rows: For repairs, ``[{"row": int, "reason": str}]`` to replace.
        model_key: Pool model this task uses (others in the pool are its fallbacks).
        part: Sub-batch number when a failed task was split (0 = the whole task).
    """

    task_id: int
    start_index: int
    record_count: int
    kind: str = "generate"
    attempt: int = 0
    repair_rows: list[dict[str, Any]] = field(default_factory=list)
    model_key: str | None = None
    part: int = 0

    @property
    def owner(self) -> int:
        """Uniqueness-claim owner id of this item."""
        return owner_id(self.task_id, self.part)

    @property
    def label(self) -> str:
        """Human-readable name for logs (``task 7`` or ``task 7 part 2``)."""
        return f"task {self.task_id}" + (f" part {self.part}" if self.part else "")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/pickle-friendly copy."""
        return asdict(self)


def _base_result(item: WorkItem) -> dict[str, Any]:
    return {"task_id": item.task_id, "part": item.part, "kind": item.kind, "attempt": item.attempt,
            "model_key": item.model_key, "status": STATUS_FAILED,
            "record_count": 0, "rejected_count": 0, "duplicate_rejections": 0, "semantic_rejections": 0,
            "regeneration_rounds": 0, "unresolved": 0, "skipped": False, "error": None,
            "llm_usage": {}, "duration": 0.0}


def _classify(reason: str, result: dict[str, Any]) -> None:
    if reason.startswith("duplicate"):
        result["duplicate_rejections"] += 1
    elif reason.startswith("semantic"):
        result["semantic_rejections"] += 1


def _generate_rows(services: WorkerServices, item: WorkItem, records: list[dict[str, Any]],
                   positions: list[int], state: BatchGuardState, result: dict[str, Any], usage: LLMUsage,
                   attempt_base: int) -> tuple[list[int], list[dict[str, Any]]]:
    """Generate, validate and regenerate ``positions`` until valid or out of rounds.

    Returns:
        Unresolved positions and the rejection log.
    """
    spec, config = services.spec, services.config
    log: list[dict[str, Any]] = []
    todo = list(positions)
    round_no = 0
    last_round = config.max_regeneration_rounds
    rescue_limit = max(config.min_rescue_records, math.ceil(config.max_invalid_fraction * len(records)))
    while todo and round_no <= last_round:
        services.cancel.raise_if_cancelled()
        attempt = attempt_base + round_no
        rescue = round_no > config.max_regeneration_rounds
        for pos in todo:
            # Seeded by the record's global index (not task id + position), so the same seed gives the same
            # dataset whatever the batch size, core count or SDE_BATCH_SIZE of the machine.
            index = item.start_index + pos
            faker = services.faker(stable_seed(services.seed_base, "faker", index, attempt, item.part))
            rng = random.Random(stable_seed(services.seed_base, "record", index, attempt, item.part))
            records[pos] = services.builder.build(index, rng, faker, attempt=round_no)
        llm_failures: dict[int, str] = {}
        if services.pool_filler is not None:  # pooled mode: recombine pooled LLM answers, no model call
            llm_failures = services.pool_filler.fill(
                records, todo, random.Random(stable_seed(services.seed_base, item.task_id, item.part, attempt, "pool")),
                widen=rescue)
        elif services.schema.llm_fields:  # jobs without LLM fields have no model chain at all
            enricher = services.enricher_for(item.model_key)
            llm_rng = random.Random(stable_seed(services.seed_base, item.task_id, item.part, attempt, "llm"))
            llm_failures = enricher.enrich(records, todo, llm_rng, usage)
        rejected = services.guards.check(item.owner, records, todo, state, preset=llm_failures)
        for pos, reason in sorted(rejected.items()):
            _classify(reason, result)
            if len(log) < _MAX_REJECTED_SAMPLES:
                log.append({"row": pos, "global_index": item.start_index + pos, "round": round_no,
                            "reason": reason, "record": records[pos]})
        result["rejected_count"] += len(rejected)
        if rejected:
            logger.debug("%s round %d: %d rejected", item.label, round_no, len(rejected))
        todo = sorted(rejected)
        if (todo and round_no == config.max_regeneration_rounds and config.rescue_rounds
                and len(todo) <= rescue_limit):
            # Only a handful left: regenerating those few is cheap, failing the task would throw away the
            # hundreds of valid records and redo them all.
            logger.info("%s: %d record(s) still invalid after %d round(s) (%s); %d rescue round(s) for "
                        "just those", item.label, len(todo), round_no + 1, rejected[todo[0]], config.rescue_rounds)
            last_round = config.max_regeneration_rounds + config.rescue_rounds
            result["rescued_rows"] = len(todo)
        round_no += 1
    result["regeneration_rounds"] += max(round_no - 1, 0)
    return todo, log


def _signatures(services: WorkerServices, records: list[dict[str, Any]], state: BatchGuardState) -> np.ndarray | None:
    if services.guards.embedder is None or not services.schema.semantic_fields:
        return None
    missing = [p for p in range(len(records)) if p not in state.vectors]
    if missing:
        vectors = services.embedder.embed([record_text(records[p], services.schema.semantic_fields) for p in missing])
        for pos, vector in zip(missing, vectors):
            state.vectors[pos] = vector
    matrix = np.stack([state.vectors[p] for p in range(len(records))])
    return services.simhasher.signatures(matrix)


def _persist(services: WorkerServices, item: WorkItem, records: list[dict[str, Any]], state: BatchGuardState,
             log: list[dict[str, Any]], result: dict[str, Any], started: float) -> dict[str, Any]:
    paths, spec = services.paths, services.spec
    batch_path, checksum = write_batch(paths, item.task_id, records, services.schema.columns, spec.output_format,
                                       item.part)
    signatures = _signatures(services, records, state)
    if signatures is not None:
        save_signatures(paths.signature_file(item.task_id, item.part), signatures)
    if item.kind == "repair":
        previous = read_json(paths.rejected_file(item.task_id), {}) or {}
        log = [*previous.get("rejected", []), *({**entry, "repair_round": item.attempt} for entry in log)]
    write_rejected(paths, item.task_id, log, unresolved=0, part=item.part)
    result.update(status=STATUS_COMPLETED, record_count=len(records), batch_file=str(batch_path),
                  checksum=checksum, warnings=len(state.warnings), duration=round(time.perf_counter() - started, 4))
    write_task_status(paths, {**result, "start_index": item.start_index})
    return result


def _release_claims(services: WorkerServices, item: WorkItem) -> None:
    """Drop the uniqueness claims a generation attempt may have left behind."""
    if item.part:
        services.registry.release_task(item.owner)
    else:
        services.registry.release_family(item.task_id)


def _fail(services: WorkerServices, item: WorkItem, result: dict[str, Any], started: float, status: str,
          error: str, log: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    result.update(status=status, error=error, duration=round(time.perf_counter() - started, 4))
    try:
        if item.kind == "generate":
            _release_claims(services, item)
        if log is not None:
            write_rejected(services.paths, item.task_id, log, unresolved=result.get("unresolved", 0), part=item.part)
        if item.kind == "generate":
            write_task_status(services.paths, {**result, "start_index": item.start_index})
    except StorageError as exc:
        # The task already failed; record the bookkeeping problem but keep the original error visible.
        logger.error("%s: could not record its failure (%s); original error: %s", item.label, exc, error)
        result["error"] = f"{error} (and failure bookkeeping failed: {exc})"
    return result


def run_task(item: WorkItem, services: WorkerServices) -> dict[str, Any]:
    """Execute one generation or repair task.

    Args:
        item: The work item.
        services: Per-process services for the job.

    Returns:
        A JSON-serialisable task result (never raises for task-level failures).
    """
    started = time.perf_counter()
    result = _base_result(item)
    usage = LLMUsage(listener=services.llm_listener)
    log: list[dict[str, Any]] = []
    try:
        if item.kind == "repair":
            return _run_repair(item, services, result, usage, started)
        verified = verified_completion(services.paths, item.task_id, item.record_count, services.spec.output_format,
                                       item.part)
        if verified is not None:
            logger.info("%s already completed and verified; skipping", item.label)
            result.update(status=STATUS_COMPLETED, record_count=item.record_count, skipped=True,
                          checksum=verified.get("checksum"))
            return result
        _release_claims(services, item)
        records: list[dict[str, Any]] = [{} for _ in range(item.record_count)]
        state = BatchGuardState()
        unresolved, log = _generate_rows(services, item, records, list(range(item.record_count)), state, result,
                                         usage, attempt_base=item.attempt * 100)
        result["llm_usage"] = usage.to_dict()
        if unresolved:
            result["unresolved"] = len(unresolved)
            reason = next((e["reason"] for e in reversed(log) if e["row"] in set(unresolved)), "see rejected report")
            rescued = " and rescue rounds" if result.get("rescued_rows") else ""
            logger.warning("%s: retry limit reached, %d record(s) still invalid after %d regeneration round(s)%s "
                           "(max_regeneration_rounds / SDE_MAX_RETRIES), e.g. %s; the task is marked failed",
                           item.label, len(unresolved), services.config.max_regeneration_rounds, rescued, reason)
            return _fail(services, item, result, started, STATUS_FAILED,
                         f"{len(unresolved)} record(s) still invalid after "
                         f"{services.config.max_regeneration_rounds} regeneration rounds{rescued} ({reason})", log)
        return _persist(services, item, records, state, log, result, started)
    except CancelledError:
        result["llm_usage"] = usage.to_dict()
        return _fail(services, item, result, started, STATUS_CANCELLED, "cancelled")
    except LLMError as exc:
        # Every model in the chain failed for this task: expected, retried by the graph.
        logger.warning("%s: all models failed: %s", item.label, redact(str(exc)))
        result["llm_usage"] = usage.to_dict()
        return _fail(services, item, result, started, STATUS_FAILED, redact(f"{type(exc).__name__}: {exc}"),
                     log or None)
    except EngineError as exc:
        logger.error("%s failed: %s", item.label, redact(str(exc)), exc_info=True)
        result["llm_usage"] = usage.to_dict()
        return _fail(services, item, result, started, STATUS_FAILED, redact(f"{type(exc).__name__}: {exc}"),
                     log or None)
    except (OSError, EOFError, ValueError, KeyError, TypeError) as exc:
        # Unexpected for a single task (likely a bug or a host problem): keep the job alive, log the stack trace.
        logger.error("%s crashed unexpectedly: %s\n%s", item.label, exc, traceback.format_exc())
        result["llm_usage"] = usage.to_dict()
        return _fail(services, item, result, started, STATUS_FAILED,
                     redact(f"unexpected {type(exc).__name__}: {exc}"), log or None)


def _run_repair(item: WorkItem, services: WorkerServices, result: dict[str, Any], usage: LLMUsage,
                started: float) -> dict[str, Any]:
    """Replace the rows global guardrails flagged in an existing batch."""
    paths, spec, guards = services.paths, services.spec, services.guards
    batch = paths.batch_file(item.task_id, spec.output_format)
    records = list(iter_records(batch, spec.output_format))
    rows = sorted({int(r["row"]) for r in item.repair_rows})
    state = BatchGuardState()
    keep = [p for p in range(len(records)) if p not in set(rows)]
    for pos in keep:
        keys = guards.keys_for(records[pos])
        state.claimed[pos] = keys
        for name, digest in keys:
            state.owners.setdefault(name, {})[digest] = pos
    if guards.embedder is not None and keep:
        vectors = services.embedder.embed([record_text(records[p], services.schema.semantic_fields) for p in keep])
        state.vectors.update(zip(keep, vectors))
    for pos in rows:
        for name, digest in guards.keys_for(records[pos]):
            services.registry.release(item.task_id, name, [digest])
    unresolved, log = _generate_rows(services, item, records, rows, state, result, usage,
                                     attempt_base=_REPAIR_ATTEMPT_BASE + item.attempt * 100)
    result["llm_usage"] = usage.to_dict()
    result["repaired_rows"] = len(rows)
    if unresolved:
        result["unresolved"] = len(unresolved)
        return _fail(services, item, result, started, STATUS_FAILED,
                     f"repair left {len(unresolved)} invalid record(s)", log)
    logger.info("Task %d: repaired %d row(s) flagged by global guardrails", item.task_id, len(rows))
    return _persist(services, item, records, state, log, result, started)
