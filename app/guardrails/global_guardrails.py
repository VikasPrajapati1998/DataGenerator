"""Global guardrails executed by the orchestrator before delivery.

The dataset is streamed batch by batch from disk - never loaded into graph
state - and checked for:

* completeness: every planned batch present, intact and correctly sized,
  total record count equal to the request;
* dataset-wide schema, business-rule and realism compliance;
* cross-node exact duplicates (unique fields and whole records) using sorted
  numpy hash arrays;
* cross-node semantic near duplicates (LSH over stored SimHash signatures);
* cross-record relationships (references must point at existing ids);
* distribution consistency and an overall quality score.

Record-level problems are returned as repair work per task (later
occurrences of a duplicate are repaired, the first one is kept).
"""

from __future__ import annotations

from array import array
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.core.exceptions import StorageError
from app.guardrails.semantic_deduplication import load_signatures, lsh_candidates
from app.guardrails.uniqueness import RECORD_KEY
from app.monitoring.logger import get_logger
from app.quality.distribution_validator import DistributionAccumulator
from app.quality.quality_scorer import QualityInputs, score_quality
from app.workers.services import WorkerServices
from app.workers.validation_worker import UniquenessIndex, validate_batch_file

logger = get_logger(__name__)
_MAX_REPORTED = 200


@dataclass
class GlobalReport:
    """Result of the global guardrail pass."""

    passed: bool
    total_records: int
    requested_records: int
    missing_tasks: list[int] = field(default_factory=list)
    affected: dict[int, list[dict[str, Any]]] = field(default_factory=dict)
    invalid_records: int = 0
    exact_duplicates: int = 0
    semantic_duplicates: int = 0
    broken_references: int = 0
    distribution_checks: list[dict[str, Any]] = field(default_factory=list)
    distribution_summary: dict[str, Any] = field(default_factory=dict)
    quality: dict[str, Any] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)

    @property
    def repairable(self) -> bool:
        """Whether every problem can be fixed by regenerating records."""
        return bool(self.affected or self.missing_tasks)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable copy (affected rows truncated)."""
        sample = {str(t): rows[:20] for t, rows in list(self.affected.items())[:_MAX_REPORTED]}
        return {"passed": self.passed, "total_records": self.total_records,
                "requested_records": self.requested_records, "missing_tasks": self.missing_tasks,
                "affected_records": sum(len(r) for r in self.affected.values()), "affected_sample": sample,
                "invalid_records": self.invalid_records, "exact_duplicates": self.exact_duplicates,
                "semantic_duplicates": self.semantic_duplicates, "broken_references": self.broken_references,
                "distribution_checks": self.distribution_checks, "distribution_summary": self.distribution_summary,
                "quality": self.quality, "failures": self.failures}


def _flag(affected: dict[int, dict[int, str]], task_id: int, row: int, reason: str) -> None:
    affected.setdefault(task_id, {}).setdefault(row, reason)


def _duplicates(index: UniquenessIndex, affected: dict[int, dict[int, str]]) -> int:
    count = 0
    for name, hashes in index.hashes.items():
        values = np.frombuffer(hashes, dtype=np.int64) if len(hashes) else np.empty(0, dtype=np.int64)
        if len(values) < 2:
            continue
        order = np.argsort(values, kind="stable")
        sorted_values = values[order]
        repeats = np.nonzero(sorted_values[1:] == sorted_values[:-1])[0] + 1
        tasks, rows = index.tasks[name], index.rows[name]
        for position in order[repeats].tolist():
            count += 1
            label = "whole record" if name == RECORD_KEY else name
            _flag(affected, int(tasks[position]), int(rows[position]),
                  f"global duplicate: {label} repeats a value from another batch")
    return count


def _references(services: WorkerServices, references: dict[str, array], ids: dict[str, array]) -> int:
    broken = 0
    for name, values in references.items():
        target = services.schema.fields[name].reference
        known = np.frombuffer(ids.get(target, array("q")), dtype=np.int64)
        refs = np.frombuffer(values, dtype=np.int64)
        valid = np.isin(refs, known)
        if services.dataset_total > services.spec.total_records:
            # Sub-job of a split run: references may point into sibling sub-jobs. Here they only need to fall
            # inside the planned ID range; the split runner checks they exist once all sub-jobs are merged.
            spec = services.schema.fields[target]  # type: ignore[index]
            low = int(spec.min) if spec.min is not None else 1
            valid |= (refs >= low) & (refs < low + services.dataset_total)
        broken += int((~valid).sum())
    return broken


def run_global_guardrails(services: WorkerServices, tasks: list[dict[str, Any]], requested: int) -> GlobalReport:
    """Run every global check over the completed batches.

    Args:
        services: Job services.
        tasks: All planned tasks.
        requested: Requested record count.

    Returns:
        The global report (with repair work in ``affected``).
    """
    config = services.config
    accumulator = DistributionAccumulator(services.schema)
    index = UniquenessIndex()
    references: dict[str, array] = {}
    ids: dict[str, array] = {}
    affected: dict[int, dict[int, str]] = {}
    report = GlobalReport(passed=False, total_records=0, requested_records=requested)

    for task in tasks:
        check = validate_batch_file(services, task, accumulator, index, references, ids)
        if not check.present:
            report.missing_tasks.append(check.task_id)
            continue
        report.total_records += check.rows
        if not check.header_ok:
            report.failures.append(f"batch {check.task_id}: column order differs from schema")
        if not check.count_ok:
            report.failures.append(f"batch {check.task_id}: {check.rows} records, expected {task['record_count']}")
        for row, reason in check.invalid:
            _flag(affected, check.task_id, row, reason)
    report.invalid_records = sum(len(r) for r in affected.values())
    report.exact_duplicates = _duplicates(index, affected)
    report.broken_references = _references(services, references, ids)
    if report.broken_references:
        report.failures.append(f"{report.broken_references} reference value(s) point to missing records")

    if config.semantic_dedup and services.schema.semantic_fields and services.pool_filler is None:
        # (Pooled texts are reused on purpose and have no signatures; see WorkerServices.use_pool.)
        signature_sets = []
        for task in tasks:
            path = services.paths.signature_file(int(task["task_id"]))
            try:
                signature_sets.append((int(task["task_id"]), load_signatures(path)))
            except StorageError as exc:
                # Exact checks still cover this batch; only cross-batch near-duplicate search loses it.
                logger.warning("Batch %s skipped by global semantic dedup: %s", task["task_id"], exc)
        for _, (task_id, row) in lsh_candidates(signature_sets):
            report.semantic_duplicates += 1
            _flag(affected, task_id, row, "global semantic duplicate: near-identical text in another batch")

    checks = accumulator.evaluate(config)
    report.distribution_checks = [c.to_dict() for c in checks]
    report.distribution_summary = accumulator.summary()
    failed_checks = [c for c in checks if not c.passed]
    scored = checks
    if services.pool_filler is not None:
        # Pooled mode reuses a limited set of answers by design: text diversity is reported (and logged when
        # low), but it neither blocks delivery nor counts in the quality score.
        scored = [c for c in checks if c.check != "distinct_text_ratio"]
        for check in failed_checks:
            if check.check == "distinct_text_ratio":
                logger.warning("Pooled mode: %s has %.1f%% distinct texts (below %s); a larger pool "
                               "(SDE_POOL_MAX_VALUES) gives more variety", check.field, float(check.observed) * 100,
                               check.expected)
    scored_failures = [c for c in scored if not c.passed]
    if scored_failures and config.enforce_distribution:
        report.failures.extend(f"distribution: {c.field} {c.check} expected {c.expected}, observed {c.observed}"
                               for c in scored_failures)
    distinct = [c.observed for c in scored if c.check == "distinct_text_ratio"]
    flagged = sum(len(r) for r in affected.values())
    report.quality = score_quality(QualityInputs(
        total_records=report.total_records, requested_records=requested,
        invalid_records=report.invalid_records, duplicate_records=report.exact_duplicates + report.semantic_duplicates,
        distribution_checks_passed=len(scored) - len(scored_failures), distribution_checks_total=len(scored),
        distinct_text_ratios=[float(d) for d in distinct]))
    if report.total_records + sum(int(t["record_count"]) for t in tasks if int(t["task_id"]) in
                                  set(report.missing_tasks)) != requested:
        report.failures.append(f"planned records do not add up to the requested {requested}")
    if report.quality["score"] < config.min_quality_score:
        report.failures.append(f"quality score {report.quality['score']} below {config.min_quality_score}")
    report.affected = {task_id: [{"row": row, "reason": reason} for row, reason in sorted(rows.items())]
                       for task_id, rows in sorted(affected.items())}
    report.passed = not report.failures and not flagged and not report.missing_tasks \
        and report.total_records == requested
    logger.info("Global guardrails: %s | records %d/%d | invalid %d | duplicates %d exact, %d semantic | "
                "broken refs %d | quality %.1f", "PASSED" if report.passed else "FAILED", report.total_records,
                requested, report.invalid_records, report.exact_duplicates, report.semantic_duplicates,
                report.broken_references, report.quality["score"])
    return report
