"""Streaming re-validation of one completed batch file for the global stage.

Each record is parsed back into typed values through the dynamic Pydantic
model, re-checked against rules and realism heuristics, fed into the
distribution accumulator and its uniqueness hashes are appended to compact
numpy-friendly arrays (so 1M+ records do not need per-record dict entries).
"""

from __future__ import annotations

from array import array
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError as PydanticValidationError

from app.quality.distribution_validator import DistributionAccumulator
from app.storage.batch_writer import verified_completion
from app.storage.formats import iter_records, read_header
from app.workers.services import WorkerServices


@dataclass
class UniquenessIndex:
    """Compact columnar index of uniqueness hashes and their locations."""

    hashes: dict[str, array] = field(default_factory=dict)
    tasks: dict[str, array] = field(default_factory=dict)
    rows: dict[str, array] = field(default_factory=dict)

    def add(self, name: str, digest: int, task_id: int, row: int) -> None:
        """Append one hash occurrence."""
        if name not in self.hashes:
            self.hashes[name], self.tasks[name], self.rows[name] = array("q"), array("l"), array("l")
        self.hashes[name].append(digest)
        self.tasks[name].append(task_id)
        self.rows[name].append(row)


@dataclass
class BatchCheck:
    """Global-stage findings for one batch."""

    task_id: int
    present: bool
    rows: int = 0
    header_ok: bool = True
    count_ok: bool = True
    invalid: list[tuple[int, str]] = field(default_factory=list)


def validate_batch_file(services: WorkerServices, task: dict[str, Any], accumulator: DistributionAccumulator,
                        index: UniquenessIndex, references: dict[str, array], ids: dict[str, array]) -> BatchCheck:
    """Re-validate one batch and index its records for cross-batch checks.

    Args:
        services: Job services (schema, validators, guardrails).
        task: Planned task (``task_id``, ``record_count``).
        accumulator: Distribution statistics (updated in place).
        index: Uniqueness index (updated in place).
        references: ``reference field -> referenced ids`` (updated in place).
        ids: ``sequence field -> ids`` (updated in place).

    Returns:
        Findings for the batch.
    """
    task_id, expected = int(task["task_id"]), int(task["record_count"])
    fmt = services.spec.output_format
    paths, schema, guards = services.paths, services.schema, services.guards
    if verified_completion(paths, task_id, expected, fmt) is None:
        return BatchCheck(task_id=task_id, present=False)
    batch = paths.batch_file(task_id, fmt)
    check = BatchCheck(task_id=task_id, present=True, header_ok=read_header(batch, fmt) == schema.columns)
    revalidate = services.config.global_schema_revalidation
    ref_fields = {n: s.reference for n, s in schema.fields.items() if s.generator == "reference"}
    seq_fields = [n for n, s in schema.fields.items() if s.generator == "sequence"]
    for row, raw in enumerate(iter_records(batch, fmt)):
        check.rows += 1
        try:
            typed = services.validator.model.model_validate(raw).model_dump()
        except PydanticValidationError as exc:
            err = exc.errors()[0]
            check.invalid.append((row, f"schema: {'.'.join(str(p) for p in err['loc'])}: {err['msg']}"))
            continue
        if revalidate:
            errors = [e for e in services.validator.rule_errors(typed) if not e.startswith("warning:")]
            errors += services.realism.check(typed)
            if errors:
                check.invalid.append((row, errors[0]))
        accumulator.add(typed)
        for name, digest in guards.keys_for(typed):
            index.add(name, digest, task_id, row)
        for name in ref_fields:
            if typed.get(name) is not None:
                references.setdefault(name, array("q")).append(int(typed[name]))
        for name in seq_fields:
            ids.setdefault(name, array("q")).append(int(typed[name]))
    check.count_ok = check.rows == expected
    return check
