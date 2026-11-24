"""Node-level guardrails applied by every generation task to its own batch.

For the records that were (re)generated in the current round, the guardrail
pipeline runs, in order:

1. schema + business rules (Pydantic model and safe rule engine);
2. realism heuristics;
3. in-batch exact uniqueness (unique fields and whole-record hash);
4. cross-task uniqueness through the shared :class:`UniquenessRegistry`;
5. semantic near-duplicate detection and free-text repetition limits.

Rejected positions are returned with reasons so the worker regenerates only
those records. State carries over between rounds, so accepted records are
never re-checked or regenerated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.config.guardrail_config import GuardrailConfig
from app.guardrails.semantic_deduplication import Embedder, find_near_duplicates, record_text
from app.guardrails.uniqueness import RECORD_KEY, UniquenessRegistry
from app.quality.realism_validator import RealismValidator
from app.schemas.input_schema import DatasetSchema
from app.schemas.validation_schema import RecordValidator
from app.utils.hashing import record_hash, value_hash


@dataclass
class BatchGuardState:
    """Per-task state that persists across regeneration rounds.

    Attributes:
        owners: ``field -> hash -> position`` of accepted values.
        claimed: ``position -> [(field, hash)]`` claims held in the registry.
        vectors: ``position -> embedding`` of accepted records.
        text_counts: ``field -> normalised text -> count`` for repetition limits.
        warnings: Non-blocking rule warnings.
    """

    owners: dict[str, dict[int, int]] = field(default_factory=dict)
    claimed: dict[int, list[tuple[str, int]]] = field(default_factory=dict)
    vectors: dict[int, np.ndarray] = field(default_factory=dict)
    text_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


class NodeGuardrails:
    """Validates a task's records and decides which ones must be regenerated.

    Args:
        schema: The normalised schema.
        config: Guardrail thresholds.
        validator: Schema/rule validator.
        realism: Realism validator.
        registry: Shared uniqueness registry (``None`` disables cross-task claims).
        embedder: Embedder for semantic checks (``None`` disables them).
    """

    def __init__(self, schema: DatasetSchema, config: GuardrailConfig, validator: RecordValidator,
                 realism: RealismValidator, registry: UniquenessRegistry | None, embedder: Embedder | None) -> None:
        self.schema = schema
        self.config = config
        self.validator = validator
        self.realism = realism
        self.registry = registry
        self.embedder = embedder if config.semantic_dedup else None
        self.unique_fields = schema.unique_fields
        self.record_fields = schema.record_fields
        self.text_fields = [n for n in schema.llm_fields if schema.fields[n].type == "string"]

    def keys_for(self, record: dict[str, Any]) -> list[tuple[str, int]]:
        """Return the ``(field, hash)`` uniqueness keys of ``record``."""
        keys = [(name, value_hash(record[name])) for name in self.unique_fields if record.get(name) is not None]
        if self.record_fields:
            keys.append((RECORD_KEY, record_hash(record, self.record_fields)))
        return keys

    def check(self, task_id: int, records: list[dict[str, Any]], positions: list[int],
              state: BatchGuardState, preset: dict[int, str] | None = None) -> dict[int, str]:
        """Validate ``records[p]`` for every ``p`` in ``positions``.

        Args:
            task_id: Owning task (for registry claims).
            records: The whole batch.
            positions: Positions generated in this round.
            state: State carried across rounds (updated in place).
            preset: Positions already rejected upstream (e.g. LLM failures).

        Returns:
            ``position -> reason`` for every rejected record.
        """
        rejected: dict[int, str] = dict(preset or {})
        candidates: list[int] = []
        for pos in positions:
            if pos in rejected:
                continue
            errors = self.validator.validate(records[pos])
            blocking = [e for e in errors if not e.startswith("warning:")]
            state.warnings.extend(e for e in errors if e.startswith("warning:"))
            if blocking:
                rejected[pos] = blocking[0]
                continue
            issues = self.realism.check(records[pos])
            if issues:
                rejected[pos] = issues[0]
                continue
            candidates.append(pos)

        candidates = self._local_uniqueness(records, candidates, state, rejected)
        candidates = self._registry_claims(task_id, records, candidates, state, rejected)
        candidates = self._semantic(records, candidates, state, rejected, task_id)
        self._accept(records, candidates, state)
        return rejected

    def _local_uniqueness(self, records: list[dict[str, Any]], candidates: list[int], state: BatchGuardState,
                          rejected: dict[int, str]) -> list[int]:
        kept: list[int] = []
        pending: dict[str, dict[int, int]] = {}
        for pos in candidates:
            clash = None
            keys = self.keys_for(records[pos])
            for name, digest in keys:
                if digest in state.owners.get(name, {}) or digest in pending.get(name, {}):
                    clash = name
                    break
            if clash:
                rejected[pos] = ("duplicate: identical record within batch" if clash == RECORD_KEY
                                 else f"duplicate: {clash} already used within batch")
                continue
            for name, digest in keys:
                pending.setdefault(name, {})[digest] = pos
            kept.append(pos)
        return kept

    def _registry_claims(self, task_id: int, records: list[dict[str, Any]], candidates: list[int],
                         state: BatchGuardState, rejected: dict[int, str]) -> list[int]:
        if self.registry is None or not candidates:
            return candidates
        by_field: dict[str, list[tuple[int, int]]] = {}
        for pos in candidates:
            for name, digest in self.keys_for(records[pos]):
                by_field.setdefault(name, []).append((pos, digest))
        losers: dict[int, str] = {}
        granted: dict[int, list[tuple[str, int]]] = {}
        for name, pairs in by_field.items():
            conflicts = self.registry.claim(task_id, name, [d for _, d in pairs])
            for pos, digest in pairs:
                if digest in conflicts:
                    losers.setdefault(pos, "duplicate: identical record exists in another task" if name == RECORD_KEY
                                      else f"duplicate: {name} already used by another task")
                else:
                    granted.setdefault(pos, []).append((name, digest))
        for pos, reason in losers.items():
            rejected[pos] = reason
            for name, digest in granted.pop(pos, []):
                self.registry.release(task_id, name, [digest])
        for pos, keys in granted.items():
            state.claimed[pos] = keys
        return [p for p in candidates if p not in losers]

    def release(self, task_id: int, positions: list[int], state: BatchGuardState) -> None:
        """Release registry claims held for ``positions`` (records being replaced)."""
        for pos in positions:
            keys = state.claimed.pop(pos, [])
            if self.registry is not None:
                for name, digest in keys:
                    self.registry.release(task_id, name, [digest])
            for name, digest in keys:
                state.owners.get(name, {}).pop(digest, None)
            state.vectors.pop(pos, None)

    def _semantic(self, records: list[dict[str, Any]], candidates: list[int], state: BatchGuardState,
                  rejected: dict[int, str], task_id: int) -> list[int]:
        dropped: dict[int, str] = {}
        for name in self.text_fields:
            counts = state.text_counts.setdefault(name, {})
            limit = max(2, int(self.config.max_value_repeat_ratio * len(records)))
            local: dict[str, int] = {}
            for pos in candidates:
                text = " ".join(str(records[pos].get(name) or "").casefold().split())
                if not text:
                    continue
                if counts.get(text, 0) + local.get(text, 0) >= limit:
                    dropped.setdefault(pos, f"semantic: {name} text repeated too often")
                else:
                    local[text] = local.get(text, 0) + 1
        remaining = [p for p in candidates if p not in dropped]
        if self.embedder is not None and remaining and self.schema.semantic_fields:
            texts = [record_text(records[p], self.schema.semantic_fields) for p in remaining]
            new_vectors = self.embedder.embed(texts)
            accepted_positions = list(state.vectors)
            if accepted_positions:
                base = np.stack([state.vectors[p] for p in accepted_positions])
                matrix = np.vstack([base, new_vectors])
            else:
                matrix = new_vectors
            offset = len(accepted_positions)
            for i, j, sim in find_near_duplicates(matrix, self.config.semantic_threshold):
                if j >= offset:
                    pos = remaining[j - offset]
                    dropped.setdefault(pos, f"semantic: near duplicate of another record (cosine {sim:.3f})")
            for pos, vector in zip(remaining, new_vectors):
                if pos not in dropped:
                    state.vectors[pos] = vector
        for pos, reason in dropped.items():
            rejected[pos] = reason
        self.release(task_id, list(dropped), state)
        return [p for p in candidates if p not in dropped]

    def _accept(self, records: list[dict[str, Any]], positions: list[int], state: BatchGuardState) -> None:
        for pos in positions:
            for name, digest in self.keys_for(records[pos]):
                state.owners.setdefault(name, {})[digest] = pos
            for name in self.text_fields:
                text = " ".join(str(records[pos].get(name) or "").casefold().split())
                if text:
                    counts = state.text_counts.setdefault(name, {})
                    counts[text] = counts.get(text, 0) + 1
