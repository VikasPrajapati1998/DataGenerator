"""Streaming distribution statistics and statistical consistency checks.

Statistics are accumulated record by record (constant memory), so the whole
dataset never needs to be loaded. Checks compare observed proportions and
moments with what the schema specifies, using standard-error based
tolerances so small datasets are not failed for normal sampling noise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from app.config.guardrail_config import GuardrailConfig
from app.schemas.input_schema import DatasetSchema


@dataclass
class _Numeric:
    count: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf

    def add(self, value: float) -> None:
        self.count += 1
        self.total += value
        self.total_sq += value * value
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else 0.0

    @property
    def std(self) -> float:
        if self.count < 2:
            return 0.0
        return math.sqrt(max(self.total_sq / self.count - self.mean ** 2, 0.0))


@dataclass
class DistributionCheck:
    """Outcome of one statistical check."""

    field: str
    check: str
    expected: Any
    observed: Any
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable copy."""
        return {"field": self.field, "check": self.check, "expected": self.expected,
                "observed": self.observed, "passed": self.passed}


@dataclass
class DistributionAccumulator:
    """Accumulates category counts, numeric moments and null rates.

    Args:
        schema: The normalised schema.
    """

    schema: DatasetSchema
    total: int = 0
    categories: dict[str, dict[str, int]] = field(default_factory=dict)
    numerics: dict[str, _Numeric] = field(default_factory=dict)
    nulls: dict[str, int] = field(default_factory=dict)
    distinct_text: dict[str, set[int]] = field(default_factory=dict)

    def add(self, record: dict[str, Any]) -> None:
        """Add one record (Python-typed values) to the statistics."""
        self.total += 1
        for name, spec in self.schema.fields.items():
            value = record.get(name)
            if value is None or value == "":
                self.nulls[name] = self.nulls.get(name, 0) + 1
                continue
            if spec.allowed_values:
                bucket = self.categories.setdefault(name, {})
                bucket[str(value)] = bucket.get(str(value), 0) + 1
            elif spec.type in ("integer", "float") and spec.generator not in ("sequence", "reference"):
                self.numerics.setdefault(name, _Numeric()).add(float(value))
            elif spec.generator == "llm" and spec.type == "string":
                seen = self.distinct_text.setdefault(name, set())
                if len(seen) < 2_000_000:
                    seen.add(hash(str(value).casefold()))

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serialisable description of the observed distributions."""
        return {
            "records": self.total,
            "categories": {n: {k: round(v / max(self.total, 1), 4) for k, v in sorted(c.items())}
                           for n, c in self.categories.items()},
            "numerics": {n: {"mean": round(s.mean, 3), "std": round(s.std, 3), "min": s.minimum, "max": s.maximum}
                         for n, s in self.numerics.items()},
            "null_rate": {n: round(c / max(self.total, 1), 4) for n, c in self.nulls.items()},
            "distinct_text_ratio": {n: round(len(s) / max(self.total, 1), 4) for n, s in self.distinct_text.items()},
        }

    def evaluate(self, config: GuardrailConfig) -> list[DistributionCheck]:  # noqa: PLR0912
        """Compare observed statistics with the schema's expectations.

        Args:
            config: Guardrail thresholds (sigma and absolute tolerance).

        Returns:
            All checks performed.
        """
        checks: list[DistributionCheck] = []
        n = max(self.total, 1)
        for name, spec in self.schema.fields.items():
            if spec.cases is not None or spec.counts is not None:
                continue  # per-case settings or exact counts: the field-level expectations do not apply
            if spec.allowed_values and spec.weights and name in self.categories:
                weight_sum = sum(spec.weights)
                non_null = n - self.nulls.get(name, 0)
                for value, weight in zip(spec.allowed_values, spec.weights):
                    expected = weight / weight_sum
                    observed = self.categories[name].get(str(value), 0) / max(non_null, 1)
                    tolerance = max(config.distribution_abs_tolerance,
                                    config.distribution_sigma * math.sqrt(expected * (1 - expected) / max(non_null, 1)))
                    checks.append(DistributionCheck(name, f"proportion[{value}]", round(expected, 4),
                                                    round(observed, 4), abs(observed - expected) <= tolerance))
            elif spec.allowed_values and name in self.categories and spec.generator == "enum":
                missing = [str(v) for v in spec.allowed_values if str(v) not in self.categories[name]]
                expected_min = n / len(spec.allowed_values)
                checks.append(DistributionCheck(name, "all_categories_present", True, not missing,
                                                not missing or expected_min < 5))
            if spec.null_ratio > 0:
                observed = self.nulls.get(name, 0) / n
                tolerance = max(config.distribution_abs_tolerance,
                                config.distribution_sigma * math.sqrt(spec.null_ratio * (1 - spec.null_ratio) / n))
                checks.append(DistributionCheck(name, "null_ratio", spec.null_ratio, round(observed, 4),
                                                abs(observed - spec.null_ratio) <= tolerance))
            stats = self.numerics.get(name)
            dist = spec.distribution
            if stats and stats.count > 1 and dist and dist.kind == "normal" and dist.mean is not None and dist.std:
                clipped = spec.min is not None or spec.max is not None
                tolerance = config.distribution_sigma * dist.std / math.sqrt(stats.count) + (
                    0.1 * dist.std if clipped else 0)
                checks.append(DistributionCheck(name, "mean", dist.mean, round(stats.mean, 3),
                                                abs(stats.mean - dist.mean) <= tolerance))
            if stats and spec.min is not None and spec.max is not None and spec.type in ("integer", "float"):
                checks.append(DistributionCheck(name, "within_bounds", [spec.min, spec.max],
                                                [stats.minimum, stats.maximum],
                                                float(spec.min) <= stats.minimum and stats.maximum <= float(spec.max)))
            if stats and stats.count >= 30 and spec.generator in ("range", "profile"):
                checks.append(DistributionCheck(name, "non_degenerate", "std > 0", round(stats.std, 3), stats.std > 0))
        for name, seen in self.distinct_text.items():
            ratio = len(seen) / max(n - self.nulls.get(name, 0), 1)
            checks.append(DistributionCheck(name, "distinct_text_ratio", f">= {config.min_distinct_ratio}",
                                            round(ratio, 4), ratio >= config.min_distinct_ratio))
        return checks
