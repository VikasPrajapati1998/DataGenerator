"""Overall dataset quality score (0-100).

The score combines independent dimensions with fixed weights:

* validity (35)     - records passing schema, rule and realism checks
* uniqueness (25)   - absence of exact and semantic duplicates
* distribution (20) - statistical checks passed
* completeness (10) - delivered vs requested records
* diversity (10)    - distinct ratio of free-text fields (1.0 when none)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

WEIGHTS = {"validity": 35.0, "uniqueness": 25.0, "distribution": 20.0, "completeness": 10.0, "diversity": 10.0}


@dataclass
class QualityInputs:
    """Raw counts the score is computed from."""

    total_records: int
    requested_records: int
    invalid_records: int
    duplicate_records: int
    distribution_checks_passed: int
    distribution_checks_total: int
    distinct_text_ratios: list[float]


def score_quality(inputs: QualityInputs) -> dict[str, Any]:
    """Compute the weighted quality score and its components.

    Args:
        inputs: Aggregated validation counts.

    Returns:
        ``{"score": float, "components": {name: 0..1}}``.
    """
    total = max(inputs.total_records, 1)
    components = {
        "validity": 1.0 - inputs.invalid_records / total,
        "uniqueness": 1.0 - inputs.duplicate_records / total,
        "distribution": (inputs.distribution_checks_passed / inputs.distribution_checks_total
                         if inputs.distribution_checks_total else 1.0),
        "completeness": min(inputs.total_records / max(inputs.requested_records, 1), 1.0),
        "diversity": (sum(min(r, 1.0) for r in inputs.distinct_text_ratios) / len(inputs.distinct_text_ratios)
                      if inputs.distinct_text_ratios else 1.0),
    }
    components = {k: max(0.0, min(1.0, v)) for k, v in components.items()}
    score = sum(WEIGHTS[k] * v for k, v in components.items())
    return {"score": round(score, 2), "components": {k: round(v, 4) for k, v in components.items()}}
