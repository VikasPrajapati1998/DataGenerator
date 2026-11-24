"""Realism validation, distribution consistency and quality scoring."""

from __future__ import annotations

import random

from app.config.guardrail_config import GuardrailConfig
from app.generation.record_builder import RecordBuilder
from app.quality.distribution_validator import DistributionAccumulator
from app.quality.quality_scorer import QualityInputs, score_quality
from app.quality.realism_validator import RealismValidator
from app.schemas.schema_parser import parse_schema
from tests.conftest import EMPLOYEE_SCHEMA
from tests.test_generation import PROFILES, _seeded

WEIGHTED = {"fields": {
    "id": {"type": "integer", "unique": True},
    "plan": {"type": "string", "allowed_values": ["free", "pro", "team"], "weights": [70, 20, 10]},
    "score": {"type": "float", "min": 0, "max": 100, "distribution": {"kind": "normal", "mean": 60, "std": 10}},
    "referrer": {"type": "string", "faker_method": "user_name", "nullable": True, "null_ratio": 0.2},
}}


def _accumulate(schema, records) -> DistributionAccumulator:
    acc = DistributionAccumulator(schema)
    for record in records:
        acc.add(record)
    return acc


def test_generated_distributions_match_the_schema() -> None:
    schema = parse_schema(WEIGHTED)
    builder = RecordBuilder(schema, 20_000)
    faker = _seeded()
    records = [builder.build(i, random.Random(i), faker) for i in range(20_000)]
    checks = _accumulate(schema, records).evaluate(GuardrailConfig())
    failed = [c.to_dict() for c in checks if not c.passed]
    assert failed == []
    assert {c.check for c in checks} >= {"proportion[free]", "mean", "null_ratio", "within_bounds"}


def test_skewed_distribution_is_detected() -> None:
    schema = parse_schema(WEIGHTED)
    records = [{"id": i, "plan": "team", "score": 50.0 + i % 20, "referrer": "x"} for i in range(2000)]
    checks = _accumulate(schema, records).evaluate(GuardrailConfig())
    failing = {c.check for c in checks if not c.passed}
    assert "proportion[free]" in failing and "null_ratio" in failing and "non_degenerate" not in failing


def test_degenerate_numeric_values_are_flagged() -> None:
    schema = parse_schema({"id": "unique integer", "age": "integer between 22 and 60"})
    records = [{"id": i, "age": 30} for i in range(100)]
    checks = _accumulate(schema, records).evaluate(GuardrailConfig())
    assert any(c.check == "non_degenerate" and not c.passed for c in checks)


def test_realism_flags_geo_and_profile_violations() -> None:
    schema = parse_schema({"id": "unique integer", "city": "Indian city", "state": "Indian state",
                           "department": ["IT", "HR", "Finance", "Sales"],
                           "salary": "realistic salary based on department"})
    realism = RealismValidator(schema, PROFILES)
    assert realism.check({"id": 1, "city": "Pune", "state": "Maharashtra", "department": "IT",
                          "salary": 700_000}) == []
    issues = realism.check({"id": 2, "city": "Pune", "state": "Kerala", "department": "IT", "salary": 10})
    assert len(issues) == 2


def test_realism_checks_age_against_birth_date() -> None:
    schema = parse_schema({"fields": {"age": {"type": "integer", "min": 0, "max": 120},
                                      "birth_date": {"type": "date", "min": "1900-01-01"}}})
    realism = RealismValidator(schema)
    assert realism.check({"age": 5, "birth_date": "1950-01-01"})


def test_quality_score_components() -> None:
    perfect = score_quality(QualityInputs(1000, 1000, 0, 0, 10, 10, [0.99]))
    assert perfect["score"] > 99
    poor = score_quality(QualityInputs(1000, 1000, 200, 100, 5, 10, [0.3]))
    assert poor["score"] < 75
    assert set(poor["components"]) == {"validity", "uniqueness", "distribution", "completeness", "diversity"}


def test_employee_dataset_scores_high() -> None:
    schema = parse_schema(EMPLOYEE_SCHEMA, "Indian employees")
    builder = RecordBuilder(schema, 5000, PROFILES)
    faker = _seeded()
    acc = _accumulate(schema, [builder.build(i, random.Random(i), faker) for i in range(5000)])
    checks = acc.evaluate(GuardrailConfig())
    assert all(c.passed for c in checks)
    summary = acc.summary()
    assert set(summary["categories"]["department"]) == {"IT", "HR", "Finance", "Sales"}
