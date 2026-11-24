"""Pooled LLM mode: templates, valid-by-construction texts, pool growth and reuse, rescue rounds."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from types import SimpleNamespace

from faker import Faker

from app.config.guardrail_config import GuardrailConfig
from app.config.model_registry import ModelConfig, ModelRegistry
from app.config.settings import Settings
from app.core.cancellation import CancellationToken
from app.generation.llm_enricher import LLMEnricher
from app.generation.llm_pool import (
    PooledFiller,
    PoolGrower,
    _indian,
    build_value_pool,
    relabel,
    render,
    save_pool,
    sentences,
    templatize,
)
from app.generation.record_builder import RecordBuilder
from app.graph.nodes import _plan_from_pool
from app.guardrails.node_guardrails import BatchGuardState
from app.guardrails.prompt_safety import check_generated_text
from app.models.mock_adapter import MockBehavior, MockLLMAdapter
from app.models.model_router import LLMUsage, ModelRouter
from app.monitoring.execution_tracker import ExecutionTracker
from app.schemas.schema_parser import parse_schema
from app.storage.formats import iter_records
from app.workers.generation_worker import WorkItem, _base_result, _generate_rows

CLAIM_SCHEMA = {
    "claim_id": "unique integer",
    "insured_name": "realistic full name",
    "policy_type": ["health", "motor", "life"],
    "city": "Indian city",
    "claim_description": "one or two sentences describing the incident behind the claim",
}
BIO_SCHEMA = {
    "customer_id": "unique integer",
    "name": "realistic full name",
    "segment": ["retail", "business", "enterprise"],
    "bio": "two-sentence profile of the customer's buying habits",
}


def _engine(tmp_path: Path, registry: ModelRegistry, **overrides):
    from app.engine import SyntheticDataEngine  # pylint: disable=import-outside-toplevel

    settings = Settings(output_dir=tmp_path / "output", log_dir=tmp_path / "logs", default_model="mock/mock-llm",
                        **overrides)
    return SyntheticDataEngine(settings=settings, registry=registry)


def _request(schema: dict, records: int, **extra) -> dict:
    return {"schema": dict(schema), "total_records": records, "output_format": "json", "model_provider": "mock",
            "model_name": "mock-llm", "batch_size": 250, "execution_mode": "thread", "seed": 5, **extra}


# -- templates --------------------------------------------------------------------------------------
def test_indian_digit_grouping() -> None:
    assert _indian(999) == "999"
    assert _indian(192100) == "1,92,100"
    assert _indian(12345678) == "1,23,45,678"


def test_template_takes_the_values_of_the_record_it_fills() -> None:
    source = {"insured_name": "Raghav Cheema", "claim_amount_inr": 192100, "incident_date": "2025-06-14",
              "city": "Kochi"}
    text = "Raghav Cheema claimed $192,100 after a fall in Kochi on June 14, 2025."
    template = templatize(text, source, list(source))
    assert "Raghav" not in template and "192" not in template and "Kochi" not in template
    assert "{insured_name}" in template and "{claim_amount_inr:comma}" in template
    assert "{incident_date:long}" in template and "{city}" in template
    other = {"insured_name": "Meera Iyer", "claim_amount_inr": 250000, "incident_date": "2024-01-05",
             "city": "Pune"}
    assert render(template, other) == "Meera Iyer claimed $250,000 after a fall in Pune on January 5, 2024."
    assert render(template, {"insured_name": "x"}) is None  # a needed value is missing


def test_sentences_ignore_abbreviation_dots() -> None:
    text = "A claim of Rs. 5,000 was filed by Mr. Rao. Dr. Shah approved it on 2 May."
    assert sentences(text) == ["A claim of Rs. 5,000 was filed by Mr. Rao.", "Dr. Shah approved it on 2 May."]


def test_answer_is_filed_under_the_category_it_names() -> None:
    schema = parse_schema(CLAIM_SCHEMA)
    assert relabel({"policy_type": "motor"}, "A health insurance claim after a fever.", schema) == {
        "policy_type": "health"}
    assert relabel({"policy_type": "motor"}, "A claim after a road accident.", schema) == {"policy_type": "motor"}


# -- filling ----------------------------------------------------------------------------------------
def _pool_file(tmp_path: Path, schema, templates: list[tuple[str, str]]) -> Path:
    path = tmp_path / "llm_pool.json"
    items = [{"context": {"policy_type": kind}, "values": {"claim_description": text}} for kind, text in templates]
    save_pool(path, schema, items, plan={"mode": "pooled", "models": ["mock/mock-llm"], "dropped": {}})
    return path


def test_filled_texts_are_valid_and_name_the_record(tmp_path) -> None:
    schema = parse_schema(CLAIM_SCHEMA)
    path = _pool_file(tmp_path, schema, [
        ("health", "{insured_name} was admitted to a hospital in {city} for three days. The bills were submitted."),
        ("health", "{insured_name} needed surgery after a fall at home. Rs. 40,000 was claimed for it."),
        ("motor", "{insured_name} hit a divider near {city} at night. The bumper and lights were damaged."),
        ("motor", "A truck rear-ended the car of {insured_name}. " + "Very long repair notes. " * 30),
    ])
    filler = PooledFiller(schema, path, max_text_length=240)
    rng = random.Random(3)
    records = [{"claim_id": i, "insured_name": f"Person {i}", "policy_type": "motor" if i % 2 else "health",
                "city": "Pune"} for i in range(300)]
    assert filler.fill(records, list(range(300)), rng) == {}
    for record in records:
        text = record["claim_description"]
        assert check_generated_text(text, 240) is None and len(text.split()) >= 3
        assert record["insured_name"] in text and "{" not in text
    assert len({r["claim_description"] for r in records}) == 300


# -- reuse and growth -------------------------------------------------------------------------------
def test_existing_pool_plan_is_reused(tmp_path) -> None:
    schema = parse_schema(CLAIM_SCHEMA)
    path = _pool_file(tmp_path, schema, [("health", "{insured_name} was treated in {city} for a fever.")])
    plan = _plan_from_pool(path, ["mock/a", "mock/b"], 1_000_000)
    assert plan["mode"] == "pooled" and plan["models"] == ["mock/mock-llm"] and plan["reused_pool"]
    old = tmp_path / "old_pool.json"
    old.write_text(json.dumps({"fields": ["claim_description"], "context_fields": ["policy_type"],
                               "items": [{"context": {"policy_type": "life"},
                                          "values": {"claim_description": "A life claim was filed."}}]}),
                   encoding="utf-8")
    legacy = _plan_from_pool(old, ["mock/a", "mock/b"], 1000)
    assert legacy["models"] == ["mock/a"]  # no stored plan: grow with one model only
    assert _plan_from_pool(tmp_path / "missing.json", ["mock/a"], 10) is None


def test_pool_grows_in_the_background_and_is_saved(tmp_path) -> None:
    schema = parse_schema(BIO_SCHEMA)
    builder = RecordBuilder(schema, 1000)
    model = ModelConfig(provider="mock", model_name="mock-llm", model_id="mock-llm",
                        retry={"max_attempts": 1, "base_delay": 0.0, "max_delay": 0.0})
    enricher = LLMEnricher(schema, ModelRouter([model], Settings()), GuardrailConfig(), None)
    items = build_value_pool(schema, builder, [enricher], 10, 60, 1, LLMUsage())
    path = tmp_path / "llm_pool.json"
    save_pool(path, schema, items)
    filler = PooledFiller(schema, path, 600)
    grower = PoolGrower(schema, builder, [enricher], filler, path, 40, 1, LLMUsage())
    grower.start()
    deadline = time.monotonic() + 20
    while filler.size < 40 and time.monotonic() < deadline:
        time.sleep(0.05)
    added = grower.stop()
    assert added > 0 and filler.size >= 40
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert len(saved["items"]) >= 40 and saved["format"] == 2


# -- rescue rounds ------------------------------------------------------------------------------------
class _FlakyGuards:
    """Rejects the first ``count`` positions of every round, for ``bad_rounds`` rounds."""

    def __init__(self, bad_rounds: int, count: int) -> None:
        self.bad_rounds, self.count, self.calls = bad_rounds, count, 0

    def check(self, owner, records, todo, state, preset=None):  # noqa: ARG002 - same signature as NodeGuardrails
        self.calls += 1
        return {p: "semantic: test" for p in todo[:self.count]} if self.calls <= self.bad_rounds else {}


def _run_rows(guards: _FlakyGuards, **config) -> tuple[list[int], dict]:
    services = SimpleNamespace(
        spec=SimpleNamespace(seed=1), seed_base=1, config=GuardrailConfig(max_regeneration_rounds=2, **config),
        cancel=CancellationToken(), faker=lambda seed: Faker(), pool_filler=None,
        builder=SimpleNamespace(build=lambda index, rng, faker, attempt=0: {"id": index}),
        schema=SimpleNamespace(llm_fields=[]), guards=guards)
    item = WorkItem(task_id=1, start_index=0, record_count=200)
    result = _base_result(item)
    todo, _ = _generate_rows(services, item, [{} for _ in range(200)], list(range(200)), BatchGuardState(), result,
                             LLMUsage(), attempt_base=0)
    return todo, result


def test_a_few_leftover_records_get_rescue_rounds() -> None:
    todo, result = _run_rows(_FlakyGuards(bad_rounds=6, count=3))
    assert todo == [] and result["rescued_rows"] == 3


def test_rescue_can_be_turned_off() -> None:
    todo, _ = _run_rows(_FlakyGuards(bad_rounds=6, count=3), rescue_rounds=0)
    assert todo == [0, 1, 2]


def test_many_leftover_records_are_not_rescued() -> None:
    todo, result = _run_rows(_FlakyGuards(bad_rounds=6, count=50))
    assert len(todo) == 50 and "rescued_rows" not in result


# -- end to end ---------------------------------------------------------------------------------------
def test_small_pool_causes_no_semantic_rejections(tmp_path, registry) -> None:
    engine = _engine(tmp_path, registry, pool_max_values=40, pool_grow=False)
    result = engine.run(_request(BIO_SCHEMA, 2000, llm_mode="pooled"))
    assert result.ok, result.error
    execution = json.loads((Path(result.job_dir) / "metadata" / "execution_summary.json").read_text(encoding="utf-8"))
    assert execution["semantic_rejections"] == 0
    assert len(list(iter_records(Path(result.final_output), "json"))) == 2000


def test_split_run_builds_the_pool_once(tmp_path, registry) -> None:
    engine = _engine(tmp_path, registry, pool_max_values=40, pool_grow=False)
    result = engine.run(_request(BIO_SCHEMA, 1200, llm_mode="pooled", split_size=600))
    assert result.ok, result.error
    assert (Path(result.job_dir) / "llm_pool.json").exists()
    # One speed-test call plus 40 answers at 8 per request; a rebuilt pool would double that.
    assert MockLLMAdapter.calls["mock-llm"] <= 1 + 40 // 8 + 2


def test_failed_job_reports_an_error_type(tmp_path, registry) -> None:
    for name in ("mock-llm", "mock-backup", "mock-flaky"):
        MockLLMAdapter.behaviors[name] = MockBehavior(fail_rate=1.0)
    engine = _engine(tmp_path, registry)
    result = engine.run({**_request(BIO_SCHEMA, 100), "quality_requirements": {"max_task_retries": 0}})
    assert result.status == "failed"
    assert result.error_type == "TaskExecutionError"


def test_progress_line_names_the_retry_round() -> None:
    tracker = ExecutionTracker("job", total_tasks=4, requested_records=100)
    tracker.failed.update({2, 3})
    assert "2 failed" in tracker.progress_line()
    tracker.set_phase("retry round 1/2")
    line = tracker.progress_line()
    assert "retry round 1/2:" in line and "2 being retried" in line
