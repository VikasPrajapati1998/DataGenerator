"""LLM speed test, feasibility check, pooled mode, timeout handling and progress warnings."""

from __future__ import annotations

import random
from pathlib import Path

from app.config.guardrail_config import GuardrailConfig
from app.config.model_registry import ModelConfig, ModelRegistry
from app.config.settings import Settings
from app.generation.llm_enricher import LLMEnricher
from app.generation.preflight import choose_models, feasibility_message, plan_llm
from app.models.mock_adapter import MockBehavior, MockLLMAdapter
from app.models.model_router import LLMUsage, ModelRouter
from app.monitoring.execution_tracker import ExecutionTracker
from app.schemas.schema_parser import parse_schema
from app.storage.formats import iter_records
from app.utils.helpers import format_duration

BIO_SCHEMA = {
    "customer_id": "unique integer",
    "name": "realistic full name",
    "segment": ["retail", "business", "enterprise"],
    "bio": "two-sentence profile of the customer's buying habits",
}


def _model(name: str, execution: str = "local") -> ModelConfig:
    return ModelConfig(provider="mock", model_name=name, model_id=name, execution=execution,
                       retry={"max_attempts": 1, "base_delay": 0.0, "max_delay": 0.0})


def _engine(tmp_path: Path, registry: ModelRegistry, **overrides):
    from app.engine import SyntheticDataEngine  # pylint: disable=import-outside-toplevel

    settings = Settings(output_dir=tmp_path / "output", log_dir=tmp_path / "logs", default_model="mock/mock-llm",
                        **overrides)
    return SyntheticDataEngine(settings=settings, registry=registry)


def _bio_request(records: int, **extra) -> dict:
    return {"schema": dict(BIO_SCHEMA), "total_records": records, "output_format": "json",
            "model_provider": "mock", "model_name": "mock-llm", "batch_size": 250, "execution_mode": "thread",
            "seed": 11, **extra}


# -- pure planning ---------------------------------------------------------------------------------
def test_format_duration_covers_days() -> None:
    assert format_duration(45) == "45s"
    assert format_duration(725) == "12m05s"
    assert format_duration(3 * 3600 + 20 * 60) == "3h20m"
    assert format_duration(4 * 86400 + 7 * 3600 + 59) == "4d07h"


def test_cpu_only_keeps_the_fastest_local_model_and_cloud_models() -> None:
    configs = [_model("slow"), _model("fast"), _model("remote", "cloud"), _model("broken")]
    results = {"mock/slow": 2.0, "mock/fast": 1.0, "mock/remote": 5.0, "mock/broken": "ProviderTimeoutError: x"}
    kept, dropped = choose_models(configs, results, has_gpu=False)
    assert [c.key for c in kept] == ["mock/fast", "mock/remote"]
    assert set(dropped) == {"mock/slow", "mock/broken"}
    assert "CPU-only" in dropped["mock/slow"] and "speed test" in dropped["mock/broken"]
    kept, dropped = choose_models(configs, results, has_gpu=True)
    assert [c.key for c in kept] == ["mock/fast", "mock/slow", "mock/remote"]
    assert set(dropped) == {"mock/broken"}


def test_plan_switches_to_pooled_for_large_text_jobs() -> None:
    model = _model("m")
    settings = Settings(pooled_min_records=10_000, pool_max_values=2000, pool_build_minutes=20)
    plan = plan_llm([model], {model.key: 0.5}, {}, {model.key: 2}, 100_000, "auto", settings, text_only=True)
    assert plan.mode == "pooled"
    assert plan.records_per_second == 4.0
    assert plan.pool_target == 2000
    assert plan.eta_seconds == plan.pooled_eta_seconds < plan.per_record_eta_seconds
    assert plan.per_record_eta_seconds == round(100_000 / 4 + 100_000 / 3000)
    small = plan_llm([model], {model.key: 0.5}, {}, {model.key: 2}, 5000, "auto", settings, text_only=True)
    assert small.mode == "per_record"
    numbers = plan_llm([model], {model.key: 0.5}, {}, {model.key: 2}, 100_000, "pooled", settings, text_only=False)
    assert numbers.mode == "per_record"  # pooled needs free-text LLM fields only


def test_feasibility_message_lists_the_options() -> None:
    model = _model("m")
    plan = plan_llm([model], {model.key: 100.0}, {}, {model.key: 1}, 1_000_000, "per_record", Settings(),
                    text_only=True)
    message = feasibility_message(plan, 1_000_000, 6)
    assert "SDE_MAX_ETA_HOURS=6" in message and "--llm-mode pooled" in message and "--force" in message


# -- router and enricher ---------------------------------------------------------------------------
def test_router_disables_a_model_after_repeated_timeouts() -> None:
    MockLLMAdapter.behaviors["mock-flaky"] = MockBehavior(timeout_rate=1.0)
    disabled: set[str] = set()
    cooldowns: dict[str, float] = {}
    router = ModelRouter([_model("mock-flaky"), _model("mock-llm")], Settings(), disabled=disabled,
                         cooldowns=cooldowns, timeouts={})
    schema = {"type": "object", "properties": {}}
    for _ in range(2):
        cooldowns.clear()  # as if the cool-down had expired, so the flaky model is tried first again
        router.generate_json("s", "u", schema, LLMUsage())
    assert "mock/mock-flaky" in disabled
    calls_before = MockLLMAdapter.calls.get("mock-flaky", 0)
    cooldowns.clear()
    router.generate_json("s", "u", schema, LLMUsage())
    assert MockLLMAdapter.calls.get("mock-flaky", 0) == calls_before


def test_enricher_splits_a_request_that_timed_out() -> None:
    MockLLMAdapter.behaviors["mock-llm"] = MockBehavior(timeout_until_call=1)
    schema = parse_schema(BIO_SCHEMA)
    router = ModelRouter([_model("mock-llm")], Settings())
    enricher = LLMEnricher(schema, router, GuardrailConfig(), None)
    records = [{"customer_id": i, "name": f"Person {i}", "segment": "retail", "bio": None} for i in range(8)]
    enricher.chunk_size = 8  # independent of the output-token budget of the mock model
    failures = enricher.enrich(records, list(range(8)), random.Random(1), LLMUsage())
    assert not failures
    assert all(r["bio"] for r in records)
    assert enricher.chunk_size == 4
    assert MockLLMAdapter.calls["mock-llm"] == 3  # 8 (timed out), then 4 + 4


# -- progress line ---------------------------------------------------------------------------------
def test_progress_line_warns_when_no_llm_answer_arrives() -> None:
    tracker = ExecutionTracker("job", total_tasks=1, requested_records=10)
    tracker.llm_required = True
    tracker.task_started(1)
    tracker.started -= 120
    assert "WARNING: no LLM answer for 2m" in tracker.progress_line()
    tracker.on_llm_event("call", 1.0)
    assert "WARNING" not in tracker.progress_line()


# -- end to end ------------------------------------------------------------------------------------
def test_job_is_refused_when_the_projected_run_time_is_too_long(tmp_path, registry) -> None:
    MockLLMAdapter.behaviors["mock-llm"] = MockBehavior(latency=0.05)
    engine = _engine(tmp_path, registry, max_eta_hours=0.00005)
    result = engine.run(_bio_request(1000, llm_mode="per_record"))
    assert not result.ok
    assert result.error_type == "FeasibilityError"
    assert "--llm-mode pooled" in result.error
    forced = engine.run(_bio_request(1000, llm_mode="per_record", force=True))
    assert forced.ok, forced.error


def test_pooled_mode_needs_far_fewer_llm_calls(tmp_path, registry) -> None:
    engine = _engine(tmp_path, registry, pool_max_values=40)
    result = engine.run({**_bio_request(300, llm_mode="pooled"), "batch_size": 100})
    assert result.ok, result.error
    assert (Path(result.job_dir) / "metadata" / "llm_pool.json").exists()
    assert MockLLMAdapter.calls["mock-llm"] < 300 // 8  # per-record mode needs at least 38 calls
    assert len(list(iter_records(Path(result.final_output), "json"))) == 300


def test_auto_mode_switches_to_pooled_above_the_threshold(tmp_path, registry) -> None:
    records = 1000
    engine = _engine(tmp_path, registry, pooled_min_records=500, pool_max_values=40)
    result = engine.run(_bio_request(records))
    assert result.ok, result.error
    assert (Path(result.job_dir) / "metadata" / "llm_pool.json").exists()
    assert MockLLMAdapter.calls["mock-llm"] < records // 8
