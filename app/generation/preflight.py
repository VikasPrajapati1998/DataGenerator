"""Speed test and feasibility plan for jobs that need LLM text.

Before any task runs, every pool model answers one small request. The measured
latency drives three decisions:

* **which models to use**: models that fail or time out are dropped for the
  job. On a CPU-only machine, local models share the same cores, so running
  several at once makes each one slower; only the fastest local model is kept
  (cloud models, which do not use local CPU, stay);
* **which LLM mode to use**: ``per_record`` or ``pooled`` (``auto`` switches to
  pooled from ``SDE_POOLED_MIN_RECORDS`` records);
* **whether to start at all**: if the projected run time exceeds
  ``SDE_MAX_ETA_HOURS``, the job stops with the estimate and the options,
  unless ``--force`` is given.
"""

from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from faker import Faker

from app.config.model_registry import ModelConfig
from app.config.settings import Settings
from app.core.exceptions import LLMError
from app.generation.llm_enricher import LLMEnricher
from app.generation.llm_pool import MIN_POOL_VALUES
from app.models.model_router import LLMUsage, ModelRouter
from app.monitoring.logger import get_logger
from app.utils.helpers import format_duration

logger = get_logger(__name__)
_FAST_RECORDS_PER_SECOND = 3000.0  # deterministic generation + validation, conservative


@dataclass
class LLMPlan:
    """Outcome of the speed test."""

    mode: str
    models: list[str]
    dropped: dict[str, str] = field(default_factory=dict)
    seconds_per_record: dict[str, float] = field(default_factory=dict)
    records_per_second: float = 0.0
    per_record_eta_seconds: float = 0.0
    pooled_eta_seconds: float = 0.0
    eta_seconds: float = 0.0
    pool_target: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable copy."""
        return asdict(self)


def benchmark(services: Any, configs: list[ModelConfig], settings: Settings, seed: int) -> dict[str, float | str]:
    """Time one single-record request per model.

    Returns:
        ``model key -> seconds per record``, or the failure reason (str).
    """
    results: dict[str, float | str] = {}
    rng = random.Random(seed)
    faker = Faker(services.schema.locale)
    faker.seed_instance(seed)
    for config in configs:
        router = ModelRouter([config], settings, cancel=services.cancel)
        enricher = LLMEnricher(services.schema, router, services.config, services.spec.instructions)
        record = services.builder.build(rng.randrange(1_000_000), rng, faker)
        started = time.perf_counter()
        try:
            failures = enricher.enrich([record], [0], rng, LLMUsage())
            results[config.key] = (time.perf_counter() - started) if not failures else f"bad answer ({failures[0]})"
        except LLMError as exc:
            results[config.key] = f"{type(exc).__name__}: {str(exc)[:120]}"
        finally:
            router.close()
        value = results[config.key]
        logger.info("Speed test %s: %s", config.key,
                    f"{value:.1f}s for one record" if isinstance(value, float) else f"failed ({value})")
    return results


def choose_models(configs: list[ModelConfig], results: dict[str, float | str], has_gpu: bool
                  ) -> tuple[list[ModelConfig], dict[str, str]]:
    """Keep the models worth using.

    Failed models are dropped. Without a GPU, local models compete for the same
    CPU cores, so only the fastest local model is kept; cloud models stay.

    Returns:
        Kept models (fastest first) and ``key -> reason`` for dropped ones.
    """
    dropped = {c.key: f"failed the speed test: {results[c.key]}" for c in configs
               if not isinstance(results.get(c.key), float)}
    working = sorted((c for c in configs if c.key not in dropped), key=lambda c: results[c.key])
    local = [c for c in working if c.execution == "local"]
    if not has_gpu and len(local) > 1:
        for slower in local[1:]:
            dropped[slower.key] = (f"CPU-only machine: {local[0].key} is faster "
                                   f"({results[local[0].key]:.1f}s vs {results[slower.key]:.1f}s per record), "
                                   "and local models running together slow each other down")
        working = [c for c in working if c.key not in dropped]
    return working, dropped


def plan_llm(models: list[ModelConfig], results: dict[str, float | str], dropped: dict[str, str],
             slots: dict[str, int], total_records: int, requested_mode: str, settings: Settings,
             text_only: bool) -> LLMPlan:
    """Project run times and choose the LLM mode.

    Args:
        models: Kept models.
        results: Speed-test results.
        dropped: Dropped models and why.
        slots: Concurrent requests allowed per model.
        total_records: Records of the whole dataset (all sub-jobs of a split run).
        requested_mode: ``auto``, ``per_record`` or ``pooled``.
        settings: Application settings.
        text_only: Whether every LLM field is free text (pooled mode needs that).

    Returns:
        The plan.
    """
    throughput = sum(slots.get(c.key, 1) / max(float(results[c.key]), 0.001) for c in models)
    per_record = total_records / throughput + total_records / _FAST_RECORDS_PER_SECOND
    budget = settings.pool_build_minutes * 60
    pool_target = int(min(settings.pool_max_values, total_records, max(MIN_POOL_VALUES, budget * throughput)))
    pooled = pool_target / throughput + total_records / _FAST_RECORDS_PER_SECOND
    mode = requested_mode
    if mode == "auto":
        mode = "pooled" if text_only and total_records >= settings.pooled_min_records else "per_record"
    if mode == "pooled" and not text_only:
        mode = "per_record"
    return LLMPlan(mode=mode, models=[c.key for c in models], dropped=dropped,
                   seconds_per_record={k: round(v, 2) for k, v in results.items() if isinstance(v, float)},
                   records_per_second=round(throughput, 4), per_record_eta_seconds=round(per_record),
                   pooled_eta_seconds=round(pooled), eta_seconds=round(per_record if mode == "per_record" else pooled),
                   pool_target=pool_target)


def feasibility_message(plan: LLMPlan, total_records: int, max_hours: float) -> str:
    """Explain why a job will not be started and what to do instead."""
    options = []
    if plan.mode == "per_record":
        options.append(f"--llm-mode pooled (about {format_duration(plan.pooled_eta_seconds)})")
    options += ["--allow-no-llm (placeholder text, runs at full speed)", "fewer --records",
                "a faster or cloud model in SDE_MODEL_POOL", "--force to run anyway"]
    return (f"Estimated run time {format_duration(plan.eta_seconds)} for {total_records:,} records in "
            f"{plan.mode} mode ({plan.records_per_second:.2f} LLM records/s with {', '.join(plan.models)}) "
            f"exceeds SDE_MAX_ETA_HOURS={max_hours:g}. Options: " + "; ".join(options))


