"""Resource-aware effective concurrency.

Two limits are computed separately:

* ``queues`` - logical core queues that run concurrently. Bounded by usable
  CPU cores, the user's cap, the number of tasks and available RAM.
* ``llm_concurrency`` - simultaneous in-flight LLM requests. Bounded by the
  provider: local Ollama serves ``OLLAMA_NUM_PARALLEL`` requests per model
  (default 1) and needs RAM/VRAM for the loaded model; cloud providers are
  bounded by their rate limits.

A queue only holds an LLM slot while waiting for a model answer, so CPU work
(Faker, validation, hashing) continues on other queues meanwhile.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from app.config.model_registry import ModelConfig
from app.config.settings import Settings
from app.core.exceptions import ResourceError
from app.scheduler.cpu_detector import ResourceInfo

_MIN_FREE_RAM_MB = 256
_OLLAMA_DEFAULT_PARALLEL = 1
_OLLAMA_MAX_LOADED = 3


@dataclass
class ConcurrencyPlan:
    """Effective concurrency decision and the reasons behind it."""

    cpu_cores: int
    queues: int
    llm_concurrency: int
    reasons: list[str] = field(default_factory=list)
    llm_slots: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable copy."""
        return asdict(self)


def plan_concurrency(resources: ResourceInfo, model: ModelConfig | None, settings: Settings, total_tasks: int,
                     max_workers: int | None = None, model_memory_mb: int | None = None,
                     llm_required: bool = True, pool: list[ModelConfig] | None = None,
                     loaded_models: set[str] | None = None) -> ConcurrencyPlan:
    """Compute effective queue and per-model LLM concurrency.

    Args:
        resources: Detected host resources.
        model: Primary model configuration (``None`` when the job calls no model).
        pool: Every model the job may call (pool plus fallbacks); each gets its
            own request limit and resident local models share the RAM budget.
        settings: Application settings.
        total_tasks: Number of planned tasks.
        max_workers: Optional user cap on queues.
        model_memory_mb: Measured model size (overrides the registry estimate).
        llm_required: Whether tasks make per-record LLM calls.
        loaded_models: Model ids Ollama currently has in memory (``/api/ps``).

    Returns:
        The concurrency plan. With too little RAM it falls back to one queue
        (``SDE_RAM_CHECK=warn``, the default).

    Raises:
        ResourceError: If RAM is insufficient and ``SDE_RAM_CHECK=strict``.
    """
    reasons: list[str] = [f"{resources.cpu_cores} usable CPU cores detected"]
    queues = resources.cpu_cores
    for cap, label in ((max_workers, "request max_workers"), (settings.max_workers, "SDE_MAX_WORKERS")):
        if cap and cap < queues:
            queues = cap
            reasons.append(f"capped to {cap} by {label}")
    if total_tasks < queues:
        queues = total_tasks
        reasons.append(f"only {total_tasks} task(s) planned")

    models = pool or ([model] if model is not None else [])
    model_mb = _model_memory_mb(models, llm_required, loaded_models or set(), model_memory_mb, reasons)
    if settings.ram_check == "off":
        reasons.append("RAM check disabled (SDE_RAM_CHECK=off)")
    else:
        budget = resources.available_ram_mb - model_mb - _MIN_FREE_RAM_MB
        if budget < settings.memory_per_worker_mb:
            message = (f"Low RAM: {resources.available_ram_mb} MB available, local model(s) still to load need "
                       f"~{model_mb} MB and each queue ~{settings.memory_per_worker_mb} MB")
            if settings.ram_check == "strict":
                raise ResourceError(f"{message} (SDE_RAM_CHECK=strict)")
            queues = 1
            reasons.append(f"{message}; running a single queue. Ollama may swap or reject local models, in which "
                           "case cloud models in the pool take over (SDE_RAM_CHECK=strict to fail instead)")
        else:
            ram_limit = budget // settings.memory_per_worker_mb
            if ram_limit < queues:
                queues = int(ram_limit)
                reasons.append(f"capped to {queues} by available RAM ({resources.available_ram_mb} MB)")

    queues = max(1, queues)
    slots: dict[str, int] = {}
    for candidate in models:
        slots[candidate.key], reason = _llm_slots(candidate, settings, queues, llm_required, resources)
        reasons.append(f"{candidate.key}: {reason}")
    if model is not None and model.key in slots:
        llm_concurrency = slots[model.key]
    else:
        llm_concurrency = next(iter(slots.values()), 0)
    if not models:
        reasons.append("no model used by this job")
    return ConcurrencyPlan(cpu_cores=resources.cpu_cores, queues=queues, llm_concurrency=llm_concurrency,
                           reasons=reasons, llm_slots=slots)


def _model_memory_mb(models: list[ModelConfig], llm_required: bool, loaded_models: set[str],
                     model_memory_mb: int | None, reasons: list[str]) -> int:
    """RAM that local models still need on top of what is already in use.

    Cloud models (including ``-cloud`` models proxied by local Ollama) need no
    local RAM. Models Ollama has already loaded are part of the "used" RAM
    already. Ollama loads models on demand and evicts idle ones, so only the
    largest not-yet-loaded local model is required up front. Further pool
    models are swapped in as needed (reported, not reserved).
    """
    if not llm_required:
        return 0
    local = [m for m in models if m.execution == "local" and m.provider.startswith("ollama")]
    pending = sorted((m for m in local if m.model_id not in loaded_models),
                     key=lambda m: m.approx_memory_mb or 0, reverse=True)
    if loaded := [m.key for m in local if m.model_id in loaded_models]:
        reasons.append(f"already loaded in Ollama: {loaded}")
    if len(local) > _OLLAMA_MAX_LOADED:
        reasons.append(f"{len(local)} local pool models but Ollama keeps ~{_OLLAMA_MAX_LOADED} loaded; "
                       "expect model swapping")
    if not pending:
        return 0
    if model_memory_mb and len(local) == 1:
        return model_memory_mb
    return pending[0].approx_memory_mb or 0


def _llm_slots(model: ModelConfig, settings: Settings, queues: int, llm_required: bool,
               resources: ResourceInfo) -> tuple[int, str]:
    """Return the in-flight request limit for one model and why."""
    if not llm_required:
        llm, reason = max(1, model.max_concurrency or queues), "no per-record LLM fields (calibration only)"
    elif model.max_concurrency:
        llm, reason = model.max_concurrency, f"max_concurrency={model.max_concurrency}"
    elif model.provider == "ollama_local" and model.execution == "local":
        llm = settings.ollama_num_parallel or _OLLAMA_DEFAULT_PARALLEL
        reason = (f"GPU {resources.gpus[0].name}, OLLAMA_NUM_PARALLEL={llm}" if resources.gpus
                  else f"CPU-only inference, OLLAMA_NUM_PARALLEL={llm}")
    else:
        rpm = model.requests_per_minute or 60
        llm, reason = max(1, min(queues, rpm // 10 or 1)), f"cloud model limited by {rpm} requests/minute"
    llm = max(1, min(llm, queues))
    return llm, f"{llm} concurrent request(s), {reason}"
