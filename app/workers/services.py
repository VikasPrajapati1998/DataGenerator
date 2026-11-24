"""Per-process service bundle used by generation tasks.

Everything a task needs (schema, builders, validators, guardrails, registry,
LLM router) is built once per process from the frozen ``job_config.json``.
Thread-mode tasks share the parent's bundle; worker processes build their own
from the same file, so both modes behave identically.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from faker import Faker

from app.config.guardrail_config import GuardrailConfig
from app.config.model_registry import ModelConfig
from app.config.settings import get_settings
from app.core.cancellation import CancellationToken
from app.core.exceptions import StorageError
from app.generation.llm_enricher import LLMEnricher
from app.generation.llm_pool import PooledFiller, PoolGrower
from app.generation.record_builder import RecordBuilder
from app.guardrails.node_guardrails import NodeGuardrails
from app.guardrails.semantic_deduplication import Embedder, HashingEmbedder, OllamaEmbedder, SimHasher
from app.guardrails.uniqueness import UniquenessRegistry
from app.models.model_factory import create_adapter
from app.models.model_router import ModelRouter, UsageListener
from app.models.rate_limiter import Limiter, TokenBucket
from app.monitoring.logger import get_logger
from app.quality.realism_validator import RealismValidator
from app.schemas.input_schema import DatasetSchema, ProfileRange
from app.schemas.validation_schema import RecordValidator
from app.storage.file_manager import JobPaths, read_json
from app.utils.hashing import stable_seed

logger = get_logger(__name__)


@dataclass
class JobSpec:
    """Frozen, serialisable description of a job (contents of ``job_config.json``)."""

    job_id: str
    job_dir: str
    request: dict[str, Any]
    schema: dict[str, Any]
    total_records: int
    output_format: str
    instructions: str | None
    model_chain: list[dict[str, Any]]
    guardrails: dict[str, Any]
    profiles: dict[str, dict[str, dict[str, float]]]
    seed: int
    batch_size: int
    tasks: list[dict[str, Any]] = field(default_factory=list)
    model_pool: list[str] = field(default_factory=list)
    llm_pool_path: str | None = None
    disabled_models: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable copy."""
        return dict(self.__dict__)

    @classmethod
    def load(cls, job_config: Path) -> JobSpec:
        """Load a job specification from ``job_config.json``."""
        data = read_json(job_config)
        return cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})


class WorkerServices:
    """Services shared by all tasks of one job inside one process.

    Args:
        spec: The job specification.
        limiters: Per-model LLM concurrency limiters.
        rate_divisor: Share of the provider rate limit available to this process.
        cancel: Cancellation token.
    """

    def __init__(self, spec: JobSpec, limiters: dict[str, Limiter] | None = None, rate_divisor: int = 1,
                 cancel: CancellationToken | None = None) -> None:
        self.spec = spec
        self.paths = JobPaths(Path(spec.job_dir))
        self.schema = DatasetSchema(**spec.schema)
        # Random streams are keyed by the job seed *and* the schema, so two datasets generated with the same
        # seed (say patients and appointments, both seed 42) do not get identical random values row by row.
        self.seed_base = stable_seed(spec.seed, json.dumps(spec.schema, sort_keys=True, default=str))
        self.config = GuardrailConfig(**spec.guardrails)
        self.cancel = cancel or CancellationToken(self.paths.root)
        profiles = {name: {cat: ProfileRange(**rng) for cat, rng in cats.items()}
                    for name, cats in spec.profiles.items()}
        sub_job = spec.request.get("sub_job") or {}
        self.dataset_total = int(sub_job.get("dataset_total_records") or spec.total_records)
        # Exact counts and parent links depend on the global record index and the *dataset* seed, so every
        # sub-job of a split run lays out its part of the same dataset.
        layout_seed = stable_seed(spec.seed if sub_job.get("dataset_seed") is None else sub_job["dataset_seed"],
                                  json.dumps(spec.schema, sort_keys=True, default=str))
        self.builder = RecordBuilder(self.schema, self.dataset_total, profiles, links=spec.request.get("links"),
                                     seed=layout_seed)
        self.builder.require_profiles()
        self.validator = RecordValidator(self.schema)
        self.realism = RealismValidator(self.schema, profiles)
        # Sub-jobs of a split run share one registry (cross-sub-job uniqueness), each in its own namespace.
        registry_path = Path(sub_job["registry_path"]) if sub_job.get("registry_path") else self.paths.registry_db
        self.registry = UniquenessRegistry(registry_path, namespace=int(sub_job.get("part", 0)))
        self.chain = [ModelConfig(**c) for c in spec.model_chain]
        self._limiters = limiters
        self._rate_divisor = rate_divisor
        self._buckets: dict[str, TokenBucket] = {}
        self.disabled_models: set[str] = set()  # shared by all routers: a model out of credits stops everywhere
        self.cooldowns: dict[str, float] = {}  # shared: a rate-limited model pauses for every task at once
        self.timeouts: dict[str, int] = {}  # shared: consecutive timeouts per model
        self.disabled_models.update(spec.disabled_models)  # e.g. dropped by the speed test
        self.pool_filler: PooledFiller | None = None
        self.pool_grower: PoolGrower | None = None
        self._routers: dict[str | None, ModelRouter] = {}
        self._enrichers: dict[str | None, LLMEnricher] = {}
        self._router_lock = threading.Lock()
        self.llm_listener: UsageListener | None = None
        self.embedder = self._embedder()
        self.simhasher = SimHasher()
        self.guards = NodeGuardrails(self.schema, self.config, self.validator, self.realism, self.registry,
                                     self.embedder if self.schema.semantic_fields else None)
        self._local = threading.local()
        if spec.llm_pool_path:
            self.use_pool(Path(spec.llm_pool_path))

    def use_pool(self, pool_path: Path) -> PooledFiller:
        """Switch LLM fields to pooled mode.

        Pooled texts are built from a limited set of answers on purpose, so the
        embedding near-duplicate check (meant to catch a model repeating itself)
        is turned off for them; exact repeats are still limited per batch
        (``max_value_repeat_ratio``).

        Args:
            pool_path: ``llm_pool.json``.

        Returns:
            The filler.
        """
        self.pool_filler = PooledFiller(self.schema, pool_path, self.config.max_text_length)
        self.guards.embedder = None
        return self.pool_filler

    def _embedder(self) -> Embedder:
        name = self.config.embedder
        if name.startswith("ollama:"):
            model = ModelConfig(provider="ollama_local", model_name=name.split(":", 1)[1],
                                model_id=name.split(":", 1)[1], structured_output="prompt_only")
            return OllamaEmbedder(create_adapter(model, get_settings()))
        return HashingEmbedder()

    def router_for(self, model_key: str | None) -> ModelRouter:
        """Return the router whose primary is ``model_key``.

        The task's assigned pool model comes first; every other model of the
        job's chain follows as a fallback. Routers share token buckets and
        concurrency limiters, so per-model limits hold across all tasks.

        Args:
            model_key: Assigned model, or ``None`` for the job's default order.

        Returns:
            A cached router.
        """
        with self._router_lock:
            if model_key not in self._routers:
                chain = sorted(self.chain, key=lambda c: c.key != model_key)
                self._routers[model_key] = ModelRouter(chain, get_settings(), self._limiters, self.cancel,
                                                       self._rate_divisor, self._buckets, self.disabled_models,
                                                       self.cooldowns, self.timeouts)
            return self._routers[model_key]

    def enricher_for(self, model_key: str | None) -> LLMEnricher:
        """Return the LLM enricher bound to :meth:`router_for` ``(model_key)``."""
        router = self.router_for(model_key)
        with self._router_lock:
            if model_key not in self._enrichers:
                self._enrichers[model_key] = LLMEnricher(self.schema, router, self.config, self.spec.instructions)
            return self._enrichers[model_key]

    def faker(self, seed: int) -> Faker:
        """Return this thread's Faker instance, re-seeded for a task."""
        instance = getattr(self._local, "faker", None)
        if instance is None:
            instance = Faker(self.schema.locale)
            self._local.faker = instance
        instance.seed_instance(seed)
        return instance

    def close(self) -> None:
        """Release network and database resources (saving any text the pool grower added first)."""
        if self.pool_grower is not None:
            grower, self.pool_grower = self.pool_grower, None
            try:
                grower.stop()
            except StorageError as exc:
                logger.warning("Text pool: the answers added in the background could not be saved: %s", exc)
        for router in self._routers.values():
            router.close()
        self._routers.clear()
        self._enrichers.clear()
        self.registry.close()
        self.builder.close()
