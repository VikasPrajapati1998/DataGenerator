"""Model fallback routing with retries, rate limits and a circuit breaker.

A router owns an ordered chain ``[primary, fallback_1, ...]``. Every request
is retried with exponential backoff on the current model; when that model is
exhausted or unavailable, the next model in the chain is tried. Models that
fail repeatedly are skipped for a cool-down period (circuit breaker).
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from app.config.model_registry import ModelConfig
from app.config.settings import Settings, get_settings
from app.core.cancellation import CancellationToken
from app.core.exceptions import (
    RETRYABLE_LLM_ERRORS,
    ConfigurationError,
    ContextLengthError,
    InvalidModelOutputError,
    LLMError,
    ModelUnavailableError,
    ProviderError,
    ProviderTimeoutError,
    QuotaExceededError,
    RateLimitError,
)
from app.models.base_adapter import AvailabilityResult, BaseLLMAdapter, LLMResponse
from app.models.model_factory import create_adapter
from app.models.rate_limiter import Limiter, TokenBucket
from app.monitoring.logger import get_logger, redact
from app.utils.retry import retry_call

logger = get_logger(__name__)
_CIRCUIT_THRESHOLD = 3
_CIRCUIT_COOLDOWN = 30.0
_SLOW_CALL_SECONDS = 45.0
_RATE_LIMIT_COOLDOWN = 60.0
_TIMEOUT_COOLDOWN = 30.0
_TIMEOUTS_BEFORE_DISABLE = 2
_RETRY_ON_SAME_MODEL = (InvalidModelOutputError, ProviderError)
"""Errors retried on the same model even when another model is available (likely to pass on retry)."""


class UsageListener(Protocol):
    """Receives live LLM events (implemented by the execution tracker)."""

    def on_llm_event(self, event: str, value: float) -> None:
        """Handle ``call`` (latency s), ``failure``, ``waiting`` (+1/-1) or ``records`` (count)."""


@dataclass
class LLMUsage:
    """Per-task LLM usage counters (merged by the execution tracker).

    Attributes:
        calls: Successful requests.
        failures: Failed attempts (including retried ones).
        fallbacks: Requests served by a model other than the primary.
        prompt_tokens: Prompt tokens reported by providers.
        completion_tokens: Completion tokens reported by providers.
        latency_total: Summed request latency in seconds.
        by_model: Successful requests per model key.
        listener: Optional live observer (the execution tracker in thread mode)
            notified of every call, slot wait and enriched record as it happens.
    """

    calls: int = 0
    failures: int = 0
    fallbacks: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_total: float = 0.0
    by_model: dict[str, int] = field(default_factory=dict)
    listener: UsageListener | None = field(default=None, repr=False, compare=False)

    def notify(self, event: str, value: float = 0.0) -> None:
        """Forward a live event (``call``, ``failure``, ``waiting``, ``records``) to the listener."""
        if self.listener is not None:
            self.listener.on_llm_event(event, value)

    def record_progress(self, records: int) -> None:
        """Report records whose LLM fields were filled (live progress)."""
        self.notify("records", records)

    def merge(self, other: dict[str, Any] | LLMUsage) -> None:
        """Add another usage record into this one.

        Args:
            other: Usage object or its ``to_dict`` form.
        """
        data = other.to_dict() if isinstance(other, LLMUsage) else other
        for name in ("calls", "failures", "fallbacks", "prompt_tokens", "completion_tokens"):
            setattr(self, name, getattr(self, name) + int(data.get(name, 0)))
        self.latency_total += float(data.get("latency_total", 0.0))
        for key, count in data.get("by_model", {}).items():
            self.by_model[key] = self.by_model.get(key, 0) + int(count)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable copy."""
        return {"calls": self.calls, "failures": self.failures, "fallbacks": self.fallbacks,
                "prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens,
                "latency_total": round(self.latency_total, 4), "by_model": dict(self.by_model)}


class ModelRouter:
    """Routes structured-generation requests across a model fallback chain.

    Args:
        chain: Model configurations, primary first.
        settings: Application settings.
        limiters: Optional per-model concurrency limiters (thread or process semaphores).
        cancel: Cancellation token checked between attempts.
        rate_divisor: Divides each model's requests-per-minute (one share per process).
        buckets: Token buckets shared with other routers of the same process, so
            a model's rate limit holds no matter which router calls it.
        disabled: Set of model keys disabled for the job, shared the same way.
        cooldowns: ``model key -> monotonic time`` until which a rate-limited or
            timing-out model is skipped, shared the same way.
        timeouts: ``model key -> consecutive timeouts``, shared the same way. After
            ``_TIMEOUTS_BEFORE_DISABLE`` in a row a model is disabled (if others remain).

    Raises:
        ConfigurationError: If ``chain`` is empty.
    """

    def __init__(self, chain: list[ModelConfig], settings: Settings | None = None,
                 limiters: dict[str, Limiter] | None = None, cancel: CancellationToken | None = None,
                 rate_divisor: int = 1, buckets: dict[str, TokenBucket] | None = None,
                 disabled: set[str] | None = None, cooldowns: dict[str, float] | None = None,
                 timeouts: dict[str, int] | None = None) -> None:
        if not chain:
            raise ConfigurationError("A model router needs at least one model")
        self.chain = chain
        self.settings = settings or get_settings()
        self.limiters = limiters or {}
        self.cancel = cancel
        self._adapters: dict[str, BaseLLMAdapter] = {}
        self._buckets = buckets if buckets is not None else {}
        for config in chain:
            if config.requests_per_minute and config.key not in self._buckets:
                self._buckets[config.key] = TokenBucket(max(1.0, config.requests_per_minute / max(rate_divisor, 1)))
        self._disabled = disabled if disabled is not None else set()
        self.slot_timeout = float(self.settings.llm_slot_timeout)
        self._failures: dict[str, int] = {}
        self._open_until = cooldowns if cooldowns is not None else {}
        self._timeouts: dict[str, int] = timeouts if timeouts is not None else {}
        self._lock = threading.Lock()

    @property
    def primary(self) -> ModelConfig:
        """The preferred model."""
        return self.chain[0]

    def adapter(self, config: ModelConfig) -> BaseLLMAdapter:
        """Return (and lazily create) the adapter for ``config``."""
        with self._lock:
            if config.key not in self._adapters:
                self._adapters[config.key] = create_adapter(config, self.settings)
            return self._adapters[config.key]

    def check_availability(self) -> dict[str, AvailabilityResult]:
        """Probe every model in the chain.

        Returns:
            Availability per model key.
        """
        return {config.key: self.adapter(config).check_availability() for config in self.chain}

    def _acquire_slot(self, slot: Limiter, key: str) -> None:
        """Wait for a free request slot, but never forever.

        A worker process that dies while holding a cross-process semaphore never
        releases it; without a bound, every other worker would wait forever.
        Waiting polls once per second so cancellation is honoured too.

        Raises:
            CancelledError: If the job is cancelled while waiting.
            ProviderTimeoutError: If no slot frees up within ``slot_timeout``.
        """
        deadline = time.monotonic() + self.slot_timeout
        while not slot.acquire(True, 1.0):
            if self.cancel is not None:
                self.cancel.raise_if_cancelled()
            if time.monotonic() > deadline:
                raise ProviderTimeoutError(
                    f"no free request slot for {key} after {self.slot_timeout:.0f}s "
                    "(the model is saturated or a worker died while holding a slot)")

    def disable(self, key: str, reason: Exception) -> None:
        """Stop using ``key`` for the rest of the job (shared by every router of the process).

        Args:
            key: Model key.
            reason: The permanent error that caused it.
        """
        with self._lock:
            if key in self._disabled:
                return
            self._disabled.add(key)
        label = "quota/credits exhausted" if isinstance(reason, QuotaExceededError) else "unavailable"
        logger.warning("Model %s disabled for the rest of this job: %s (%s). Its tasks continue on the other "
                       "models of the pool.", key, label, redact(str(reason))[:200])

    def is_disabled(self, key: str) -> bool:
        """Whether ``key`` was disabled for this job."""
        return key in self._disabled

    def _circuit_open(self, key: str) -> bool:
        with self._lock:
            return self._open_until.get(key, 0.0) > time.monotonic()

    def _record(self, key: str, success: bool) -> None:
        with self._lock:
            if success:
                self._failures[key] = 0
                return
            self._failures[key] = self._failures.get(key, 0) + 1
            if self._failures[key] >= _CIRCUIT_THRESHOLD:
                self._open_until[key] = time.monotonic() + _CIRCUIT_COOLDOWN
                self._failures[key] = 0
                logger.warning("Circuit opened for %s for %.0fs after repeated failures", key, _CIRCUIT_COOLDOWN)

    def generate_json(self, system: str, user: str, json_schema: dict[str, Any], usage: LLMUsage,
                      validator: Callable[[Any], Any] | None = None) -> Any:
        """Generate structured output, retrying and falling back as needed.

        Args:
            system: System prompt.
            user: User prompt.
            json_schema: JSON Schema of the expected answer.
            usage: Counters updated in place.
            validator: Optional callable that checks/converts the parsed JSON
                and raises :class:`InvalidModelOutputError` to trigger a retry.

        Returns:
            The (validated) parsed JSON.

        Raises:
            ModelUnavailableError: If every model in the chain failed.
        """
        errors: list[str] = [f"{c.key}: disabled for this job" for c in self.chain if c.key in self._disabled]
        timeouts: list[str] = []
        order = self._candidates()
        for index, (position, config) in enumerate(order):
            # With another healthy model left, a rate limit or timeout switches to it at once (the model
            # cools down); only the last option waits and retries on the same model.
            has_alternative = any(not self._circuit_open(c.key) for _, c in order[index + 1:])
            retry_on = _RETRY_ON_SAME_MODEL if has_alternative else RETRYABLE_LLM_ERRORS
            adapter = self.adapter(config)

            def attempt(adapter: BaseLLMAdapter = adapter, config: ModelConfig = config) -> Any:
                started = time.perf_counter()
                try:
                    if bucket := self._buckets.get(config.key):
                        bucket.acquire()
                    slot = self.limiters.get(config.key)
                    usage.notify("waiting", 1)
                    try:
                        if slot is not None:
                            self._acquire_slot(slot, config.key)
                    finally:
                        usage.notify("waiting", -1)
                    try:
                        started = time.perf_counter()
                        data, response = adapter.generate_json(system, user, json_schema)
                    finally:
                        if slot is not None:
                            slot.release()
                    result = validator(data) if validator else data
                except RETRYABLE_LLM_ERRORS as exc:
                    usage.failures += 1
                    usage.notify("failure")
                    logger.info("LLM call to %s failed after %.1fs: %s", config.key, time.perf_counter() - started,
                                type(exc).__name__)
                    raise
                if response.latency > _SLOW_CALL_SECONDS:
                    logger.warning("Slow LLM call: %s took %.1fs (%d prompt / %d output tokens)", config.key,
                                   response.latency, response.prompt_tokens, response.completion_tokens)
                self._account(usage, config, response, position)
                return result

            try:
                result = retry_call(attempt, config.retry.to_policy(), retry_on,
                                    description=f"LLM call {config.key}", cancel=self.cancel)
                self._record(config.key, True)
                with self._lock:
                    self._timeouts[config.key] = 0
                return result
            except ModelUnavailableError as exc:
                # Permanent (no credits, wrong id, no access): stop using this model for the rest of the job.
                usage.failures += 1
                self.disable(config.key, exc)
                errors.append(f"{config.key}: {exc}")
            except RateLimitError as exc:
                self.cool_down(config.key, exc.retry_after or _RATE_LIMIT_COOLDOWN, "rate limit reached")
                errors.append(f"{config.key}: rate limited")
            except ProviderTimeoutError as exc:
                timeouts.append(config.key)
                errors.append(f"{config.key}: {exc}")
                with self._lock:
                    self._timeouts[config.key] = self._timeouts.get(config.key, 0) + 1
                    repeated = self._timeouts[config.key] >= _TIMEOUTS_BEFORE_DISABLE
                others = [c for c in self.chain if c.key != config.key and c.key not in self._disabled]
                if repeated and others:
                    # A model that keeps timing out wastes the whole timeout every time: stop using it.
                    self.disable(config.key, exc)
                else:
                    self.cool_down(config.key, _TIMEOUT_COOLDOWN, "timed out")
            except ContextLengthError as exc:
                # This request is too large for this model only: try a model with a bigger window.
                logger.warning("%s: request exceeds its context window; trying the next model", config.key)
                errors.append(f"{config.key}: {exc}")
            except LLMError as exc:
                self._record(config.key, False)
                errors.append(f"{config.key}: {type(exc).__name__}: {exc}")
            if index < len(order) - 1:
                logger.warning("Switching from %s to %s", config.key, order[index + 1][1].key)
        if order and len(timeouts) == len(order):
            # Every model ran out of time: the request is probably too big; callers can retry it smaller.
            raise ProviderTimeoutError("All models timed out: " + " | ".join(errors))
        raise ModelUnavailableError("All models failed: " + " | ".join(errors))

    def _candidates(self) -> list[tuple[int, ModelConfig]]:
        """Usable models in chain order: healthy ones first, then cooling-down ones (soonest first)."""
        usable = [(p, c) for p, c in enumerate(self.chain) if c.key not in self._disabled]
        healthy = [(p, c) for p, c in usable if not self._circuit_open(c.key)]
        with self._lock:
            cooling = sorted((pc for pc in usable if pc not in healthy),
                             key=lambda pc: self._open_until.get(pc[1].key, 0.0))
        return healthy + cooling

    def cool_down(self, key: str, seconds: float, reason: str) -> None:
        """Take ``key`` out of rotation for ``seconds`` (shared by every router of the process).

        Args:
            key: Model key.
            seconds: Cool-down length.
            reason: Why, for the log.
        """
        with self._lock:
            already = self._open_until.get(key, 0.0) > time.monotonic()
            self._open_until[key] = max(self._open_until.get(key, 0.0), time.monotonic() + seconds)
        if not already:
            logger.warning("Model %s %s; pausing it for %.0fs and using the other models meanwhile",
                           key, reason, seconds)

    @staticmethod
    def _account(usage: LLMUsage, config: ModelConfig, response: LLMResponse, position: int) -> None:
        usage.calls += 1
        usage.fallbacks += 1 if position > 0 else 0
        usage.prompt_tokens += response.prompt_tokens
        usage.completion_tokens += response.completion_tokens
        usage.latency_total += response.latency
        usage.by_model[config.key] = usage.by_model.get(config.key, 0) + 1
        usage.notify("call", response.latency)

    def close(self) -> None:
        """Close every adapter's network resources."""
        with self._lock:
            for adapter in self._adapters.values():
                adapter.close()
            self._adapters.clear()


__all__ = ["InvalidModelOutputError", "LLMUsage", "ModelRouter"]
