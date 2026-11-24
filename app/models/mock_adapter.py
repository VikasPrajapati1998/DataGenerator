"""Deterministic offline adapter used by tests, CI and dry runs.

It understands the two prompt types the engine sends (record enrichment and
profile calibration) and returns plausible structured answers. Failure modes
(timeouts, rate limits, invalid JSON, provider errors) can be injected per
model name through :data:`MockLLMAdapter.behaviors` to exercise retries and
fallback without a real model.
"""

from __future__ import annotations

import json
import random
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, ClassVar

from app.core.exceptions import (
    ContextLengthError,
    ProviderError,
    ProviderTimeoutError,
    QuotaExceededError,
    RateLimitError,
)
from app.models.base_adapter import AvailabilityResult, BaseLLMAdapter, LLMResponse
from app.utils.hashing import stable_seed

_RECORDS_BLOCK = re.compile(r"<records>\s*(.*?)\s*</records>", re.DOTALL)
_OPENERS = ("Seasoned", "Detail-oriented", "Energetic", "Pragmatic", "Curious", "Reliable", "Creative",
            "Analytical", "Customer-focused", "Resourceful", "Methodical", "Collaborative", "Driven",
            "Thoughtful", "Versatile", "Hands-on", "Strategic", "Patient", "Adaptable", "Meticulous")
_CORES = ("specialist", "professional", "team member", "contributor", "practitioner", "problem solver",
          "planner", "communicator", "organiser", "analyst")
_FOCUS = ("process improvement", "stakeholder management", "data quality", "client relationships",
          "cost control", "mentoring juniors", "automation", "compliance", "forecasting", "vendor coordination",
          "documentation", "quality assurance", "budget tracking", "onboarding", "reporting", "negotiation",
          "risk assessment", "scheduling", "product launches", "user research")
_EXTRAS = ("Enjoys cross-team projects", "Recently completed a certification", "Known for clear writing",
           "Prefers structured workflows", "Volunteers for pilot programmes", "Balances speed with accuracy",
           "Regularly leads review sessions", "Comfortable with ambiguity", "Keeps meticulous records",
           "Often trains new colleagues")


@dataclass
class MockBehavior:
    """Failure-injection settings for one mock model name.

    Attributes:
        fail_rate: Probability of a generic provider error.
        timeout_rate: Probability of a timeout.
        rate_limit_rate: Probability of a rate-limit error.
        invalid_json_rate: Probability of returning malformed JSON.
        latency: Seconds to sleep per request.
        available: What the availability probe reports.
        drop_rate: Probability of omitting a record from an enrichment answer.
        fail_until_call: The first N calls to this model fail with a provider error.
        timeout_until_call: The first N calls to this model time out.
        quota_exhausted: Every call fails like an HTTP 402 "out of credits" answer.
        context_overflow: Every call fails as too long for the model's context window.
    """

    fail_rate: float = 0.0
    timeout_rate: float = 0.0
    rate_limit_rate: float = 0.0
    invalid_json_rate: float = 0.0
    latency: float = 0.0
    available: bool = True
    drop_rate: float = 0.0
    fail_until_call: int = 0
    timeout_until_call: int = 0
    quota_exhausted: bool = False
    context_overflow: bool = False


class MockLLMAdapter(BaseLLMAdapter):
    """Offline adapter that fabricates plausible structured answers."""

    behaviors: ClassVar[dict[str, MockBehavior]] = {}
    _counter_lock: ClassVar[threading.Lock] = threading.Lock()
    calls: ClassVar[dict[str, int]] = {}

    def _behavior(self) -> MockBehavior:
        return self.behaviors.get(self.config.model_name, MockBehavior())

    def _chat(self, system: str, user: str, json_schema: dict[str, Any] | None, json_mode: bool) -> LLMResponse:
        with self._counter_lock:
            self.calls[self.config.model_name] = self.calls.get(self.config.model_name, 0) + 1
            call_no = self.calls[self.config.model_name]
        behavior = self._behavior()
        rng = random.Random(stable_seed(self.config.model_name, user, call_no))
        if behavior.latency:
            time.sleep(behavior.latency)
        if behavior.context_overflow:
            raise ContextLengthError(f"{self.key}: request exceeds the context window (simulated)")
        if behavior.quota_exhausted:
            raise QuotaExceededError(f"{self.key}: quota or credits exhausted (HTTP 402, simulated)")
        if call_no <= behavior.fail_until_call:
            raise ProviderError(f"{self.key} simulated outage (call {call_no})")
        if call_no <= behavior.timeout_until_call:
            raise ProviderTimeoutError(f"{self.key} simulated timeout (call {call_no})")
        roll = rng.random()
        if roll < behavior.timeout_rate:
            raise ProviderTimeoutError(f"{self.key} simulated timeout")
        roll -= behavior.timeout_rate
        if roll < behavior.rate_limit_rate:
            raise RateLimitError(f"{self.key} simulated rate limit")
        roll -= behavior.rate_limit_rate
        if roll < behavior.fail_rate:
            raise ProviderError(f"{self.key} simulated provider error")
        roll -= behavior.fail_rate
        if roll < behavior.invalid_json_rate:
            return LLMResponse(text='{"records": [ {"index": 0, ', latency=0.0)

        schema = json_schema or self._schema_from_prompt(user)
        properties = (schema or {}).get("properties", {})
        # Answer content depends only on the prompt, never on the call counter: with several threads the
        # call order varies, and the mock must still give the same dataset for the same seed.
        content_rng = random.Random(stable_seed(self.config.model_name, user))
        if "records" in properties:
            payload = self._enrich(user, properties["records"]["items"], content_rng, behavior)
        else:
            payload = self._profile(properties, content_rng)
        return LLMResponse(text=json.dumps(payload), latency=behavior.latency, prompt_tokens=len(user) // 4,
                           completion_tokens=len(json.dumps(payload)) // 4)

    @staticmethod
    def _schema_from_prompt(user: str) -> dict[str, Any] | None:
        marker = "matches this JSON Schema:\n"
        if marker in user:
            return json.loads(user.split(marker, 1)[1])
        return None

    def _enrich(self, user: str, item_schema: dict[str, Any], rng: random.Random,
                behavior: MockBehavior) -> dict[str, Any]:
        match = _RECORDS_BLOCK.search(user)
        rows = json.loads(match.group(1)) if match else []
        fields = {k: v for k, v in item_schema.get("properties", {}).items() if k != "index"}
        out = []
        for row in rows:
            if behavior.drop_rate and rng.random() < behavior.drop_rate:
                continue
            item: dict[str, Any] = {"index": row["index"]}
            local = random.Random(stable_seed(json.dumps(row, sort_keys=True), rng.random()))
            for name, prop in fields.items():
                item[name] = self._value(name, prop, row.get("context", {}), local)
            out.append(item)
        return {"records": out}

    @staticmethod
    def _value(name: str, prop: dict[str, Any], context: dict[str, Any], rng: random.Random) -> Any:
        kind = prop.get("type")
        if kind in ("number", "integer"):
            low = float(prop.get("minimum", 0))
            high = float(prop.get("maximum", low + 1000))
            value = rng.uniform(low, high)
            return int(value) if kind == "integer" else round(value, 2)
        if kind == "boolean":
            return rng.random() < 0.5
        if "enum" in prop:
            return rng.choice(prop["enum"])
        subject = next((str(v) for k, v in context.items() if isinstance(v, str) and k in
                        ("department", "category", "role", "product", "city")), "their field")
        text = (f"{rng.choice(_OPENERS)} {rng.choice(_CORES)} in {subject} focused on {rng.choice(_FOCUS)} "
                f"and {rng.choice(_FOCUS)}. {rng.choice(_EXTRAS)}.")
        limit = int(prop.get("maxLength", 400))
        return text[:limit]

    @staticmethod
    def _profile(properties: dict[str, Any], rng: random.Random) -> dict[str, Any]:
        out = {}
        for category in properties:
            base = 300_000 + (stable_seed(category) % 8) * 90_000
            low = round(base * rng.uniform(0.9, 1.1), -3)
            high = round(low * rng.uniform(2.6, 3.4), -3)
            out[category] = {"min": low, "max": high, "typical": round(low + (high - low) * 0.38, -3)}
        return out

    def check_availability(self) -> AvailabilityResult:
        """Report the configured mock availability."""
        behavior = self._behavior()
        return AvailabilityResult(behavior.available, "mock model" if behavior.available else "mock unavailable")

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return deterministic pseudo-embeddings (for tests)."""
        from app.guardrails.semantic_deduplication import HashingEmbedder  # pylint: disable=import-outside-toplevel

        return HashingEmbedder().embed(texts).tolist()
