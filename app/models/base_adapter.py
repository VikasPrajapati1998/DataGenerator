"""Common interface every LLM provider adapter implements.

Workers never talk to a provider directly: they call
:meth:`BaseLLMAdapter.generate_json`, which handles structured-output modes,
JSON extraction and error normalisation identically for every provider.
"""

from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import httpx

from app.config.model_registry import ModelConfig
from app.config.settings import Settings
from app.monitoring.logger import redact
from app.core.exceptions import (
    ContextLengthError,
    InvalidModelOutputError,
    ModelUnavailableError,
    ProviderError,
    QuotaExceededError,
    ProviderTimeoutError,
    RateLimitError,
)

_CONTEXT_WORDS = ("context length", "context window", "maximum context", "context_length_exceeded",
                  "too many tokens", "prompt is too long", "input length")
_QUOTA_WORDS = ("credit", "quota", "billing", "subscribe", "insufficient_quota", "payment required")
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


@dataclass
class LLMResponse:
    """Raw provider answer plus usage information.

    Attributes:
        text: Text content returned by the model.
        latency: Wall-clock seconds the request took.
        prompt_tokens: Prompt tokens reported by the provider, if any.
        completion_tokens: Completion tokens reported by the provider, if any.
    """

    text: str
    latency: float
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class AvailabilityResult:
    """Outcome of a model availability probe.

    Attributes:
        available: Whether the model can serve requests.
        detail: Human-readable explanation.
        metadata: Provider-specific facts (size, context length, capabilities...).
    """

    available: bool
    detail: str
    metadata: dict[str, Any] = field(default_factory=dict)


def extract_json(text: str) -> Any:
    """Extract the first JSON object or array from model output.

    Handles ``<think>`` blocks, Markdown fences and leading/trailing prose.

    Args:
        text: Raw model output.

    Returns:
        The parsed JSON value.

    Raises:
        InvalidModelOutputError: If no valid JSON can be found.
    """
    cleaned = _THINK_BLOCK.sub("", text).strip()
    if fence := _FENCE.search(cleaned):
        cleaned = fence.group(1).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for index, char in enumerate(cleaned):
        if char in "{[":
            try:
                value, _ = decoder.raw_decode(cleaned[index:])
                return value
            except json.JSONDecodeError:
                continue
    raise InvalidModelOutputError(f"Model output is not valid JSON: {cleaned[:200]!r}")


class BaseLLMAdapter(ABC):
    """Template for provider adapters.

    Args:
        config: Model configuration from the registry.
        settings: Application settings (endpoints and secrets).
    """

    def __init__(self, config: ModelConfig, settings: Settings) -> None:
        self.config = config
        self.settings = settings

    @property
    def key(self) -> str:
        """Registry key of the model this adapter serves."""
        return self.config.key

    def generate_json(self, system: str, user: str, json_schema: dict[str, Any]) -> tuple[Any, LLMResponse]:
        """Ask the model for JSON matching ``json_schema``.

        Uses schema-constrained decoding when the model supports it, JSON
        mode otherwise, and always parses/validates the result client-side.

        Args:
            system: System prompt.
            user: User prompt.
            json_schema: JSON Schema the answer must follow.

        Returns:
            The parsed JSON value and the raw response.

        Raises:
            InvalidModelOutputError: If the output is not valid JSON.
            LLMError: Any normalised provider failure.
        """
        mode = self.config.structured_output
        if mode != "json_schema":
            user = (f"{user}\n\nRespond with JSON only (no prose, no Markdown) that matches this JSON Schema:\n"
                    f"{json.dumps(json_schema, separators=(',', ':'))}")
        response = self._chat(system, user, json_schema if mode == "json_schema" else None, mode != "prompt_only")
        return extract_json(response.text), response

    @abstractmethod
    def _chat(self, system: str, user: str, json_schema: dict[str, Any] | None, json_mode: bool) -> LLMResponse:
        """Send one chat request and return the text answer."""

    @abstractmethod
    def check_availability(self) -> AvailabilityResult:
        """Probe whether the configured model can serve requests."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return embeddings for ``texts`` (optional capability).

        Args:
            texts: Texts to embed.

        Returns:
            One vector per text.

        Raises:
            NotImplementedError: If the provider has no embedding endpoint.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support embeddings")

    def close(self) -> None:
        """Release network resources (no-op by default)."""


class HTTPAdapterMixin:
    """Shared ``httpx`` plumbing and error normalisation for HTTP providers."""

    config: ModelConfig
    _client: httpx.Client | None = None

    def _http(self, base_url: str, headers: dict[str, str] | None = None) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(base_url=base_url, headers=headers or {},
                                        timeout=httpx.Timeout(self.config.timeout, connect=10.0))
        return self._client

    def _post(self, client: httpx.Client, path: str, payload: dict[str, Any]) -> tuple[dict[str, Any], float]:
        started = time.perf_counter()
        try:
            response = client.post(path, json=payload)
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(f"{self.config.key} timed out after {self.config.timeout}s") from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"{self.config.key} request failed: {type(exc).__name__}: {exc}") from exc
        latency = time.perf_counter() - started
        self._raise_for_status(response)
        try:
            body = response.json()
        except ValueError as exc:
            raise ProviderError(f"{self.config.key} returned a non-JSON HTTP body") from exc
        if not isinstance(body, dict):
            raise ProviderError(f"{self.config.key} returned an unexpected HTTP body")
        return body, latency

    def _get(self, client: httpx.Client, path: str) -> dict[str, Any]:
        try:
            response = client.get(path, timeout=10.0)
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(f"{self.config.key}: GET {path} timed out") from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"{self.config.key} is unreachable: {type(exc).__name__}") from exc
        self._raise_for_status(response)
        try:
            body = response.json()
        except ValueError as exc:
            raise ProviderError(f"{self.config.key} returned a non-JSON body for GET {path}") from exc
        if not isinstance(body, dict):
            raise ProviderError(f"{self.config.key} returned an unexpected body for GET {path}")
        return body

    def _raise_for_status(self, response: httpx.Response) -> None:
        if response.status_code < 400:
            return
        body = redact(response.text[:300])  # provider error bodies can echo request headers
        if response.status_code == 402 or (response.status_code in (403, 429)
                                           and any(w in body.lower() for w in _QUOTA_WORDS)):
            # Out of credits/quota is permanent for this job: never worth a retry.
            raise QuotaExceededError(f"{self.config.key}: quota or credits exhausted "
                                     f"(HTTP {response.status_code}): {body}")
        if response.status_code in (400, 413, 422, 500) and any(w in body.lower() for w in _CONTEXT_WORDS):
            raise ContextLengthError(f"{self.config.key}: request exceeds the context window "
                                     f"(HTTP {response.status_code}): {body}")
        if response.status_code == 429:
            retry_after = response.headers.get("retry-after", "")
            try:
                wait = float(retry_after) if retry_after else None
            except ValueError:
                wait = None  # HTTP-date form: fall back to exponential backoff
            raise RateLimitError(f"{self.config.key} rate limited: {body}", wait)
        if response.status_code == 503:
            raise ProviderError(f"{self.config.key} is overloaded (HTTP 503, queue full?): {body}")
        if response.status_code in (408, 504):
            raise ProviderTimeoutError(f"{self.config.key} gateway timeout ({response.status_code})")
        if response.status_code in (401, 403, 404):
            raise ModelUnavailableError(f"{self.config.key} unavailable (HTTP {response.status_code}): {body}")
        raise ProviderError(f"{self.config.key} HTTP {response.status_code}: {body}")

    def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client is not None:
            self._client.close()
            self._client = None
