"""Adapter for a local Ollama server (``/api/chat``, ``/api/tags``, ``/api/embed``).

Models with a ``-cloud``/``:cloud`` suffix are forwarded by a signed-in local
server to Ollama Cloud; those do not support schema-constrained output, so
the registry marks them ``json_mode``.
"""

from __future__ import annotations

from typing import Any

from app.config.model_registry import ModelConfig
from app.config.settings import Settings
from app.core.exceptions import InvalidModelOutputError, LLMError, ProviderError
from app.models.base_adapter import AvailabilityResult, BaseLLMAdapter, HTTPAdapterMixin, LLMResponse
from app.monitoring.logger import get_logger

logger = get_logger(__name__)
_UNLOAD_TIMEOUT_SECONDS = 5.0


def unload_local_models(chain: list[ModelConfig], settings: Settings, used: set[str]) -> None:
    """Unload the local (non-cloud) Ollama models of ``chain`` the job used, freeing their RAM now.

    Ollama otherwise keeps a model resident for ``OLLAMA_KEEP_ALIVE`` (5 minutes by default).
    Models the job never called are left alone, and a short timeout keeps a stopped server from
    delaying the end of the job. Failures are logged and ignored: unloading is a courtesy, never
    a reason to fail a job.
    """
    for config in chain:
        if config.provider != "ollama_local" or config.execution != "local" or config.key not in used:
            continue
        adapter = OllamaLocalAdapter(config.model_copy(update={"timeout": _UNLOAD_TIMEOUT_SECONDS}), settings)
        try:
            adapter.unload()
            logger.info("Unloaded %s from Ollama", config.key)
        except LLMError as exc:
            logger.debug("Could not unload %s: %s", config.key, exc)
        finally:
            adapter.close()


class OllamaLocalAdapter(HTTPAdapterMixin, BaseLLMAdapter):
    """Talks to ``$OLLAMA_HOST`` (default ``http://localhost:11434``)."""

    def __init__(self, config: ModelConfig, settings: Settings) -> None:
        super().__init__(config, settings)
        self._client = None

    def _base_url(self) -> str:
        return (self.config.endpoint or self.settings.ollama_host).rstrip("/")

    def _headers(self) -> dict[str, str]:
        return {}

    def _client_(self):  # noqa: ANN202 - small private helper
        return self._http(self._base_url(), self._headers())

    def _chat(self, system: str, user: str, json_schema: dict[str, Any] | None, json_mode: bool) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.config.model_id,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_output_tokens,
                "num_ctx": min(self.config.context_window, 8192),
            },
        }
        if self.config.seed is not None:
            payload["options"]["seed"] = self.config.seed
        if json_schema is not None:
            payload["format"] = json_schema
        elif json_mode:
            payload["format"] = "json"
        if self.config.thinking:
            payload["think"] = False
        body, latency = self._post(self._client_(), "/api/chat", payload)
        message = body.get("message") or {}
        text = message.get("content") or ""
        if not text.strip():
            raise InvalidModelOutputError(f"{self.key} returned an empty message")
        return LLMResponse(text=text, latency=latency,
                           prompt_tokens=int(body.get("prompt_eval_count") or 0),
                           completion_tokens=int(body.get("eval_count") or 0))

    def list_models(self) -> list[dict[str, Any]]:
        """Return the models installed on the server (``/api/tags``)."""
        return list(self._get(self._client_(), "/api/tags").get("models", []))

    @property
    def base_url(self) -> str:
        """Server URL this adapter talks to."""
        return self._base_url()

    def check_availability(self) -> AvailabilityResult:
        """Check that the server is up and the model is installed."""
        try:
            models = self.list_models()
        except LLMError as exc:
            return AvailabilityResult(False, f"Ollama server not reachable at {self._base_url()}: {exc}")
        return self.availability_from(models)

    def availability_from(self, models: list[dict[str, Any]]) -> AvailabilityResult:
        """Decide availability from an already fetched ``/api/tags`` listing."""
        for model in models:
            if model.get("name") == self.config.model_id or model.get("model") == self.config.model_id:
                details = model.get("details") or {}
                return AvailabilityResult(True, "installed", {
                    "size_bytes": model.get("size"), "parameter_size": details.get("parameter_size"),
                    "context_length": details.get("context_length"), "capabilities": model.get("capabilities"),
                    "remote_host": model.get("remote_host"),
                })
        return AvailabilityResult(False, f"Model {self.config.model_id!r} is not installed; run "
                                         f"'ollama pull {self.config.model_id}'")

    def loaded_models(self) -> list[dict[str, Any]]:
        """Return models currently loaded in memory (``/api/ps``)."""
        try:
            return list(self._get(self._client_(), "/api/ps").get("models", []))
        except LLMError:
            return []

    def unload(self) -> None:
        """Ask the server to evict this model from memory now (``keep_alive: 0``)."""
        self._post(self._client_(), "/api/generate", {"model": self.config.model_id, "keep_alive": 0})

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed ``texts`` with this (embedding) model via ``/api/embed``."""
        body, _ = self._post(self._client_(), "/api/embed", {"model": self.config.model_id, "input": texts})
        vectors = body.get("embeddings")
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise ProviderError(f"{self.key} returned malformed embeddings")
        return vectors
