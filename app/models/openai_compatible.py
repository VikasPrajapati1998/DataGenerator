"""Adapter for OpenAI-compatible chat APIs.

Covers NVIDIA's hosted API (Nemotron, ``https://integrate.api.nvidia.com/v1``),
OpenRouter (e.g. Reflection AI Beam) and any self-hosted server that speaks
the ``/v1/chat/completions`` protocol (vLLM, llama.cpp server, LM Studio).
"""

from __future__ import annotations

from typing import Any

from app.core.exceptions import ConfigurationError, InvalidModelOutputError, LLMError
from app.models.base_adapter import AvailabilityResult, BaseLLMAdapter, HTTPAdapterMixin, LLMResponse


class OpenAICompatibleAdapter(HTTPAdapterMixin, BaseLLMAdapter):
    """Generic ``/chat/completions`` adapter."""

    _client = None

    def _client_(self):  # noqa: ANN202 - small private helper
        if not self.config.endpoint:
            raise ConfigurationError(f"{self.key}: an endpoint is required for OpenAI-compatible providers")
        headers = {}
        secret = self.settings.secret_for(self.config.auth_env)
        if secret:
            headers["Authorization"] = f"Bearer {secret.get_secret_value()}"
        return self._http(self.config.endpoint.rstrip("/"), headers)

    def _chat(self, system: str, user: str, json_schema: dict[str, Any] | None, json_mode: bool) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.config.model_id,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_output_tokens,
        }
        if self.config.seed is not None:
            payload["seed"] = self.config.seed
        if json_schema is not None:
            payload["response_format"] = {"type": "json_schema",
                                          "json_schema": {"name": "records", "schema": json_schema}}
        elif json_mode:
            payload["response_format"] = {"type": "json_object"}
        body, latency = self._post(self._client_(), "/chat/completions", payload)
        try:
            text = body["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise InvalidModelOutputError(f"{self.key} returned an unexpected response shape") from exc
        usage = body.get("usage") or {}
        return LLMResponse(text=text, latency=latency, prompt_tokens=int(usage.get("prompt_tokens") or 0),
                           completion_tokens=int(usage.get("completion_tokens") or 0))

    def check_availability(self) -> AvailabilityResult:
        """Check credentials and that ``/models`` lists the model."""
        if self.config.auth_env and not self.settings.secret_for(self.config.auth_env):
            return AvailabilityResult(False, f"Environment variable {self.config.auth_env} is not set")
        try:
            body = self._get(self._client_(), "/models")
        except (LLMError, ConfigurationError) as exc:
            return AvailabilityResult(False, f"Provider not reachable: {exc}")
        ids = {item.get("id") for item in body.get("data", []) if isinstance(item, dict)}
        if self.config.model_id in ids:
            return AvailabilityResult(True, "listed by provider")
        return AvailabilityResult(False, f"Model {self.config.model_id!r} not listed by {self.config.endpoint}")
