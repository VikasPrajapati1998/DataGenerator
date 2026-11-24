"""Centralised model registry.

Model definitions live in ``app/config/models.json`` (overridable through
``SDE_MODEL_REGISTRY``). A model is addressed by the key
``"<provider>/<model_name>"``; unknown models of a known provider are created
on the fly from provider defaults, so users can switch models without editing
the registry.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.core.exceptions import ConfigurationError
from app.utils.retry import RetryPolicy

StructuredMode = Literal["json_schema", "json_mode", "prompt_only"]
Availability = Literal["unknown", "available", "unavailable"]


class RetrySettings(BaseModel):
    """Serializable retry policy for one model."""

    max_attempts: int = Field(default=3, ge=1, le=10)
    base_delay: float = Field(default=1.0, ge=0)
    max_delay: float = Field(default=30.0, ge=0)

    def to_policy(self) -> RetryPolicy:
        """Return the equivalent runtime :class:`RetryPolicy`."""
        return RetryPolicy(max_attempts=self.max_attempts, base_delay=self.base_delay, max_delay=self.max_delay)


class ModelConfig(BaseModel):
    """Everything needed to talk to one model of one provider.

    Attributes:
        provider: Provider adapter name (``ollama_local``, ``ollama_cloud``,
            ``openai_compatible``, ``nvidia_nim``, ``openrouter``, ``mock``).
        model_name: Name the user refers to the model by.
        model_id: Identifier sent to the provider API.
        execution: Whether inference runs locally or in the cloud.
        endpoint: Base URL of the provider API; ``None`` uses the provider default.
        auth_env: Environment variable that holds the API key (never the key itself).
        context_window: Maximum context tokens.
        max_output_tokens: Maximum tokens generated per request.
        temperature: Sampling temperature.
        seed: Sampling seed sent to the provider (``None`` = provider default, random).
        timeout: Request timeout in seconds.
        retry: Retry policy for transient failures.
        max_concurrency: Concurrent in-flight requests allowed for this model.
        requests_per_minute: Client-side rate limit; ``None`` disables it.
        structured_output: How structured output is requested.
        thinking: Whether the model has a thinking mode (it is disabled for generation).
        approx_memory_mb: Approximate RAM/VRAM the loaded model needs (local only).
        availability: Last known availability status (runtime only).
        notes: Free-text provenance notes.
    """

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    provider: str
    model_name: str
    model_id: str
    execution: Literal["local", "cloud"] = "local"
    endpoint: str | None = None
    auth_env: str | None = None
    context_window: int = 8192
    max_output_tokens: int = 2048
    temperature: float = Field(default=0.8, ge=0, le=2)
    seed: int | None = None
    timeout: float = Field(default=120.0, gt=0)
    retry: RetrySettings = Field(default_factory=RetrySettings)
    max_concurrency: int | None = Field(default=None, ge=1)
    requests_per_minute: int | None = Field(default=None, ge=1)
    structured_output: StructuredMode = "json_schema"
    thinking: bool = False
    approx_memory_mb: int | None = None
    availability: Availability = "unknown"
    notes: str = ""

    @property
    def key(self) -> str:
        """Registry key ``provider/model_name``."""
        return f"{self.provider}/{self.model_name}"

    def with_retry_cap(self, max_retries: int | None) -> ModelConfig:
        """Return a copy whose LLM call attempts are at most ``1 + max_retries``.

        Args:
            max_retries: Global retry cap (``SDE_MAX_RETRIES``), or ``None`` for no cap.

        Returns:
            The (possibly) capped configuration.
        """
        if max_retries is None or self.retry.max_attempts <= max_retries + 1:
            return self
        return self.model_copy(update={"retry": self.retry.model_copy(update={"max_attempts": max_retries + 1})})

    def public_dict(self) -> dict[str, Any]:
        """Return a log/JSON-safe description (contains no secrets by design)."""
        return self.model_dump(mode="json")


_PROVIDER_DEFAULTS: dict[str, dict[str, Any]] = {
    "ollama_local": {"execution": "local", "structured_output": "json_schema"},
    "ollama_cloud": {"execution": "cloud", "auth_env": "OLLAMA_API_KEY", "structured_output": "json_mode",
                     "requests_per_minute": 30},
    "nvidia_nim": {"execution": "cloud", "auth_env": "NVIDIA_API_KEY", "structured_output": "json_mode",
                   "endpoint": "https://integrate.api.nvidia.com/v1", "requests_per_minute": 35},
    "openrouter": {"execution": "cloud", "auth_env": "OPENROUTER_API_KEY", "structured_output": "json_mode",
                   "endpoint": "https://openrouter.ai/api/v1", "requests_per_minute": 20},
    "openai_compatible": {"execution": "cloud", "structured_output": "json_mode"},
    "mock": {"execution": "local", "structured_output": "json_schema", "timeout": 10},
}


class ModelRegistry:
    """Thread-safe registry of :class:`ModelConfig` entries.

    Args:
        entries: Initial model configurations.
        default_key: Key used when a request does not name a model.
    """

    def __init__(self, entries: list[ModelConfig], default_key: str) -> None:
        self._lock = threading.Lock()
        self._models: dict[str, ModelConfig] = {entry.key: entry for entry in entries}
        self.default_key = default_key

    @classmethod
    def from_file(cls, path: Path, default_key: str) -> ModelRegistry:
        """Load a registry from a JSON file.

        Args:
            path: JSON file with a top-level ``models`` list.
            default_key: Default model key.

        Returns:
            The loaded registry.

        Raises:
            ConfigurationError: If the file is missing or malformed.
        """
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            entries = [ModelConfig(**item) for item in payload["models"]]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ConfigurationError(f"Cannot load model registry {path}: {exc}") from exc
        return cls(entries, payload.get("default", default_key))

    def keys(self) -> list[str]:
        """Return all registered model keys."""
        with self._lock:
            return sorted(self._models)

    def all(self) -> list[ModelConfig]:
        """Return copies of all registered models."""
        with self._lock:
            return [model.model_copy() for model in self._models.values()]

    def register(self, config: ModelConfig) -> None:
        """Add or replace a model definition.

        Args:
            config: The model to register.
        """
        with self._lock:
            self._models[config.key] = config

    def get(self, key: str) -> ModelConfig:
        """Return the model registered under ``key``.

        Args:
            key: ``provider/model_name``.

        Returns:
            A copy of the model configuration.

        Raises:
            ConfigurationError: If the key is unknown.
        """
        with self._lock:
            if key not in self._models:
                raise ConfigurationError(f"Model {key!r} is not registered")
            return self._models[key].model_copy(deep=True)

    def resolve(self, provider: str | None, model_name: str | None) -> ModelConfig:
        """Return the model for ``provider``/``model_name``, creating it if needed.

        Args:
            provider: Provider adapter name; ``None`` uses the default model.
            model_name: Model name; ``None`` uses the provider's first registered
                model or the global default.

        Returns:
            The resolved model configuration.

        Raises:
            ConfigurationError: If the provider is unknown.
        """
        if not provider and not model_name:
            return self.get(self.default_key)
        if model_name and "/" in model_name and not provider:
            provider, model_name = model_name.split("/", 1)
        provider = provider or self.default_key.split("/", 1)[0]
        if provider == "ollama":
            provider = "ollama_local"
        if provider not in _PROVIDER_DEFAULTS:
            raise ConfigurationError(
                f"Unknown model provider {provider!r}; supported: {sorted(_PROVIDER_DEFAULTS)}"
            )
        if model_name is None:
            candidates = [k for k in self.keys() if k.startswith(f"{provider}/")]
            if not candidates:
                raise ConfigurationError(f"No model registered for provider {provider!r}")
            return self.get(candidates[0])
        key = f"{provider}/{model_name}"
        with self._lock:
            if key in self._models:
                return self._models[key].model_copy(deep=True)
        config = ModelConfig(provider=provider, model_name=model_name, model_id=model_name,
                             **_PROVIDER_DEFAULTS[provider])
        if provider == "ollama_local" and model_name.endswith(("-cloud", ":cloud")):
            config.execution = "cloud"
            config.structured_output = "json_mode"
        self.register(config)
        return config.model_copy(deep=True)

    def set_availability(self, key: str, status: Availability) -> None:
        """Record the latest availability status of a model.

        Args:
            key: Model key.
            status: New status.
        """
        with self._lock:
            if key in self._models:
                self._models[key].availability = status


def load_registry() -> ModelRegistry:
    """Load the registry configured in the application settings."""
    from app.config.settings import get_settings  # pylint: disable=import-outside-toplevel

    settings = get_settings()
    return ModelRegistry.from_file(settings.model_registry_file, settings.default_model)
