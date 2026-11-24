"""Provider factory: maps provider names to adapter classes.

New providers register themselves with :func:`register_provider`; nothing in
the graph or the workers changes when a provider is added.
"""

from __future__ import annotations

from collections.abc import Callable

from app.config.model_registry import ModelConfig
from app.config.settings import Settings, get_settings
from app.core.exceptions import ConfigurationError
from app.models.base_adapter import BaseLLMAdapter
from app.models.mock_adapter import MockLLMAdapter
from app.models.ollama_cloud import OllamaCloudAdapter
from app.models.ollama_local import OllamaLocalAdapter
from app.models.openai_compatible import OpenAICompatibleAdapter

AdapterClass = type[BaseLLMAdapter]
_PROVIDERS: dict[str, AdapterClass] = {
    "ollama_local": OllamaLocalAdapter,
    "ollama_cloud": OllamaCloudAdapter,
    "openai_compatible": OpenAICompatibleAdapter,
    "nvidia_nim": OpenAICompatibleAdapter,
    "openrouter": OpenAICompatibleAdapter,
    "mock": MockLLMAdapter,
}


def register_provider(name: str) -> Callable[[AdapterClass], AdapterClass]:
    """Class decorator that registers an adapter under ``name``.

    Args:
        name: Provider name used in model configurations.

    Returns:
        The decorator.
    """
    def decorator(cls: AdapterClass) -> AdapterClass:
        _PROVIDERS[name] = cls
        return cls
    return decorator


def supported_providers() -> list[str]:
    """Return the registered provider names."""
    return sorted(_PROVIDERS)


def create_adapter(config: ModelConfig, settings: Settings | None = None) -> BaseLLMAdapter:
    """Instantiate the adapter for ``config``.

    Args:
        config: Model configuration.
        settings: Application settings; defaults to :func:`get_settings`.

    Returns:
        A ready-to-use adapter.

    Raises:
        ConfigurationError: If the provider is unknown.
    """
    try:
        cls = _PROVIDERS[config.provider]
    except KeyError as exc:
        raise ConfigurationError(f"No adapter registered for provider {config.provider!r}") from exc
    return cls(config, settings or get_settings())
