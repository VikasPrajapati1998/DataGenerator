"""Environment-driven application settings.

Values come from environment variables (optionally loaded from a ``.env``
file). Secrets are held as :class:`pydantic.SecretStr` so they are never
printed by accident.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr

from app.core.exceptions import ConfigurationError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAX_RETRIES_LIMIT = 20


def _env_int(name: str) -> int | None:
    raw = os.getenv(name)
    return int(raw) if raw and raw.strip().isdigit() else None


def _env_bounded_int(name: str, low: int, high: int) -> int | None:
    """Read an optional integer env var, rejecting malformed or out-of-range values loudly."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    try:
        value = int(raw.strip())
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer between {low} and {high}, got {raw!r}") from exc
    if not low <= value <= high:
        raise ConfigurationError(f"{name} must be between {low} and {high}, got {value}")
    return value


def _env_secret(name: str) -> SecretStr | None:
    raw = os.getenv(name)
    return SecretStr(raw) if raw else None


class Settings(BaseModel):
    """Process-wide configuration resolved from the environment.

    Attributes:
        output_dir: Root directory for job folders.
        log_dir: Root directory for log files.
        log_level: Default log level.
        model_registry_file: JSON file with model definitions.
        ollama_host: Base URL of the local Ollama server.
        ollama_cloud_host: Base URL of Ollama Cloud.
        ollama_api_key: API key for direct Ollama Cloud access.
        ollama_num_parallel: Parallel requests the local Ollama server accepts.
        nvidia_api_key: API key for NVIDIA's hosted OpenAI-compatible API.
        openrouter_api_key: API key for OpenRouter's OpenAI-compatible API.
        default_model: Registry key used when a request names no model.
        model_pool: Default model pool (``SDE_MODEL_POOL``, comma-separated keys)
            used when a request has no ``model_pool``.
        default_batch_size: Fixed target records per task (``SDE_BATCH_SIZE``); ``None``
            lets the planner choose a batch size at run time.
        max_workers: Optional hard cap on logical core queues.
        max_retries: Global retry cap (``SDE_MAX_RETRIES``): retries per LLM call,
            regeneration rounds per task, failed-task retry rounds and repair
            rounds can each be at most this. ``None`` keeps the per-setting defaults.
        ram_check: ``SDE_RAM_CHECK``: ``warn`` (default) runs with fewer queues when
            RAM is tight, ``strict`` stops the job, ``off`` skips the RAM budget.
        max_records_per_job: ``SDE_MAX_RECORDS_PER_JOB``: larger requests are split
            into sequential sub-jobs of at most this size (0 disables splitting).
        stall_timeout: ``SDE_STALL_TIMEOUT``: seconds without any progress (no record, LLM
            answer or finished task) after which the watchdog stops the job (0 disables).
        task_timeout: ``SDE_TASK_TIMEOUT``: seconds one task may run in a worker process
            before the process is terminated (0 disables).
        llm_slot_timeout: ``SDE_LLM_SLOT_TIMEOUT``: max seconds to wait for a free model slot.
        unload_models: ``SDE_UNLOAD_MODELS``: ask Ollama to unload the local models a job used
            when it ends, so their RAM is released immediately.
        max_eta_hours: ``SDE_MAX_ETA_HOURS``: refuse to start a job whose projected run time is
            longer (0 disables; ``--force`` overrides).
        llm_mode: ``SDE_LLM_MODE``: ``per_record`` (one LLM answer per record), ``pooled`` (a text
            pool built once and recombined) or ``auto`` (pooled from ``pooled_min_records``).
        pooled_min_records: ``SDE_POOLED_MIN_RECORDS``: size from which ``auto`` uses pooled mode.
        pool_build_minutes: ``SDE_POOL_BUILD_MINUTES``: time budget for building the text pool.
        pool_max_values: ``SDE_POOL_MAX_VALUES``: maximum number of pooled LLM answers.
        pool_grow: ``SDE_POOL_GROW``: keep adding answers to the pool in the background while
            records are generated, until ``pool_max_values`` (later sub-jobs use the larger pool).
        max_invalid_fraction: ``SDE_MAX_INVALID_FRACTION``: share of a task's records (at least
            10) that may still be invalid after the regeneration rounds and get extra rescue
            rounds instead of failing the whole task.
        memory_per_worker_mb: Estimated RAM each queue needs (excluding the model).
    """

    output_dir: Path = Field(default_factory=lambda: PROJECT_ROOT / "output")
    log_dir: Path = Field(default_factory=lambda: PROJECT_ROOT / "logs")
    log_level: str = "INFO"
    model_registry_file: Path = Field(default_factory=lambda: PROJECT_ROOT / "app" / "config" / "models.json")
    # 127.0.0.1 rather than "localhost": on Windows "localhost" tries IPv6 first, and every failed
    # connection then costs about 2 s extra.
    ollama_host: str = "http://127.0.0.1:11434"
    ollama_cloud_host: str = "https://ollama.com"
    ollama_api_key: SecretStr | None = None
    ollama_num_parallel: int | None = None
    nvidia_api_key: SecretStr | None = None
    openrouter_api_key: SecretStr | None = None
    default_model: str = "ollama_local/qwen3:0.6b"
    model_pool: list[str] = Field(default_factory=list)
    default_batch_size: int | None = None
    max_workers: int | None = None
    max_retries: int | None = Field(default=None, ge=0, le=20)
    ram_check: Literal["strict", "warn", "off"] = "warn"
    max_records_per_job: int = Field(default=100_000, ge=0)
    stall_timeout: int = Field(default=900, ge=0)
    task_timeout: int = Field(default=7200, ge=0)
    llm_slot_timeout: int = Field(default=900, ge=1)
    unload_models: bool = True
    max_eta_hours: float = Field(default=6.0, ge=0)
    llm_mode: Literal["auto", "per_record", "pooled"] = "auto"
    pooled_min_records: int = Field(default=10_000, ge=1)
    pool_build_minutes: float = Field(default=20.0, gt=0)
    pool_max_values: int = Field(default=2000, ge=10)
    pool_grow: bool = True
    max_invalid_fraction: float = Field(default=0.001, ge=0, le=0.5)
    memory_per_worker_mb: int = 150

    def secret_for(self, env_name: str | None) -> SecretStr | None:
        """Return the secret stored in environment variable ``env_name``.

        Args:
            env_name: Name of the environment variable, or ``None``.

        Returns:
            The secret, or ``None`` when unset.
        """
        if not env_name:
            return None
        return _env_secret(env_name)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings once per process from the environment and ``.env``."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    values: dict[str, object] = {}
    mapping = {
        "SDE_OUTPUT_DIR": "output_dir", "SDE_LOG_DIR": "log_dir", "SDE_LOG_LEVEL": "log_level",
        "SDE_MODEL_REGISTRY": "model_registry_file", "OLLAMA_HOST": "ollama_host",
        "OLLAMA_CLOUD_HOST": "ollama_cloud_host", "SDE_DEFAULT_MODEL": "default_model",
    }
    for env_name, field in mapping.items():
        if os.getenv(env_name):
            values[field] = os.environ[env_name]
    if values.get("ollama_host") and not str(values["ollama_host"]).startswith("http"):
        values["ollama_host"] = f"http://{values['ollama_host']}"
    if sys.platform == "win32" and "//localhost" in str(values.get("ollama_host", "")):
        # Windows resolves "localhost" to IPv6 first; Ollama listens on 127.0.0.1, so every
        # connection (and every failed one) would pay an extra ~2 s.
        values["ollama_host"] = str(values["ollama_host"]).replace("//localhost", "//127.0.0.1")
    values["ollama_api_key"] = _env_secret("OLLAMA_API_KEY")
    values["nvidia_api_key"] = _env_secret("NVIDIA_API_KEY")
    values["openrouter_api_key"] = _env_secret("OPENROUTER_API_KEY")
    values["ollama_num_parallel"] = _env_int("OLLAMA_NUM_PARALLEL")
    values["max_retries"] = _env_bounded_int("SDE_MAX_RETRIES", 0, MAX_RETRIES_LIMIT)
    for env_name, field, low in (("SDE_STALL_TIMEOUT", "stall_timeout", 0), ("SDE_TASK_TIMEOUT", "task_timeout", 0),
                                 ("SDE_LLM_SLOT_TIMEOUT", "llm_slot_timeout", 1)):
        if (seconds := _env_bounded_int(env_name, low, 7 * 24 * 3600)) is not None:
            values[field] = seconds
    for env_name, field in (("SDE_UNLOAD_MODELS", "unload_models"), ("SDE_POOL_GROW", "pool_grow")):
        if flag := os.getenv(env_name, "").strip().lower():
            if flag not in ("1", "0", "true", "false", "yes", "no"):
                raise ConfigurationError(f"{env_name} must be true or false, got {flag!r}")
            values[field] = flag in ("1", "true", "yes")
    if (fraction := os.getenv("SDE_MAX_INVALID_FRACTION", "").strip()):
        try:
            values["max_invalid_fraction"] = float(fraction)
        except ValueError as exc:
            raise ConfigurationError(
                f"SDE_MAX_INVALID_FRACTION must be a fraction such as 0.001, got {fraction!r}") from exc
        if not 0 <= values["max_invalid_fraction"] <= 0.5:
            raise ConfigurationError(f"SDE_MAX_INVALID_FRACTION must be between 0 and 0.5, got {fraction!r}")
    if (eta := os.getenv("SDE_MAX_ETA_HOURS", "").strip()):
        try:
            values["max_eta_hours"] = float(eta)
        except ValueError as exc:
            raise ConfigurationError(f"SDE_MAX_ETA_HOURS must be a number of hours, got {eta!r}") from exc
    if (mode := os.getenv("SDE_LLM_MODE", "").strip().lower()):
        if mode not in ("auto", "per_record", "pooled"):
            raise ConfigurationError(f"SDE_LLM_MODE must be auto, per_record or pooled, got {mode!r}")
        values["llm_mode"] = mode
    if (pooled := _env_bounded_int("SDE_POOLED_MIN_RECORDS", 1, 1_000_000_000)) is not None:
        values["pooled_min_records"] = pooled
    if (minutes := _env_bounded_int("SDE_POOL_BUILD_MINUTES", 1, 24 * 60)) is not None:
        values["pool_build_minutes"] = float(minutes)
    if (pool_values := _env_bounded_int("SDE_POOL_MAX_VALUES", 10, 1_000_000)) is not None:
        values["pool_max_values"] = pool_values
    if (per_job := _env_bounded_int("SDE_MAX_RECORDS_PER_JOB", 0, 50_000_000)) is not None:
        values["max_records_per_job"] = per_job
    if ram_check := os.getenv("SDE_RAM_CHECK", "").strip().lower():
        if ram_check not in ("strict", "warn", "off"):
            raise ConfigurationError(f"SDE_RAM_CHECK must be strict, warn or off, got {ram_check!r}")
        values["ram_check"] = ram_check
    if pool := os.getenv("SDE_MODEL_POOL"):
        values["model_pool"] = [key.strip() for key in pool.split(",") if key.strip()]
    if (batch := _env_int("SDE_BATCH_SIZE")) is not None:
        values["default_batch_size"] = batch
    if (workers := _env_int("SDE_MAX_WORKERS")) is not None:
        values["max_workers"] = workers
    return Settings(**values)
