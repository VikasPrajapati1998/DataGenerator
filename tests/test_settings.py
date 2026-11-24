"""Environment settings and the global retry cap (SDE_MAX_RETRIES)."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config.guardrail_config import GuardrailConfig
from app.config.model_registry import ModelConfig
from app.config.settings import Settings, get_settings
from app.core.exceptions import ConfigurationError
from app.models.mock_adapter import MockBehavior, MockLLMAdapter


@pytest.fixture
def fresh_settings():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_max_retries_is_read_from_the_environment(monkeypatch, fresh_settings) -> None:
    monkeypatch.setenv("SDE_MAX_RETRIES", "1")
    assert get_settings().max_retries == 1


@pytest.mark.parametrize("raw", ["abc", "-1", "99", "1.5"])
def test_invalid_max_retries_is_rejected(monkeypatch, fresh_settings, raw: str) -> None:
    monkeypatch.setenv("SDE_MAX_RETRIES", raw)
    with pytest.raises(ConfigurationError, match="SDE_MAX_RETRIES"):
        get_settings()


def test_ram_check_mode_is_validated(monkeypatch, fresh_settings) -> None:
    monkeypatch.setenv("SDE_RAM_CHECK", "Strict")
    assert get_settings().ram_check == "strict"
    get_settings.cache_clear()
    monkeypatch.setenv("SDE_RAM_CHECK", "sometimes")
    with pytest.raises(ConfigurationError, match="SDE_RAM_CHECK"):
        get_settings()


def test_cap_bounds_every_retry_loop() -> None:
    requested = {"max_regeneration_rounds": 6, "max_task_retries": 3, "max_repair_rounds": 2}
    capped = GuardrailConfig.from_requirements(requested, max_retries=1)
    assert (capped.max_regeneration_rounds, capped.max_task_retries, capped.max_repair_rounds) == (1, 1, 1)
    uncapped = GuardrailConfig.from_requirements(requested)
    assert uncapped.max_regeneration_rounds == 6
    assert GuardrailConfig.from_requirements({"max_task_retries": 0}, max_retries=5).max_task_retries == 0


def test_cap_bounds_llm_call_attempts() -> None:
    model = ModelConfig(provider="mock", model_name="m", model_id="m", retry={"max_attempts": 5})
    assert model.with_retry_cap(0).retry.max_attempts == 1
    assert model.with_retry_cap(2).retry.max_attempts == 3
    assert model.with_retry_cap(10).retry.max_attempts == 5
    assert model.with_retry_cap(None) is model


def test_job_stops_at_the_retry_limit(tmp_path: Path, registry) -> None:
    from app.engine import SyntheticDataEngine  # pylint: disable=import-outside-toplevel

    MockLLMAdapter.behaviors["mock-flaky"] = MockBehavior(fail_until_call=1)
    request = {"schema": {"id": "unique integer", "bio": "short bio"}, "total_records": 20, "batch_size": 20,
               "model_pool": ["mock/mock-flaky"], "execution_mode": "thread"}
    capped = SyntheticDataEngine(settings=Settings(output_dir=tmp_path / "a", log_dir=tmp_path / "logs",
                                                   max_retries=0), registry=registry)
    result = capped.run(request)
    assert result.status == "failed" and "0 retry round" in (result.error or "")

    MockLLMAdapter.calls.clear()
    default = SyntheticDataEngine(settings=Settings(output_dir=tmp_path / "b", log_dir=tmp_path / "logs"),
                                  registry=registry)
    assert default.run(request).ok  # the second attempt of the same call succeeds without the cap
