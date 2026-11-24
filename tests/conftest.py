"""Shared pytest fixtures: isolated settings, mock models and sample requests."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from app.config.model_registry import ModelConfig, ModelRegistry
from app.config.settings import Settings
from app.models.mock_adapter import MockLLMAdapter

EMPLOYEE_SCHEMA: dict[str, Any] = {
    "employee_id": "unique integer",
    "name": "realistic full name",
    "email": "valid email",
    "age": "integer between 22 and 60",
    "city": "Indian city",
    "department": ["IT", "HR", "Finance", "Sales"],
    "salary": "realistic salary based on department",
}


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings that write jobs and logs into a temporary directory."""
    return Settings(output_dir=tmp_path / "output", log_dir=tmp_path / "logs", default_model="mock/mock-llm")


@pytest.fixture
def registry() -> ModelRegistry:
    """Registry containing only offline mock models."""
    entries = [
        ModelConfig(provider="mock", model_name="mock-llm", model_id="mock-llm", max_concurrency=16,
                    retry={"max_attempts": 3, "base_delay": 0.0, "max_delay": 0.0}),
        ModelConfig(provider="mock", model_name="mock-backup", model_id="mock-backup", max_concurrency=16,
                    retry={"max_attempts": 2, "base_delay": 0.0, "max_delay": 0.0}),
        ModelConfig(provider="mock", model_name="mock-flaky", model_id="mock-flaky", max_concurrency=16,
                    retry={"max_attempts": 2, "base_delay": 0.0, "max_delay": 0.0}),
    ]
    return ModelRegistry(entries, "mock/mock-llm")


@pytest.fixture(autouse=True)
def reset_mock_behaviors():
    """Ensure failure injection never leaks between tests."""
    MockLLMAdapter.behaviors.clear()
    MockLLMAdapter.calls.clear()
    yield
    MockLLMAdapter.behaviors.clear()
    MockLLMAdapter.calls.clear()


@pytest.fixture
def employee_request() -> dict[str, Any]:
    """The employee example from the specification, sized for tests."""
    return {
        "schema": copy.deepcopy(EMPLOYEE_SCHEMA),
        "total_records": 1000,
        "output_format": "csv",
        "generation_instructions": "Generate realistic Indian employee records.",
        "model_provider": "mock",
        "model_name": "mock-llm",
        "batch_size": 250,
        "execution_mode": "thread",
        "seed": 7,
    }


@pytest.fixture
def engine(settings: Settings, registry: ModelRegistry):
    """Engine wired to temporary settings and mock models."""
    from app.engine import SyntheticDataEngine  # pylint: disable=import-outside-toplevel

    return SyntheticDataEngine(settings=settings, registry=registry)
