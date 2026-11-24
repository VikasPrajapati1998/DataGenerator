"""Real-provider integration tests.

* Ollama Local tests run only when a local server is reachable and the model
  is installed (``SDE_TEST_OLLAMA_MODEL``, default ``qwen3:0.6b``).
* Ollama Cloud tests run only when ``OLLAMA_API_KEY`` is set *and*
  ``SDE_RUN_CLOUD_TESTS=1`` (they consume the account's cloud quota).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from app.config.model_registry import ModelConfig, ModelRegistry
from app.config.settings import Settings, get_settings
from app.generation.profile_calibrator import calibrate_profiles
from app.guardrails.semantic_deduplication import OllamaEmbedder
from app.models.model_router import LLMUsage, ModelRouter
from app.models.ollama_cloud import OllamaCloudAdapter
from app.models.ollama_local import OllamaLocalAdapter
from app.schemas.schema_parser import parse_schema
from tests.conftest import EMPLOYEE_SCHEMA

MODEL = os.getenv("SDE_TEST_OLLAMA_MODEL", "qwen3:0.6b")
EMBED_MODEL = os.getenv("SDE_TEST_OLLAMA_EMBED_MODEL", "nomic-embed-text:v1.5")


def _local(model: str, **kwargs) -> ModelConfig:
    return ModelConfig(provider="ollama_local", model_name=model, model_id=model, thinking=model.startswith("qwen3"),
                       temperature=0.7, timeout=300, retry={"max_attempts": 3, "base_delay": 1.0}, **kwargs)


def _installed(model: str) -> bool:
    adapter = OllamaLocalAdapter(_local(model), get_settings())
    try:
        return adapter.check_availability().available
    finally:
        adapter.close()


ollama = pytest.mark.skipif(not _installed(MODEL), reason=f"local Ollama with {MODEL} not available")


@pytest.mark.ollama
@ollama
def test_ollama_local_detects_installed_models() -> None:
    adapter = OllamaLocalAdapter(_local(MODEL), get_settings())
    names = {m["name"] for m in adapter.list_models()}
    assert MODEL in names
    assert adapter.check_availability().metadata.get("size_bytes")
    missing = OllamaLocalAdapter(_local("definitely-not-installed:1b"), get_settings())
    assert not missing.check_availability().available


@pytest.mark.ollama
@ollama
def test_ollama_local_structured_output() -> None:
    router = ModelRouter([_local(MODEL)], get_settings())
    schema = {"type": "object", "properties": {"city": {"type": "string"}, "population_millions": {"type": "number"}},
              "required": ["city", "population_millions"]}
    usage = LLMUsage()
    data = router.generate_json("Answer with JSON only.", "Name one large Indian city and its population.",
                                schema, usage)
    assert isinstance(data["city"], str) and float(data["population_millions"]) > 0
    assert usage.calls == 1 and usage.completion_tokens > 0


@pytest.mark.ollama
@ollama
def test_ollama_local_profile_calibration() -> None:
    schema = parse_schema(EMPLOYEE_SCHEMA, "Generate realistic Indian employee records.")
    router = ModelRouter([_local(MODEL)], get_settings())
    profiles = calibrate_profiles(schema, router, "Generate realistic Indian employee records.", LLMUsage())
    ranges = profiles["salary"]
    assert set(ranges) == {"IT", "HR", "Finance", "Sales"}
    assert all(r.min < r.max for r in ranges.values())


@pytest.mark.ollama
@pytest.mark.skipif(not _installed(EMBED_MODEL), reason=f"{EMBED_MODEL} not installed")
def test_ollama_embeddings_for_semantic_dedup() -> None:
    adapter = OllamaLocalAdapter(_local(EMBED_MODEL, structured_output="prompt_only"), get_settings())
    vectors = OllamaEmbedder(adapter).embed(["Loves hiking in the hills.", "Enjoys hill hiking.",
                                             "Quarterly revenue report for finance."])
    sims = vectors @ vectors.T
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-3)
    assert sims[0, 1] > sims[0, 2]


@pytest.mark.ollama
@ollama
def test_ollama_local_end_to_end_small_job(tmp_path: Path) -> None:
    from app.engine import SyntheticDataEngine  # pylint: disable=import-outside-toplevel

    registry = ModelRegistry([_local(MODEL)], f"ollama_local/{MODEL}")
    engine = SyntheticDataEngine(settings=Settings(output_dir=tmp_path / "out", log_dir=tmp_path / "logs"),
                                 registry=registry)
    request = {
        "schema": {"customer_id": "unique integer", "name": "realistic full name",
                   "segment": ["retail", "business"], "bio": "one-sentence description of the customer's interests"},
        "total_records": 16, "batch_size": 8, "model_provider": "ollama", "model_name": MODEL,
        "execution_mode": "thread", "quality_requirements": {"max_regeneration_rounds": 6, "min_quality_score": 70},
    }
    result = engine.run(request)
    assert result.ok, result.error


cloud = pytest.mark.skipif(not (os.getenv("OLLAMA_API_KEY") and os.getenv("SDE_RUN_CLOUD_TESTS") == "1"),
                           reason="set OLLAMA_API_KEY and SDE_RUN_CLOUD_TESTS=1 to run Ollama Cloud tests")


@pytest.mark.cloud
@cloud
def test_ollama_cloud_json_mode() -> None:
    config = ModelConfig(provider="ollama_cloud", model_name="gpt-oss:120b", model_id="gpt-oss:120b",
                         execution="cloud", auth_env="OLLAMA_API_KEY", structured_output="json_mode", thinking=True)
    adapter = OllamaCloudAdapter(config, get_settings())
    assert adapter.check_availability().available
    router = ModelRouter([config], get_settings())
    data = router.generate_json("JSON only.", "Give one Indian city.", {"type": "object", "properties": {
        "city": {"type": "string"}}, "required": ["city"]}, LLMUsage())
    assert data["city"]
