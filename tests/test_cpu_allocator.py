"""CPU-core-based task allocation and resource-aware concurrency."""

from __future__ import annotations

import pytest

from app.config.model_registry import ModelConfig
from app.config.settings import Settings
from app.core.exceptions import ResourceError
from app.scheduler.concurrency_manager import plan_concurrency
from app.scheduler.cpu_detector import ResourceInfo, detect_resources
from app.scheduler.task_allocator import allocate_to_cores


def ids(n: int) -> list[int]:
    return list(range(1, n + 1))


def test_scenario_a_four_cores_twenty_tasks() -> None:
    allocation = allocate_to_cores(ids(20), 4)
    assert allocation == {"core_1": [1, 2, 3, 4, 5], "core_2": [6, 7, 8, 9, 10],
                          "core_3": [11, 12, 13, 14, 15], "core_4": [16, 17, 18, 19, 20]}


def test_scenario_b_four_cores_nineteen_tasks() -> None:
    allocation = allocate_to_cores(ids(19), 4)
    assert [len(v) for v in allocation.values()] == [5, 5, 5, 4]
    assert allocation["core_4"] == [16, 17, 18, 19]


def test_scenario_c_eight_cores_sixty_four_tasks() -> None:
    allocation = allocate_to_cores(ids(64), 8)
    assert all(len(v) == 8 for v in allocation.values())


def test_two_cores_twenty_tasks() -> None:
    assert [len(v) for v in allocate_to_cores(ids(20), 2).values()] == [10, 10]


@pytest.mark.parametrize(("tasks", "cores"), [(7, 3), (1024, 6), (100, 7), (3, 3), (5, 8)])
def test_balanced_and_complete(tasks: int, cores: int) -> None:
    allocation = allocate_to_cores(ids(tasks), cores)
    sizes = [len(v) for v in allocation.values()]
    assert max(sizes) - min(sizes) <= 1
    flattened = [t for queue in allocation.values() for t in queue]
    assert flattened == ids(tasks)  # every task exactly once, contiguous blocks


def test_one_task_per_core() -> None:
    assert allocate_to_cores(ids(4), 4) == {f"core_{i}": [i] for i in range(1, 5)}


def test_fewer_tasks_than_cores_creates_no_empty_queues() -> None:
    allocation = allocate_to_cores(ids(2), 8)
    assert list(allocation) == ["core_1", "core_2"]


def _resources(cores: int = 8, ram: int = 16000) -> ResourceInfo:
    return ResourceInfo(logical_cores=cores, physical_cores=cores // 2, usable_cores=cores, total_ram_mb=ram,
                        available_ram_mb=ram)


def _model(**kwargs) -> ModelConfig:
    return ModelConfig(provider="ollama_local", model_name="qwen3:0.6b", model_id="qwen3:0.6b",
                       approx_memory_mb=1000, **kwargs)


def test_detect_resources_reports_real_host() -> None:
    info = detect_resources()
    assert info.logical_cores >= 1 and info.cpu_cores >= 1 and info.total_ram_mb > 0


def test_concurrency_defaults_to_core_count() -> None:
    plan = plan_concurrency(_resources(8), _model(), Settings(), total_tasks=64)
    assert plan.queues == 8
    assert plan.llm_concurrency == 1  # OLLAMA_NUM_PARALLEL default


def test_concurrency_override_and_task_limit() -> None:
    assert plan_concurrency(_resources(8), _model(), Settings(), 64, max_workers=3).queues == 3
    assert plan_concurrency(_resources(8), _model(), Settings(), 2).queues == 2


def test_ollama_parallelism_from_settings() -> None:
    plan = plan_concurrency(_resources(8), _model(), Settings(ollama_num_parallel=4), 64)
    assert plan.llm_concurrency == 4


def test_ram_constrained_execution() -> None:
    plan = plan_concurrency(_resources(16, ram=1700), _model(), Settings(memory_per_worker_mb=150), 64)
    assert plan.queues == 2  # (1700 - 1000 - 256) // 150
    tight = plan_concurrency(_resources(8, ram=1200), _model(), Settings(), 64)  # default SDE_RAM_CHECK=warn
    assert tight.queues == 1 and any(r.startswith("Low RAM") for r in tight.reasons)
    with pytest.raises(ResourceError):
        plan_concurrency(_resources(8, ram=1200), _model(), Settings(ram_check="strict"), 64)
    assert plan_concurrency(_resources(8, ram=1200), _model(), Settings(ram_check="off"), 64).queues == 8


def test_ram_budget_counts_only_local_models_still_to_load() -> None:
    pool = [_model(),
            ModelConfig(provider="ollama_local", model_name="llama3.2:1b", model_id="llama3.2:1b",
                        approx_memory_mb=1700),
            ModelConfig(provider="ollama_local", model_name="gpt-oss:120b-cloud", model_id="gpt-oss:120b-cloud",
                        execution="cloud", approx_memory_mb=0)]
    # Largest not-yet-loaded local model (1700 MB) is reserved once; the cloud model needs nothing.
    assert plan_concurrency(_resources(8, ram=2400), pool[0], Settings(), 64, pool=pool).queues == 2
    # With llama already loaded in Ollama, only qwen (1000 MB) still has to fit.
    loaded = plan_concurrency(_resources(8, ram=2400), pool[0], Settings(), 64, pool=pool,
                              loaded_models={"llama3.2:1b"})
    assert loaded.queues == 7  # (2400 - 1000 - 256) // 150
    cloud_only = plan_concurrency(_resources(8, ram=600), pool[2], Settings(), 64, pool=[pool[2]])
    assert cloud_only.queues == 2  # (600 - 0 - 256) // 150


def test_cloud_model_limited_by_rate() -> None:
    model = ModelConfig(provider="ollama_cloud", model_name="gpt-oss:120b", model_id="gpt-oss:120b",
                        execution="cloud", requests_per_minute=30)
    plan = plan_concurrency(_resources(8), model, Settings(), 64)
    assert plan.llm_concurrency == 3 and plan.queues == 8
