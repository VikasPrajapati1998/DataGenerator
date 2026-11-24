"""Shared LangGraph state and the runtime context.

The state holds only metadata - task specs, allocations, file paths,
statuses and reports. Generated records live in batch files on disk and are
never placed in graph state.

Non-serialisable runtime objects (scheduler, executor, tracker, services,
cancellation token) are passed through LangGraph's runtime context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict

from app.config.model_registry import ModelRegistry
from app.config.settings import Settings
from app.core.cancellation import CancellationToken
from app.core.exceptions import EngineError
from app.models.ollama_local import unload_local_models
from app.monitoring.logger import get_logger
from app.graph.reducers import concat
from app.monitoring.execution_tracker import ExecutionTracker
from app.scheduler.work_stealing import CoreQueueScheduler
from app.storage.file_manager import JobPaths
from app.workers.services import WorkerServices
from app.workers.worker_manager import TaskExecutor


logger = get_logger(__name__)


class GraphState(TypedDict, total=False):
    """Structured shared state of one generation job."""

    # identity and input
    job_id: str
    job_dir: str
    resume: bool
    request: dict[str, Any]
    schema: dict[str, Any]
    output_format: str
    total_records: int
    batch_size: int
    seed: int
    # models
    model_chain: list[dict[str, Any]]
    model_pool: list[str]
    ollama_loaded: list[str]
    model_status: dict[str, Any]
    profiles: dict[str, Any]
    llm_required: bool
    # planning and scheduling
    total_tasks: int
    tasks: list[dict[str, Any]]
    resources: dict[str, Any]
    concurrency: dict[str, Any]
    execution_mode: str
    core_allocation: dict[str, list[int | str]]
    pending_work: list[dict[str, Any]]
    dispatch_round: int
    phase: str
    retry_round: int
    repair_round: int
    already_completed: list[int]
    # results (written by parallel branches -> reducers)
    task_results: Annotated[list[dict[str, Any]], concat]
    queue_reports: Annotated[list[dict[str, Any]], concat]
    errors: Annotated[list[str], concat]
    # aggregation and outputs
    completed_tasks: list[int]
    failed_tasks: list[int]
    batch_files: list[str]
    global_report: dict[str, Any]
    quality: dict[str, Any]
    merge_result: dict[str, Any]
    final_validation: dict[str, Any]
    final_output: str
    status: str
    summary: dict[str, Any]


class CoreWorkerInput(TypedDict):
    """Payload of each ``Send`` to the reusable core-queue worker node."""

    core_id: str
    dispatch_round: int


@dataclass
class EngineContext:
    """Mutable holder for runtime objects shared by graph nodes.

    Attributes:
        settings: Application settings.
        registry: Model registry.
        cancel: Cancellation token (replaced once the job directory is known).
        paths: Job directory layout.
        tracker: Execution tracker.
        services: Parent-process worker services.
        executor: Thread/process task executor.
        scheduler: Scheduler of the current dispatch round.
        options: Request options needed at runtime (work stealing, pinning...).
    """

    settings: Settings
    registry: ModelRegistry
    cancel: CancellationToken = field(default_factory=CancellationToken)
    paths: JobPaths | None = None
    tracker: ExecutionTracker | None = None
    services: WorkerServices | None = None
    executor: TaskExecutor | None = None
    scheduler: CoreQueueScheduler | None = None
    options: dict[str, Any] = field(default_factory=dict)

    def close(self, force: bool = False) -> None:
        """Release every resource of the job; safe to call more than once and from signal handlers.

        Order: stop worker processes, stop the monitor, close HTTP clients and the
        registry, then ask Ollama to unload the local models the job used.
        Each step is isolated, so one failing step never skips the others.

        Args:
            force: Terminate worker processes without a grace period.
        """
        executor, self.executor = self.executor, None
        services, self.services = self.services, None
        tracker = self.tracker
        steps: list[tuple[str, Any]] = []
        if executor is not None:
            steps.append(("worker processes", lambda: executor.shutdown(force=force)))
        if tracker is not None:
            steps.append(("progress monitor", tracker.stop_monitoring))
        if services is not None:
            steps.append(("services", services.close))
            used = set(tracker.llm.by_model) if tracker is not None else set()
            if self.settings.unload_models and used:
                steps.append(("Ollama models", lambda: unload_local_models(services.chain, self.settings, used)))
        for label, step in steps:
            try:
                step()
            except (EngineError, OSError, RuntimeError, ValueError) as exc:
                logger.warning("Releasing %s failed: %s", label, exc)
