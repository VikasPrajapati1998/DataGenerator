"""Tunable thresholds for node-level and global guardrails.

Users override any value through the request's ``quality_requirements``.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class GuardrailConfig(BaseModel):
    """Thresholds and limits applied by guardrails and validators.

    Attributes:
        max_regeneration_rounds: Times a task may regenerate rejected records.
        max_task_retries: Graph-level retries of a failed task.
        max_repair_rounds: Global repair rounds after cross-node validation.
        semantic_dedup: Enable embedding-based near-duplicate detection.
        semantic_threshold: Cosine similarity at or above which records are near duplicates.
        embedder: ``hashing`` (local, fast) or ``ollama:<model>``.
        min_distinct_ratio: Minimum distinct/total ratio for free-text fields per batch.
        max_value_repeat_ratio: Max share of a batch a single free-text value may take.
        max_text_length: Upper bound for any LLM-generated text value.
        distribution_sigma: Allowed deviation, in standard errors, for distribution checks.
        distribution_abs_tolerance: Minimum absolute tolerance for proportion checks.
        min_quality_score: Global quality score (0-100) required to deliver the dataset.
        global_schema_revalidation: Re-validate every record of every batch globally.
        llm_chunk_size: Records enriched per LLM request.
        enforce_distribution: Fail the job when distribution checks fail.
        split_failed_tasks: On retry, split a failed task into sub-batches that
            several core queues generate in parallel.
        min_split_records: Smallest sub-batch a split may create.
        max_split_parts: Upper bound on sub-batches per failed task.
        max_invalid_fraction: Leftover invalid records (share of a task, at least
            ``min_rescue_records``) that get ``rescue_rounds`` extra rounds instead of
            failing the whole task.
        min_rescue_records: Leftover records that are always rescued, whatever the task size.
        rescue_rounds: Extra regeneration rounds for those few leftover records.
    """

    model_config = ConfigDict(extra="forbid")

    max_regeneration_rounds: int = Field(default=4, ge=0, le=20)
    max_task_retries: int = Field(default=2, ge=0, le=10)
    max_repair_rounds: int = Field(default=2, ge=0, le=10)
    semantic_dedup: bool = True
    semantic_threshold: float = Field(default=0.97, gt=0.5, le=1.0)
    embedder: str = "hashing"
    min_distinct_ratio: float = Field(default=0.6, ge=0.0, le=1.0)
    max_value_repeat_ratio: float = Field(default=0.05, gt=0.0, le=1.0)
    max_text_length: int = Field(default=600, ge=10)
    distribution_sigma: float = Field(default=4.5, gt=0)
    distribution_abs_tolerance: float = Field(default=0.02, ge=0)
    min_quality_score: float = Field(default=85.0, ge=0, le=100)
    global_schema_revalidation: bool = True
    llm_chunk_size: int = Field(default=8, ge=1, le=50)
    enforce_distribution: bool = True
    split_failed_tasks: bool = True
    min_split_records: int = Field(default=50, ge=1)
    max_split_parts: int = Field(default=8, ge=2, le=64)
    max_invalid_fraction: float = Field(default=0.001, ge=0.0, le=0.5)
    min_rescue_records: int = Field(default=10, ge=0)
    rescue_rounds: int = Field(default=10, ge=0, le=50)

    @classmethod
    def from_requirements(cls, requirements: dict[str, Any] | None, max_retries: int | None = None,
                          max_invalid_fraction: float | None = None) -> GuardrailConfig:
        """Build a config from the request's ``quality_requirements``.

        Args:
            requirements: Partial overrides; unknown keys are rejected.
            max_retries: Global retry cap (``SDE_MAX_RETRIES``). When set, it is an
                upper bound for regeneration rounds, task retries and repair rounds,
                whatever the request asks for.
            max_invalid_fraction: ``SDE_MAX_INVALID_FRACTION``, used unless the request sets it.

        Returns:
            The merged configuration.
        """
        requirements = dict(requirements or {})
        if max_invalid_fraction is not None:
            requirements.setdefault("max_invalid_fraction", max_invalid_fraction)
        config = cls(**requirements)
        if max_retries is None:
            return config
        return config.model_copy(update={
            "max_regeneration_rounds": min(config.max_regeneration_rounds, max_retries),
            "max_task_retries": min(config.max_task_retries, max_retries),
            "max_repair_rounds": min(config.max_repair_rounds, max_retries),
        })
