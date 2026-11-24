"""Project-wide exception hierarchy.

Every error raised intentionally by the engine derives from :class:`EngineError`
so callers can catch engine failures without catching unrelated bugs.
"""

from __future__ import annotations


class EngineError(Exception):
    """Base class for all synthetic-data-engine errors."""


class ConfigurationError(EngineError):
    """Invalid settings, registry entries, or request options."""


class SchemaError(EngineError):
    """The user-supplied data schema is invalid or unsupported."""


class ValidationError(EngineError):
    """Generated data failed validation in a way that cannot be repaired."""


class ResourceError(EngineError):
    """Insufficient RAM, disk space, or other host resources."""


class FeasibilityError(ResourceError):
    """The job would take far longer than allowed (``SDE_MAX_ETA_HOURS``) with the available models."""


class SchedulingError(EngineError):
    """The task planner or core-queue scheduler reached an invalid state."""


class TaskExecutionError(EngineError):
    """A generation task failed after exhausting its retries."""


class CancelledError(EngineError):
    """The job was cancelled by the user or a shutdown signal."""


class StorageError(EngineError):
    """Reading, writing, or verifying a batch/output file failed."""


class MergeError(StorageError):
    """The deterministic merger refused to produce the final dataset."""


class RecoveryError(StorageError):
    """Recovery state (split manifest, part batches) is inconsistent or unreadable."""


class WorkerCrashError(TaskExecutionError):
    """A worker process died or could not run a task (e.g. broken process pool)."""


class LLMError(EngineError):
    """Base class for language-model provider failures."""


class ModelUnavailableError(LLMError):
    """The requested model is not installed, reachable, or authorised."""


class QuotaExceededError(ModelUnavailableError):
    """The provider refuses service because credits or quota are exhausted (e.g. HTTP 402).

    Permanent for the duration of a job: the model is disabled, never retried.
    """


class ContextLengthError(LLMError):
    """The request does not fit this model's context window (another model may take it)."""


class ProviderTimeoutError(LLMError):
    """The provider did not answer within the configured timeout."""


class RateLimitError(LLMError):
    """The provider rejected the request because of rate limiting.

    Attributes:
        retry_after: Seconds the provider asked us to wait, when known.
    """

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class InvalidModelOutputError(LLMError):
    """The model answered, but the output was not valid structured data."""


class ProviderError(LLMError):
    """Any other provider-side failure (HTTP 5xx, connection errors...)."""


RETRYABLE_LLM_ERRORS: tuple[type[Exception], ...] = (
    ProviderTimeoutError,
    RateLimitError,
    InvalidModelOutputError,
    ProviderError,
)
"""LLM errors that are worth retrying with backoff."""
