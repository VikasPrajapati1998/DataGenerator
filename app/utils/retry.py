"""Exponential backoff with jitter for retryable operations."""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from app.core.cancellation import CancellationToken
from app.core.exceptions import RateLimitError
from app.monitoring.logger import get_logger

T = TypeVar("T")
logger = get_logger(__name__)


@dataclass(frozen=True)
class RetryPolicy:
    """How often and how patiently an operation is retried.

    Attributes:
        max_attempts: Total attempts including the first one.
        base_delay: Delay before the first retry, in seconds.
        max_delay: Upper bound for a single delay, in seconds.
        multiplier: Growth factor between consecutive delays.
        jitter: Fraction of random jitter added to each delay (0..1).
    """

    max_attempts: int = 3
    base_delay: float = 0.5
    max_delay: float = 30.0
    multiplier: float = 2.0
    jitter: float = 0.25

    def delay_for(self, attempt: int) -> float:
        """Return the delay to wait after failed attempt number ``attempt`` (1-based).

        Args:
            attempt: The attempt that just failed.

        Returns:
            Seconds to sleep before the next attempt.
        """
        raw = min(self.max_delay, self.base_delay * (self.multiplier ** (attempt - 1)))
        return raw + raw * self.jitter * random.random()


def retry_call(
    func: Callable[[], T],
    policy: RetryPolicy,
    retry_on: tuple[type[Exception], ...],
    description: str = "operation",
    cancel: CancellationToken | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call ``func`` and retry it with exponential backoff on selected errors.

    Rate-limit errors that carry ``retry_after`` wait at least that long.

    Args:
        func: Zero-argument callable to execute.
        policy: Retry policy to apply.
        retry_on: Exception types that trigger a retry.
        description: Human-readable name used in log messages.
        cancel: Optional token; cancellation stops retrying immediately.
        sleep: Sleep function (injectable for tests).

    Returns:
        Whatever ``func`` returns on its first successful attempt.

    Raises:
        Exception: The last error once all attempts are exhausted, or any
            error not listed in ``retry_on``.
    """
    attempt = 1
    while True:
        if cancel is not None:
            cancel.raise_if_cancelled()
        try:
            return func()
        except retry_on as exc:
            if attempt >= policy.max_attempts:
                logger.warning("%s: retry limit reached, failed after %d attempt(s): %s", description, attempt, exc)
                raise
            delay = policy.delay_for(attempt)
            if isinstance(exc, RateLimitError) and exc.retry_after:
                delay = max(delay, exc.retry_after)
            logger.info(
                "%s attempt %d/%d failed (%s: %s); retrying in %.2fs",
                description, attempt, policy.max_attempts, type(exc).__name__, exc, delay,
            )
            sleep(delay)
            attempt += 1
