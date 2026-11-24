"""Client-side rate limiting and concurrency limiting for providers."""

from __future__ import annotations

import threading
import time
from typing import Protocol


class Limiter(Protocol):
    """Anything with ``acquire``/``release`` (threading or multiprocessing semaphores)."""

    def acquire(self, blocking: bool = True, timeout: float | None = None) -> bool:
        """Acquire one slot."""

    def release(self) -> None:
        """Release one slot."""


class TokenBucket:
    """Thread-safe token bucket enforcing a requests-per-minute budget.

    Args:
        requests_per_minute: Sustained request rate.
        burst: Maximum burst size; defaults to 1/6 of the per-minute rate.
    """

    def __init__(self, requests_per_minute: float, burst: int | None = None) -> None:
        self.rate = requests_per_minute / 60.0
        self.capacity = float(burst or max(1, int(requests_per_minute // 6)))
        self._tokens = self.capacity
        self._updated = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> float:
        """Block until a token is available.

        Returns:
            Seconds spent waiting.
        """
        waited = 0.0
        while True:
            with self._lock:
                now = time.monotonic()
                self._tokens = min(self.capacity, self._tokens + (now - self._updated) * self.rate)
                self._updated = now
                if self._tokens >= 1:
                    self._tokens -= 1
                    return waited
                needed = (1 - self._tokens) / self.rate
            time.sleep(needed)
            waited += needed
