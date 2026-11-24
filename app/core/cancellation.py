"""Cooperative cancellation shared by threads and worker processes.

A job can be cancelled in-process (Ctrl+C sets the event) or from another
terminal (``python -m app.main cancel <job_id>`` creates a ``CANCEL`` marker
file in the job directory). Worker processes only see the marker file, which
makes the mechanism work identically on Windows (spawn) and Linux (fork).
"""

from __future__ import annotations

import threading
from pathlib import Path

from app.core.exceptions import CancelledError

CANCEL_MARKER = "CANCEL"


class CancellationToken:
    """Thread-safe and process-visible cancellation flag.

    Args:
        job_dir: Job directory whose ``CANCEL`` marker file is also honoured.
    """

    def __init__(self, job_dir: Path | None = None) -> None:
        self._event = threading.Event()
        self._marker = job_dir / CANCEL_MARKER if job_dir else None

    @property
    def marker_path(self) -> Path | None:
        """Path of the cross-process cancel marker file, if any."""
        return self._marker

    def cancel(self) -> None:
        """Request cancellation for this process and every worker process."""
        self._event.set()
        if self._marker is not None:
            self._marker.parent.mkdir(parents=True, exist_ok=True)
            self._marker.touch(exist_ok=True)

    def is_cancelled(self) -> bool:
        """Return ``True`` when cancellation was requested anywhere."""
        if self._event.is_set():
            return True
        if self._marker is not None and self._marker.exists():
            self._event.set()
            return True
        return False

    def raise_if_cancelled(self) -> None:
        """Raise :class:`CancelledError` when cancellation was requested.

        Raises:
            CancelledError: If the job has been cancelled.
        """
        if self.is_cancelled():
            raise CancelledError("Job cancellation requested")

    def clear(self) -> None:
        """Reset the token and remove the marker (used when resuming a job)."""
        self._event.clear()
        if self._marker is not None and self._marker.exists():
            self._marker.unlink()
