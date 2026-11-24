"""Optional physical CPU pinning for queue threads or worker processes.

Logical queues are *not* bound to physical cores by default: the OS
scheduler is usually better at placing threads, and LLM inference runs in the
Ollama server process anyway. Pinning is offered for CPU-bound workloads on
dedicated machines (``cpu_affinity: true``) and only where the OS supports it:

* Linux: ``os.sched_setaffinity(0, ...)`` applies to the calling thread.
* Windows: ``SetThreadAffinityMask`` on the current thread via ``ctypes``.
* macOS: not supported (the kernel offers no affinity API) - logged and skipped.
"""

from __future__ import annotations

import os
import sys

from app.monitoring.logger import get_logger

logger = get_logger(__name__)


def pin_current_thread(cpu_index: int) -> bool:
    """Pin the calling thread to logical CPU ``cpu_index`` when supported.

    Args:
        cpu_index: Zero-based logical CPU number.

    Returns:
        ``True`` if pinning succeeded.
    """
    cpu_count = os.cpu_count() or 1
    cpu = cpu_index % cpu_count
    try:
        if sys.platform.startswith("linux"):
            os.sched_setaffinity(0, {cpu})  # type: ignore[attr-defined]
            return True
        if sys.platform == "win32":
            import ctypes  # pylint: disable=import-outside-toplevel

            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            kernel32.GetCurrentThread.restype = ctypes.c_void_p
            kernel32.SetThreadAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            kernel32.SetThreadAffinityMask.restype = ctypes.c_size_t
            return bool(kernel32.SetThreadAffinityMask(kernel32.GetCurrentThread(), 1 << cpu))
    except (OSError, AttributeError, ValueError) as exc:
        logger.warning("CPU pinning to %d failed: %s", cpu, exc)
        return False
    logger.info("CPU affinity is not supported on %s; continuing unpinned", sys.platform)
    return False


def unpin_current_thread() -> None:
    """Allow the calling thread to run on every logical CPU again (pool threads are reused)."""
    cpu_count = os.cpu_count() or 1
    try:
        if sys.platform.startswith("linux"):
            os.sched_setaffinity(0, set(range(cpu_count)))  # type: ignore[attr-defined]
        elif sys.platform == "win32":
            import ctypes  # pylint: disable=import-outside-toplevel

            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            kernel32.GetCurrentThread.restype = ctypes.c_void_p
            kernel32.SetThreadAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            kernel32.SetThreadAffinityMask(kernel32.GetCurrentThread(), (1 << cpu_count) - 1)
    except (OSError, AttributeError, ValueError) as exc:
        logger.warning("Restoring CPU affinity failed: %s", exc)


def core_index(core_id: str) -> int:
    """Return the zero-based index of a queue id like ``core_3``."""
    return int(core_id.rsplit("_", 1)[-1]) - 1
