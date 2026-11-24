"""Host resource detection: CPU cores, RAM, GPU/VRAM and affinity support."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from typing import Any

import psutil

from app.monitoring.logger import get_logger

logger = get_logger(__name__)


@dataclass
class GPUInfo:
    """One NVIDIA GPU as reported by ``nvidia-smi``."""

    name: str
    memory_total_mb: int
    memory_free_mb: int
    utilization_pct: float


@dataclass
class ResourceInfo:
    """Snapshot of host resources relevant to scheduling.

    Attributes:
        logical_cores: ``os.cpu_count()`` (hardware threads).
        physical_cores: Physical cores when detectable.
        usable_cores: Cores this process may run on (affinity mask), if known.
        total_ram_mb: Installed RAM.
        available_ram_mb: RAM currently available.
        gpus: Detected NVIDIA GPUs (empty when none or no driver).
        affinity_supported: Whether per-thread/process CPU pinning is possible.
        platform: ``sys.platform``.
    """

    logical_cores: int
    physical_cores: int | None
    usable_cores: int | None
    total_ram_mb: int
    available_ram_mb: int
    gpus: list[GPUInfo] = field(default_factory=list)
    affinity_supported: bool = False
    platform: str = sys.platform

    @property
    def cpu_cores(self) -> int:
        """Cores used as the default concurrency limit."""
        return self.usable_cores or self.logical_cores

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable copy."""
        data = asdict(self)
        data["cpu_cores"] = self.cpu_cores
        return data


def query_gpus() -> list[GPUInfo]:
    """Return NVIDIA GPUs via ``nvidia-smi`` (empty list when unavailable)."""
    if shutil.which("nvidia-smi") is None:
        return []
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout
    except (subprocess.SubprocessError, OSError) as exc:
        logger.debug("nvidia-smi failed: %s", exc)
        return []
    gpus = []
    for line in output.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 4:
            try:
                gpus.append(GPUInfo(parts[0], int(float(parts[1])), int(float(parts[2])), float(parts[3])))
            except ValueError:
                continue
    return gpus


def detect_resources() -> ResourceInfo:
    """Detect the resources available to this process.

    Returns:
        The resource snapshot.
    """
    logical = os.cpu_count() or 1
    usable: int | None = None
    affinity_supported = False
    try:
        usable = len(psutil.Process().cpu_affinity())  # type: ignore[attr-defined]
        affinity_supported = True
    except (AttributeError, psutil.Error, OSError):
        if hasattr(os, "sched_getaffinity"):
            usable = len(os.sched_getaffinity(0))
            affinity_supported = True
    memory = psutil.virtual_memory()
    info = ResourceInfo(
        logical_cores=logical,
        physical_cores=psutil.cpu_count(logical=False),
        usable_cores=usable,
        total_ram_mb=memory.total // (1024 * 1024),
        available_ram_mb=memory.available // (1024 * 1024),
        gpus=query_gpus(),
        affinity_supported=affinity_supported,
    )
    logger.info("Detected %d logical cores (%s physical, %s usable), %d MB RAM available, %d GPU(s)",
                info.logical_cores, info.physical_cores, info.usable_cores, info.available_ram_mb, len(info.gpus))
    return info
