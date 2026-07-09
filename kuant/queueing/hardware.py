"""Hardware detection — foundation for the throttle layer.

Design principle: this module knows what device we have, exposes it as an
immutable snapshot, and NEVER holds references to user arrays. Only
metadata (bytes, counts, capability) lives here.

Detection runs ONCE at import via detect_hw(); users read the DEVICE
singleton after that.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeviceProfile:
    """Immutable snapshot of the compute device.

    Frozen so users cannot mutate accidentally. All fields are primitives
    (int/float/str/tuple) — no array references possible.
    """

    tier: str
    """One of: cpu, consumer_low, consumer_mid, consumer_high, workstation,
    datacenter. Coarse category driving default batch sizes."""

    name: str
    """Human-readable device name (e.g. "NVIDIA RTX 4090" or "CPU only")."""

    vram_total_gb: float
    """Total VRAM in gigabytes. 0.0 for CPU."""

    vram_free_gb: float
    """Free VRAM at detection time."""

    sm_count: int
    """Streaming multiprocessor count. Rough proxy for parallelism budget."""

    compute_capability: tuple[int, int]
    """CUDA compute capability (major, minor). (0, 0) for CPU."""

    has_gpu: bool
    """True iff a CUDA-capable GPU was detected."""


def _classify_tier(vram_bytes: int, sm_count: int) -> str:
    # TODO: per-GPU MAP for better classification. VRAM thresholds for now.
    gb = vram_bytes / 1e9
    if gb < 8:
        return "consumer_low"
    if gb < 16:
        return "consumer_mid"
    if gb < 32:
        return "consumer_high"
    if gb < 48:
        return "workstation"
    return "datacenter"


def detect_hw() -> DeviceProfile:
    """Detect the current compute device and return a snapshot.

    Runs once at import (see DEVICE), but exposed for manual re-detection
    (e.g. after switching CUDA_VISIBLE_DEVICES). Does NOT allocate GPU memory.
    """
    try:
        import cupy as cp
    except ImportError:
        return DeviceProfile(
            tier="cpu",
            name="CPU only",
            vram_total_gb=0.0,
            vram_free_gb=0.0,
            sm_count=0,
            compute_capability=(0, 0),
            has_gpu=False,
        )

    # cupy can import without a working driver; verify the runtime.
    try:
        dev = cp.cuda.Device()
        free, total = dev.mem_info
        props = cp.cuda.runtime.getDeviceProperties(dev.id)
    except (cp.cuda.runtime.CUDARuntimeError, RuntimeError):
        return DeviceProfile(
            tier="cpu",
            name="CPU only (GPU present but unreachable)",
            vram_total_gb=0.0,
            vram_free_gb=0.0,
            sm_count=0,
            compute_capability=(0, 0),
            has_gpu=False,
        )

    name = props["name"].decode("utf-8", errors="replace")
    return DeviceProfile(
        tier=_classify_tier(total, props["multiProcessorCount"]),
        name=name,
        vram_total_gb=total / 1e9,
        vram_free_gb=free / 1e9,
        sm_count=props["multiProcessorCount"],
        compute_capability=(props["major"], props["minor"]),
        has_gpu=True,
    )


# Singleton — users read this throughout the library.
# For forced re-detection, call detect_hw() and rebind explicitly.
DEVICE: DeviceProfile = detect_hw()
