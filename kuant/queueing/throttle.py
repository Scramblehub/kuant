"""Chunk sizer — decides how to slice large batches to fit VRAM safely.

DESIGN INVARIANTS (leak-safety):
  1. State is ONLY primitives (dict[str, list[tuple[int, float]]]).
     No array references. Bugs here cannot hold GPU memory alive.
  2. State is BOUNDED by max_history.
  3. All methods are pure w.r.t. GPU memory.
  4. suggest_chunk_size returns an int; caller does slicing and allocation.

The throttle is deliberately DUMB in v1: static heuristics only. Timing
recording exists so we can add adaptation later, without changing the
kernel-side API.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .hardware import DEVICE


# Tier → default VRAM budget per call (fraction of free memory).
# Conservative because: concurrent kernels, pool fragmentation, OS/display overhead.
_VRAM_SAFETY_BY_TIER: dict[str, float] = {
    "cpu": 1.0,
    "consumer_low": 0.5,  # 3060 users often share GPU with display
    "consumer_mid": 0.6,
    "consumer_high": 0.7,
    "workstation": 0.75,
    "datacenter": 0.8,  # typically dedicated
}


@dataclass
class ChunkSizer:
    """Decides chunk sizes for kernels that split large batches.

    Holds NO references to arrays — only primitive timing metadata.
    """

    max_history: int = 20
    """Timing samples kept per kernel_id. Bounded so long-running processes
    can't grow the dict indefinitely."""

    _timings: dict[str, list[tuple[int, float]]] = field(default_factory=dict)
    """Per-kernel history: {kernel_id: [(n_elems, elapsed_ms), ...]}.
    Primitives only — this is the invariant that makes bugs here incapable
    of leaking GPU memory."""

    def record(self, kernel_id: str, n_elems: int, elapsed_ms: float) -> None:
        """Log one completed kernel call. Numbers only, never arrays."""
        buf = self._timings.setdefault(kernel_id, [])
        buf.append((int(n_elems), float(elapsed_ms)))
        if len(buf) > self.max_history:
            buf.pop(0)

    def suggest_chunk_size(
        self,
        kernel_id: str,
        total_elems: int,
        bytes_per_elem: int,
        target_ms: float = 50.0,
    ) -> int:
        """Return elements per chunk. No allocation happens here.

        Heuristic:
          1. VRAM ceiling: free * safety_fraction / bytes_per_elem
          2. If ≥3 timing samples, cap by target_ms budget
          3. Never < 1024 (launch overhead dominates)
          4. Never > total_elems

        Parameters
        ----------
        kernel_id : str
            Key for timing lookups. Convention: match the function name.
        total_elems : int
            Total elements to process.
        bytes_per_elem : int
            VRAM cost per element. Typical unary: `x.itemsize * 2` (in + out).
        target_ms : float, default 50
            Target wall-clock per chunk. Kept short so the GPU stays
            responsive for other jobs.
        """
        if not DEVICE.has_gpu:
            return int(total_elems)

        # VRAM ceiling. Query fresh — free may have changed since detect_hw().
        import cupy as cp

        free_bytes = cp.cuda.Device().mem_info[0]
        safety = _VRAM_SAFETY_BY_TIER.get(DEVICE.tier, 0.5)
        vram_ceiling = int((free_bytes * safety) // max(bytes_per_elem, 1))

        # Latency ceiling from timing history (if any).
        latency_ceiling = int(total_elems)
        buf = self._timings.get(kernel_id)
        if buf and len(buf) >= 3:
            rates = sorted((n / ms) for n, ms in buf if ms > 0.01)
            if rates:
                median_rate = rates[len(rates) // 2]
                latency_ceiling = int(median_rate * target_ms)

        MIN_CHUNK = 1024
        return max(MIN_CHUNK, min(vram_ceiling, latency_ceiling, int(total_elems)))


# Module-level singleton. Kernels do `from kuant.queueing import THROTTLE`.
# Process-local by design — not shared across unrelated processes.
THROTTLE: ChunkSizer = ChunkSizer()
