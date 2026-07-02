"""Chunk sizer — decides how to slice large batches to fit VRAM safely.

DESIGN INVARIANTS (guard against the leak footguns we discussed):

  1. State is ONLY primitives (dict[str, list[tuple[int, float]]]).
     No array references. Impossible for a bug here to hold GPU memory alive.

  2. State is BOUNDED by MAX_HISTORY. Long-running processes can't grow it
     unboundedly.

  3. All methods are pure with respect to GPU memory. Nothing here allocates
     or holds GPU arrays.

  4. `suggest_chunk_size` returns an int. Just an int. The caller does the
     actual slicing and allocation. Separation of concerns.

The throttle is deliberately DUMB in v1: static heuristics only, no learned
adaptation. Timing recording exists so we can add learning later, but the
current `suggest_chunk_size` doesn't use it. If we need dynamic adaptation,
we add it in one place without changing the kernel-side API.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .hardware import DEVICE


# Tier → default VRAM budget per kernel call (fraction of free memory).
# Conservative fractions because:
#   - We may have multiple kernels concurrent
#   - cupy memory pool fragments; usable is < physical free
#   - OS + display + other processes eat some
_VRAM_SAFETY_BY_TIER: dict[str, float] = {
    "cpu": 1.0,           # unused
    "consumer_low": 0.5,  # 3060 users often share GPU with display
    "consumer_mid": 0.6,
    "consumer_high": 0.7,
    "workstation": 0.75,
    "datacenter": 0.8,    # data-center cards typically dedicated
}


@dataclass
class ChunkSizer:
    """Decides chunk sizes for kernels that need to split large batches.

    The class holds NO references to arrays. Only primitive timing metadata.
    """

    max_history: int = 20
    """How many timing samples to keep per kernel_id. Bounded to prevent
    the timing dict from growing without limit in long-running processes."""

    _timings: dict[str, list[tuple[int, float]]] = field(default_factory=dict)
    """Per-kernel timing history: {kernel_id: [(n_elems, elapsed_ms), ...]}.

    ONLY primitives. No array references possible. This is the safety
    invariant that makes bugs in this class incapable of leaking GPU
    memory.
    """

    def record(self, kernel_id: str, n_elems: int, elapsed_ms: float) -> None:
        """Log one completed kernel call. Only takes numbers, never arrays."""
        buf = self._timings.setdefault(kernel_id, [])
        buf.append((int(n_elems), float(elapsed_ms)))
        if len(buf) > self.max_history:
            buf.pop(0)  # bounded — cannot grow unboundedly

    def suggest_chunk_size(
        self,
        kernel_id: str,
        total_elems: int,
        bytes_per_elem: int,
        target_ms: float = 50.0,
    ) -> int:
        """Compute how many elements to process per chunk.

        Returns an int (element count). No allocation happens here.

        The heuristic is intentionally simple:
          1. Compute VRAM ceiling: free * safety_fraction / bytes_per_elem
          2. If we have timing history, cap by target_ms budget
          3. Never smaller than 1024 (kernel launch overhead dominates)
          4. Never larger than total_elems (no point splitting further)

        Callers use it as::

            n = throttle.suggest_chunk_size("mykernel", x.size, x.itemsize * 2)
            for start in range(0, x.size, n):
                ...

        Parameters
        ----------
        kernel_id : str
            Unique key naming this kernel (used for timing lookups).
            Convention: match the function name, e.g. "normcdf" or "bsput".

        total_elems : int
            Total elements the user wants processed. If this fits under the
            ceiling, we return `total_elems` — no chunking needed.

        bytes_per_elem : int
            Estimated VRAM cost per element. Include input + output + any
            scratch. Typical for a simple unary kernel: `x.itemsize * 2`
            (input + output).

        target_ms : float, default 50
            Target wall-clock per chunk. Kept short so the GPU stays
            responsive between chunks (allows other jobs to interleave).
        """
        if not DEVICE.has_gpu:
            # CPU has no VRAM constraint — return everything.
            return int(total_elems)

        # 1. VRAM ceiling based on tier defaults
        # Query fresh at call time — free may have changed since detect_hw()
        import cupy as cp

        free_bytes = cp.cuda.Device().mem_info[0]
        safety = _VRAM_SAFETY_BY_TIER.get(DEVICE.tier, 0.5)
        vram_ceiling = int((free_bytes * safety) // max(bytes_per_elem, 1))

        # 2. Latency ceiling from timing history (if we have any)
        latency_ceiling = int(total_elems)  # no history → don't cap
        buf = self._timings.get(kernel_id)
        if buf and len(buf) >= 3:
            # Median rate = elements per millisecond
            rates = sorted((n / ms) for n, ms in buf if ms > 0.01)
            if rates:
                median_rate = rates[len(rates) // 2]
                latency_ceiling = int(median_rate * target_ms)

        # 3. Never so small that kernel launch overhead dominates.
        MIN_CHUNK = 1024

        # 4. Never larger than total_elems.
        return max(MIN_CHUNK, min(vram_ceiling, latency_ceiling, int(total_elems)))


# Module-level singleton. Kernels do `from kuant.queueing import THROTTLE`.
# Keeping it module-level (not global) means it lives with the process and
# is not shared across concurrent unrelated processes — that's intentional.
THROTTLE: ChunkSizer = ChunkSizer()
