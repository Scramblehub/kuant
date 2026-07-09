"""Shared pytest fixtures for the kuant test suite.

Fixtures:
  rng      — deterministic numpy Generator (seed 42), for reproducible tests
  has_gpu  — bool, True if cupy imports and a CUDA GPU is available
"""

import numpy as np
import pytest


@pytest.fixture
def rng():
    """Deterministic RNG for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture
def has_gpu():
    """True iff cupy imports and a CUDA-capable GPU is reachable."""
    try:
        import cupy  # noqa: F401

        return cupy.cuda.is_available()
    except (ImportError, RuntimeError):
        return False


@pytest.fixture
def skip_no_gpu(has_gpu):
    """Convenience: skip current test if no GPU available.

    Usage:
        def test_gpu_only(skip_no_gpu):
            ...
    """
    if not has_gpu:
        pytest.skip("GPU (cupy + CUDA) not available")


@pytest.fixture(autouse=True)
def _check_no_gpu_leak(has_gpu):
    """Fail any test that retains > 100 MB of GPU memory after completion.

    Auto-applied to every test (autouse=True) — no opt-in needed.

    Catches the common bugs:
      - Throttle holding array references
      - Kernel wrappers accumulating outputs in a global
      - Memory pool growing unboundedly

    Threshold is 100 MB (not 0) because cupy caches allocations by design;
    small pool growth between tests is normal.
    """
    if not has_gpu:
        yield
        return

    import cupy

    pool = cupy.get_default_memory_pool()
    used_before = pool.used_bytes()
    yield
    used_after = pool.used_bytes()
    delta_bytes = used_after - used_before

    if delta_bytes > 100 * 1024 * 1024:
        pytest.fail(
            f"Potential GPU memory leak: {delta_bytes / 1e6:.1f} MB retained "
            f"after test completed. Check for held array references, "
            f"unbounded state accumulation, or missing __del__ / free."
        )
