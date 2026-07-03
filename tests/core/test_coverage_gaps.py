"""Targeted tests for real coverage gaps (not defensive-branch boilerplate).

Coverage audit surfaced two paths uncovered by kernel-specific tests:

  1. `kuant.core.normcdf` chunked GPU path — fires when the throttle
     layer decides a batch is larger than one chunk. Normally hidden
     unless the input is very large.

  2. `kuant.core._special_bridge` H↔D fallback for functions with no
     cupyx equivalent — currently `stdtrit` for `tppf`. Fires when a
     cupy input is passed to `tppf`.

Both need a GPU present; skip cleanly otherwise.
"""

from __future__ import annotations

import numpy as np


def test_normcdf_chunked_gpu_path(skip_no_gpu):
    """Exercise the chunked GPU path in kuant.core.normcdf.

    Force a large array; on any GPU the throttle target of ~50ms
    should still route a big enough input through the chunk loop.
    """
    import cupy as cp

    from kuant.core import normcdf
    from kuant.queueing import THROTTLE

    # Push a large array; if the throttle decides one chunk is enough,
    # we still exercise the GPU pricing path. If it decides to chunk,
    # we exercise the chunked path. Either way, values must match CPU.
    saved_timings = {k: list(v) for k, v in THROTTLE._timings.items()}
    try:
        THROTTLE._timings.clear()  # empty history → conservative chunks
        n = 2_000_000  # ~16 MB per array in float64
        x_cpu = np.linspace(-5, 5, n)
        x_gpu = cp.asarray(x_cpu)
        r_cpu = normcdf(x_cpu)
        r_gpu = cp.asnumpy(normcdf(x_gpu))
        np.testing.assert_allclose(r_cpu, r_gpu, atol=1e-12)
    finally:
        THROTTLE._timings.clear()
        THROTTLE._timings.update(saved_timings)


def test_tppf_gpu_input_via_hd_fallback(skip_no_gpu):
    """Exercise the H↔D fallback in kuant.core._special_bridge.

    tppf → stdtrit is not in cupyx.scipy.special, so a cupy input
    routes through the fallback: .get() → scipy call → cp.asarray().
    """
    import cupy as cp

    from kuant.core import tppf

    p_gpu = cp.asarray([0.1, 0.25, 0.5, 0.75, 0.9])
    df_gpu = cp.asarray([5.0, 5.0, 5.0, 5.0, 5.0])
    result = tppf(p_gpu, df_gpu)

    # Result should be a cupy array (routing preserved backend)
    assert isinstance(result, cp.ndarray)

    # Values match scipy
    from scipy.stats import t as sp_t

    expected = sp_t.ppf([0.1, 0.25, 0.5, 0.75, 0.9], df=5.0)
    np.testing.assert_allclose(cp.asnumpy(result), expected, atol=1e-10)


def test_special_bridge_binary_dispatch_gpu(skip_no_gpu):
    """Exercise `_dispatch_binary` on GPU input — used by gammaln, betainc etc.

    Even though cupyx.scipy.special has gammaln + betainc (so this hits
    the "spx is available" branch, not the H↔D fallback), we still
    want a GPU call routed through the bridge to prove it doesn't
    error and returns a cupy array.
    """
    import cupy as cp

    from kuant.core import tcdf

    x_gpu = cp.asarray([-1.0, 0.0, 1.0, 2.0])
    df_gpu = cp.asarray([5.0, 5.0, 5.0, 5.0])
    result = tcdf(x_gpu, df_gpu)
    assert isinstance(result, cp.ndarray)

    from scipy.stats import t as sp_t

    expected = sp_t.cdf([-1.0, 0.0, 1.0, 2.0], df=5.0)
    np.testing.assert_allclose(cp.asnumpy(result), expected, atol=1e-10)


def test_normcdf_empty_array():
    """Cover the `if arr.size == 0` early return path."""
    from kuant.core import normcdf

    result = normcdf(np.array([]))
    assert result.shape == (0,)
