"""Discrete wavelet transform for time-series decomposition.

Multi-level Haar (or optional Daubechies via PyWavelets) decomposition
that separates a signal into detail coefficients at each dyadic scale
plus a coarse approximation. Useful for:

- Trend / noise separation at chosen frequency bands.
- Feature engineering: wavelet variance per scale as inputs to a
  predictive model.
- Denoising via soft-threshold on detail coefficients.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kuant._validation import require_1d, require_positive
from kuant.errors import KuantValueError


@dataclass
class WaveletResult:
    approximation: np.ndarray  # coarse (low-freq) coefficients
    details: list  # detail coefs per scale, list of 1D arrays
    variances: np.ndarray  # variance of details per scale
    n_scales: int
    kernel: str

    def summary(self) -> str:
        return (
            "=== WaveletResult ===\n"
            f"n scales:      {self.n_scales}\n"
            f"kernel:        {self.kernel}\n"
            f"approximation: len {self.approximation.size}\n"
            f"variance/scale: {np.round(self.variances, 5)}"
        )


def _haar_step(x):
    if x.size % 2 == 1:
        x = x[:-1]
    a = (x[::2] + x[1::2]) / np.sqrt(2)
    d = (x[::2] - x[1::2]) / np.sqrt(2)
    return a, d


def wavelet(x, *, n_scales: int = 5, kernel: str = "haar") -> WaveletResult:
    """Multi-level wavelet transform.

    Parameters
    ----------
    x : 1D array
    n_scales : int, default 5
        Number of decomposition levels. Capped internally at log2(n).
    kernel : {"haar", "db2", "db4"}, default "haar"
        Haar is implemented natively; db2 / db4 require pywt.

    Returns
    -------
    WaveletResult

    References
    ----------
    Mallat 1989, "A theory for multiresolution signal decomposition."
    """
    arr = np.asarray(x, dtype=np.float64)
    require_1d(arr, "x", kernel="wavelet")
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n < 32:
        raise KuantValueError(
            f"kuant.wavelet: only {n} finite values; need at least 32.  " f"[KE-VAL-MIN-CLEAN]"
        )
    require_positive(n_scales, "n_scales", kernel="wavelet", kind="int")
    max_scales = int(np.floor(np.log2(n)))
    n_scales = min(int(n_scales), max_scales)

    if kernel == "haar":
        cur = arr.copy()
        details = []
        for _ in range(n_scales):
            if cur.size < 2:
                break
            a, d = _haar_step(cur)
            details.append(d)
            cur = a
        approximation = cur
    elif kernel in ("db2", "db4"):
        try:
            import pywt
        except ImportError as e:
            raise KuantValueError(
                f"kuant.wavelet: kernel {kernel!r} requires PyWavelets.  "
                f"[KE-DEP-MISSING]\n"
                f"  -> Fix: pip install PyWavelets"
            ) from e
        coefs = pywt.wavedec(arr, wavelet=kernel, level=int(n_scales))
        approximation = np.asarray(coefs[0])
        details = [np.asarray(c) for c in coefs[1:]]
    else:
        raise KuantValueError(
            f"kuant.wavelet: 'kernel' must be one of {{haar, db2, db4}}; "
            f"got {kernel!r}.  [KE-VAL-RANGE]"
        )

    variances = np.array([float(np.var(d, ddof=1)) if d.size > 1 else 0.0 for d in details])
    return WaveletResult(
        approximation=approximation,
        details=details,
        variances=variances,
        n_scales=len(details),
        kernel=str(kernel),
    )


__all__ = ["WaveletResult", "wavelet"]
