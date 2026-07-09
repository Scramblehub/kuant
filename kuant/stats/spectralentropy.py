"""Spectral entropy: Shannon entropy of the power spectrum.

`spectralentropy(x)` returns the normalized Shannon entropy of the
periodogram of `x`. Bounded in [0, 1]:

- 0: pure sinusoid (all power at one frequency)
- 1: white noise (uniform spectrum)

A concise single-number complexity metric that complements time-domain
kernels (Lyapunov, sample entropy) with a frequency-domain view.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kuant._validation import require_1d
from kuant.errors import KuantValueError


@dataclass
class SpectralEntropyResult:
    entropy: float
    normalized: float
    n_bins: int
    n_samples: int

    def summary(self) -> str:
        return (
            "=== SpectralEntropyResult ===\n"
            f"entropy:      {self.entropy:.4f} nats\n"
            f"normalized:   {self.normalized:.4f}\n"
            f"n bins:       {self.n_bins}\n"
            f"n samples:    {self.n_samples}"
        )


def spectralentropy(x, *, detrend: bool = True) -> SpectralEntropyResult:
    """Shannon entropy of the power spectrum.

    Parameters
    ----------
    x : 1D array
    detrend : bool, default True
        Subtract mean before computing FFT.

    Returns
    -------
    SpectralEntropyResult

    References
    ----------
    Inouye et al 1991, "Quantification of EEG irregularity by use of the
    entropy of the power spectrum."
    """
    arr = np.asarray(x, dtype=np.float64)
    require_1d(arr, "x", kernel="spectralentropy")
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n < 32:
        raise KuantValueError(
            f"kuant.spectralentropy: only {n} finite values; need at least "
            f"32.  [KE-VAL-MIN-CLEAN]"
        )
    if detrend:
        arr = arr - arr.mean()
    # Real FFT power (drop DC).
    fft_vals = np.fft.rfft(arr)
    power = np.abs(fft_vals) ** 2
    power = power[1:]  # drop DC
    total = power.sum()
    if total < 1e-15:
        return SpectralEntropyResult(
            entropy=float("nan"),
            normalized=float("nan"),
            n_bins=int(power.size),
            n_samples=int(n),
        )
    p = power / total
    p_nonzero = p[p > 0]
    ent = float(-np.sum(p_nonzero * np.log(p_nonzero)))
    max_ent = float(np.log(p.size))
    normalized = ent / max_ent if max_ent > 0 else 0.0
    return SpectralEntropyResult(
        entropy=ent,
        normalized=normalized,
        n_bins=int(p.size),
        n_samples=int(n),
    )


__all__ = ["SpectralEntropyResult", "spectralentropy"]
