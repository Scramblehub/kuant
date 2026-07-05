"""Information Coefficient decay curve.

For each forecast horizon `h`, compute the Spearman rank correlation
between:

    signal[t]                           (the signal AT time t)
    cumulative forward return over [t+1, t+h]

That correlation is the "Information Coefficient" (IC) at horizon `h`.
Sweeping `h` produces the DECAY CURVE — how quickly the signal's
edge evaporates.

Two standard outputs per horizon:

- **`ic`** — the Spearman correlation itself. Positive means "high
  signal → high forward return". Financial industry rule of thumb:
  |IC| > 0.02 is "real", |IC| > 0.05 is "very good".
- **`ic_tstat`** — IC divided by its approximate standard error
  `1/sqrt(n)`. A t-statistic > 2 is roughly the threshold for
  "distinguishable from zero" given the sample size.

Warns via `KuantNumericWarning` when the standard error at ANY tested
horizon exceeds the IC magnitude — that's "indistinguishable from
noise at this sample size", the failure mode most likely to make a
sales pitch disappear on real capital.

Design: docs/kernels/signals/icdecay.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kuant._validation import (
    require_1d,
    require_dep,
    require_equal_length,
    require_positive,
    warn_kuant,
)
from kuant.errors import KuantNumericWarning, KuantValueError


@dataclass
class ICDecayResult:
    """Per-horizon information coefficient results.

    Attributes
    ----------
    horizons : 1D np.ndarray
        The horizons tested.
    ic : 1D np.ndarray
        Spearman IC at each horizon.
    ic_stderr : 1D np.ndarray
        Approximate standard error `1/sqrt(n)` at each horizon.
    ic_tstat : 1D np.ndarray
        `ic / ic_stderr` per horizon. |t| > 2 is the rough "real" threshold.
    n : 1D np.ndarray
        Number of overlapping observations used at each horizon.
    """

    horizons: np.ndarray
    ic: np.ndarray
    ic_stderr: np.ndarray
    ic_tstat: np.ndarray
    n: np.ndarray

    def summary(self) -> str:
        parts = ["=== ICDecayResult ===", f"{'horizon':>8s} {'IC':>10s} {'t-stat':>10s} {'n':>8s}"]
        for h, ic, t, n in zip(self.horizons, self.ic, self.ic_tstat, self.n):
            parts.append(f"{int(h):>8d} {ic:>+10.4f} {t:>+10.2f} {int(n):>8d}")
        # Peak IC / horizon.
        peak_idx = int(np.argmax(np.abs(self.ic)))
        parts.append("")
        parts.append(
            f"peak |IC|:  {self.ic[peak_idx]:+.4f} at horizon "
            f"{int(self.horizons[peak_idx])} (t={self.ic_tstat[peak_idx]:+.2f})"
        )
        return "\n".join(parts)

    def to_parquet(self, path) -> None:
        """Write the decay curve to parquet. Requires `pyarrow`."""
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as e:
            require_dep(
                "pyarrow",
                kernel="icdecay.to_parquet",
                install="pip install pyarrow",
                cause=e,
            )
        table = pa.table(
            {
                "horizon": pa.array(self.horizons),
                "ic": pa.array(self.ic),
                "ic_stderr": pa.array(self.ic_stderr),
                "ic_tstat": pa.array(self.ic_tstat),
                "n": pa.array(self.n),
            }
        )
        pq.write_table(table, path)


def icdecay(
    signal,
    forward_returns,
    horizons=(1, 5, 21, 63),
) -> ICDecayResult:
    """Spearman IC at multiple forward horizons.

    Parameters
    ----------
    signal : 1D array, length T
        Signal value at each time `t`.
    forward_returns : 1D array, length T
        Periodic returns. `forward_returns[t]` is interpreted as the
        return realized between time `t` and time `t+1`. Cumulative
        returns over horizon `h` are computed by summing consecutive
        values — appropriate when the input is log-returns. For simple
        returns this is an approximation that gets worse as `h` grows;
        for large horizons pass log-returns.
    horizons : sequence of int, default (1, 5, 21, 63)
        Forecast horizons in periods. Common: 1 (daily), 5 (weekly),
        21 (monthly), 63 (quarterly), 252 (annual).

    Returns
    -------
    ICDecayResult
        Per-horizon `ic`, `ic_stderr`, `ic_tstat`, `n`.

    Warnings
    --------
    - `KuantNumericWarning` (`KW-IC-NOISE-FLOOR`) at any horizon where
      `abs(ic) < ic_stderr` — indistinguishable from zero at this
      sample size.

    Notes
    -----
    - Requires `scipy.stats.spearmanr`. Lazy import.
    - Rows with NaN in signal or in the forward-return sum are dropped
      before ranking.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> T = 500
    >>> # Synthetic setup: signal predicts next-day return with real IC.
    >>> signal = rng.standard_normal(T)
    >>> forward_ret = 0.02 * signal + rng.standard_normal(T) * 0.02
    >>> r = icdecay(signal, forward_ret, horizons=(1, 5, 21))
    >>> r.ic[0] > 0.2                                          # positive IC at h=1
    True
    """
    try:
        from scipy.stats import spearmanr
    except ImportError as e:
        require_dep(
            "scipy",
            kernel="icdecay",
            install="pip install scipy",
            cause=e,
        )

    signal_arr = np.asarray(signal, dtype=np.float64)
    ret_arr = np.asarray(forward_returns, dtype=np.float64)
    require_1d(signal_arr, "signal", kernel="icdecay")
    require_1d(ret_arr, "forward_returns", kernel="icdecay")
    require_equal_length(signal_arr, "signal", ret_arr, "forward_returns", kernel="icdecay")

    horizons_arr = np.asarray(list(horizons), dtype=np.int64)
    if horizons_arr.size == 0:
        raise KuantValueError(
            "kuant.icdecay: 'horizons' must contain at least one entry.  "
            "[KE-VAL-RANGE]\n"
            "  → Fix: pass a non-empty sequence like (1, 5, 21, 63)"
        )
    for h in horizons_arr:
        require_positive(int(h), "horizon", kernel="icdecay", kind="int")
        if int(h) >= signal_arr.size:
            raise KuantValueError(
                f"kuant.icdecay: horizon {int(h)} >= len(signal) "
                f"({signal_arr.size}); need at least one overlapping "
                f"observation.  [KE-VAL-RANGE]\n"
                f"  → Fix: use shorter horizons, or provide more data"
            )

    T = signal_arr.size
    ic_out = np.full(horizons_arr.size, np.nan)
    stderr_out = np.full(horizons_arr.size, np.nan)
    tstat_out = np.full(horizons_arr.size, np.nan)
    n_out = np.zeros(horizons_arr.size, dtype=np.int64)

    # Cumulative sum trick for the horizon-h forward return.
    # Prepend a 0 so csum[t] = sum(ret_arr[0..t-1]), then cum_ret_h[t] =
    # csum[t+h+1] - csum[t+1] gives sum(ret_arr[t+1..t+h]).
    csum = np.concatenate([[0.0], np.cumsum(np.nan_to_num(ret_arr, nan=0.0))])
    # NaN in ret_arr should propagate — track a NaN mask separately.
    isnan = np.isnan(ret_arr)
    csum_nan = np.concatenate([[0], np.cumsum(isnan.astype(np.int64))])

    for i, h in enumerate(horizons_arr):
        h = int(h)
        # Windows starting at t map to signal[t] vs sum(ret[t+1..t+h]).
        # Valid t ∈ [0, T - h - 1].
        end = T - h
        if end <= 0:
            continue
        # Forward return over [t+1, t+h] = csum[t+h+1] - csum[t+1].
        sig = signal_arr[:end]
        fwd = csum[1 + h : 1 + end + h] - csum[1 : 1 + end]
        # Any NaN in ret_arr[t+1..t+h] means fwd[t] is NaN.
        nan_in_window = csum_nan[1 + h : 1 + end + h] - csum_nan[1 : 1 + end]
        clean = np.isfinite(sig) & (nan_in_window == 0)
        n = int(clean.sum())
        n_out[i] = n
        if n < 3:
            continue
        rho, _ = spearmanr(sig[clean], fwd[clean])
        if np.isfinite(rho):
            ic_out[i] = float(rho)
            se = 1.0 / np.sqrt(n)
            stderr_out[i] = float(se)
            tstat_out[i] = float(rho / se)

    # Noise-floor warning.
    finite_ic = np.isfinite(ic_out)
    if finite_ic.any():
        below_noise = np.abs(ic_out[finite_ic]) < stderr_out[finite_ic]
        if below_noise.any():
            first_bad = int(np.where(finite_ic)[0][int(np.argmax(below_noise))])
            warn_kuant(
                kernel="icdecay",
                code="KW-IC-NOISE-FLOOR",
                what=(
                    f"IC at horizon {int(horizons_arr[first_bad])} is "
                    f"{ic_out[first_bad]:+.4f} with stderr "
                    f"{stderr_out[first_bad]:.4f}; indistinguishable from "
                    f"noise (n={int(n_out[first_bad])})"
                ),
                fix=(
                    "collect more data (stderr shrinks like 1/√n), or "
                    "accept that this signal has no edge at this horizon"
                ),
                category=KuantNumericWarning,
            )

    return ICDecayResult(
        horizons=horizons_arr,
        ic=ic_out,
        ic_stderr=stderr_out,
        ic_tstat=tstat_out,
        n=n_out,
    )


__all__ = ["icdecay", "ICDecayResult"]
