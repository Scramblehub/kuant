"""Multifractal detrended fluctuation analysis (Kantelhardt 2002).

Extension of DFA to multifractal scaling. Computes the q-dependent
generalized Hurst exponent h(q):

- h(q) constant across q -> monofractal (classical DFA)
- h(q) varies with q -> multifractal; range of variation quantifies
  multifractality

For financial time series, multifractality is often present. The
width of the singularity spectrum (max h(q) - min h(q)) is a
diagnostic of intermittency and long-range correlations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kuant._validation import require_1d, require_range
from kuant.errors import KuantValueError


@dataclass
class MfdfaResult:
    q_values: np.ndarray
    h_q: np.ndarray
    multifractal_width: float
    F_q_s: np.ndarray  # (n_q, n_scales)
    scales: np.ndarray

    def summary(self) -> str:
        return (
            "=== MfdfaResult ===\n"
            f"q values:        {list(self.q_values)}\n"
            f"h(q):            {list(np.round(self.h_q, 4))}\n"
            f"h(2) (DFA):      {self.h_q[np.argmin(np.abs(self.q_values - 2))]:.4f}\n"
            f"multifractal width:  {self.multifractal_width:.4f}"
        )


def mfdfa(
    x,
    *,
    q_values=None,
    scales=None,
    order: int = 1,
) -> MfdfaResult:
    """Multifractal DFA.

    Parameters
    ----------
    x : 1D array
    q_values : sequence of float, optional
        Moment orders to evaluate. Default: [-3, -2, -1, 0.5, 1, 2, 3, 4].
        q = 2 is classical DFA.
    scales : sequence of int, optional
        Segment sizes. Default: log-spaced 10 to n/4.
    order : int, default 1
        Polynomial detrending order (1 = linear).

    Returns
    -------
    MfdfaResult

    References
    ----------
    Kantelhardt et al 2002, "Multifractal detrended fluctuation
    analysis of nonstationary time series."
    """
    arr = np.asarray(x, dtype=np.float64)
    require_1d(arr, "x", kernel="mfdfa")
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n < 200:
        raise KuantValueError(
            f"kuant.mfdfa: only {n} finite values; need at least 200.  " f"[KE-VAL-MIN-CLEAN]"
        )
    require_range(order, "order", kernel="mfdfa", lo=1, hi=5)

    if q_values is None:
        q_values = np.array([-3.0, -2.0, -1.0, 0.5, 1.0, 2.0, 3.0, 4.0])
    else:
        q_values = np.asarray(q_values, dtype=np.float64)
    if scales is None:
        scales = np.unique(np.round(np.logspace(np.log10(10), np.log10(n // 4), 15)).astype(int))
    else:
        scales = np.asarray(scales, dtype=int)

    # Integrated series.
    Y = np.cumsum(arr - arr.mean())

    F_q_s = np.full((q_values.size, scales.size), np.nan, dtype=np.float64)

    for si, s in enumerate(scales):
        s = int(s)
        if s < order + 2 or s > n // 2:
            continue
        n_segments = n // s
        # Forward + backward pass (Kantelhardt style).
        var_segments = []
        for seg in range(n_segments):
            i0 = seg * s
            seg_data = Y[i0 : i0 + s]
            t = np.arange(s)
            # Poly-fit detrend.
            coefs = np.polyfit(t, seg_data, order)
            trend = np.polyval(coefs, t)
            resid = seg_data - trend
            var_segments.append(float(np.mean(resid * resid)))
        for seg in range(n_segments):
            i0 = n - (seg + 1) * s
            seg_data = Y[i0 : i0 + s]
            t = np.arange(s)
            coefs = np.polyfit(t, seg_data, order)
            trend = np.polyval(coefs, t)
            resid = seg_data - trend
            var_segments.append(float(np.mean(resid * resid)))
        var_arr = np.asarray(var_segments)
        var_arr = var_arr[np.isfinite(var_arr) & (var_arr > 0)]
        if var_arr.size < 3:
            continue
        for qi, q in enumerate(q_values):
            if abs(q) < 1e-8:
                # Special q=0 case: log-average.
                F_q_s[qi, si] = float(np.exp(0.5 * np.mean(np.log(var_arr))))
            else:
                F_q_s[qi, si] = float(np.mean(var_arr ** (q / 2.0)) ** (1.0 / q))

    # Fit log(F_q(s)) vs log(s) for each q -> h(q).
    log_s = np.log(scales.astype(np.float64))
    h_q = np.zeros(q_values.size, dtype=np.float64)
    for qi in range(q_values.size):
        row = F_q_s[qi]
        mask = np.isfinite(row) & (row > 0)
        if mask.sum() < 4:
            h_q[qi] = np.nan
            continue
        slope, _ = np.polyfit(log_s[mask], np.log(row[mask]), 1)
        h_q[qi] = float(slope)

    valid_hq = h_q[np.isfinite(h_q)]
    width = float(valid_hq.max() - valid_hq.min()) if valid_hq.size > 0 else float("nan")

    return MfdfaResult(
        q_values=q_values,
        h_q=h_q,
        multifractal_width=width,
        F_q_s=F_q_s,
        scales=scales,
    )


__all__ = ["MfdfaResult", "mfdfa"]
