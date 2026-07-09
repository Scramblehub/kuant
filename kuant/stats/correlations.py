"""Nonparametric and asymmetric correlation measures.

Five workhorse variants that complement Pearson `rollcorr`:

- `kendalltau` (Kendall 1938): rank-based; robust to monotonic
  transforms and outliers. Slower than Spearman (O(n log n) with
  merge-sort trick, but our reference is O(n^2)).
- `spearmanrank`: Pearson correlation of ranks. Robust to monotone
  transforms; fast.
- `distancecorr` (Szekely-Rizzo 2007): captures ANY dependence,
  including nonlinear. Zero if and only if X and Y are independent
  (not just uncorrelated). O(n^2) memory.
- `chatterjeexi` (Chatterjee 2020): a simpler alternative to distance
  correlation. Bounded in [-0.5, 1]; O(n log n). Catches nonlinearity
  cheaply.
- `downsidecorr`: Pearson correlation conditional on BOTH series being
  below a threshold (default 0). Captures tail co-movement missed by
  standard correlation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kuant._validation import require_1d
from kuant.errors import KuantValueError


@dataclass
class CorrelationResult:
    coef: float
    p_value: float
    n: int
    method: str

    def summary(self) -> str:
        return (
            f"=== CorrelationResult ({self.method}) ===\n"
            f"coef:     {self.coef:+.4f}\n"
            f"p-value:  {self.p_value:.4g}\n"
            f"n:        {self.n}"
        )


def _check_pair(x, y, kernel: str):
    xf = np.asarray(x, dtype=np.float64)
    yf = np.asarray(y, dtype=np.float64)
    require_1d(xf, "x", kernel=kernel)
    require_1d(yf, "y", kernel=kernel)
    if xf.size != yf.size:
        raise KuantValueError(
            f"kuant.{kernel}: 'x' and 'y' must be equal length; got "
            f"{xf.size} and {yf.size}.  [KE-SHAPE-EQUAL-LEN]"
        )
    mask = np.isfinite(xf) & np.isfinite(yf)
    xf = xf[mask]
    yf = yf[mask]
    if xf.size < 20:
        raise KuantValueError(
            f"kuant.{kernel}: only {xf.size} paired finite values; need "
            f"at least 20.  [KE-VAL-MIN-CLEAN]"
        )
    return xf, yf


def _norm_sf(z: float) -> float:
    from math import erf, sqrt

    return 0.5 * (1.0 - erf(abs(z) / sqrt(2.0)))


# ---------- Kendall's tau --------------------------------------------


def kendalltau(x, y) -> CorrelationResult:
    """Kendall's tau-b rank correlation with tie adjustment.

    Parameters
    ----------
    x, y : 1D arrays of equal length

    Returns
    -------
    CorrelationResult

    References
    ----------
    Kendall 1938, "A new measure of rank correlation."
    """
    xf, yf = _check_pair(x, y, "kendalltau")
    try:
        from scipy.stats import kendalltau as _kt

        tau, p = _kt(xf, yf)
        return CorrelationResult(
            coef=float(tau),
            p_value=float(p),
            n=int(xf.size),
            method="kendall-tau-b",
        )
    except ImportError:
        # Reference O(n^2) implementation with tie adjustment.
        n = xf.size
        conc = disc = 0
        tie_x = tie_y = 0
        for i in range(n - 1):
            dx = np.sign(xf[i + 1 :] - xf[i])
            dy = np.sign(yf[i + 1 :] - yf[i])
            conc += int(np.sum((dx * dy) > 0))
            disc += int(np.sum((dx * dy) < 0))
            tie_x += int(np.sum(dx == 0))
            tie_y += int(np.sum(dy == 0))
        denom = np.sqrt((conc + disc + tie_x) * (conc + disc + tie_y))
        tau = (conc - disc) / denom if denom > 0 else 0.0
        # Asymptotic p-value.
        var = 2 * (2 * n + 5) / (9 * n * (n - 1))
        z = tau / np.sqrt(var) if var > 0 else 0.0
        p = 2.0 * _norm_sf(z)
        return CorrelationResult(
            coef=float(tau),
            p_value=float(p),
            n=int(n),
            method="kendall-tau-b",
        )


# ---------- Spearman rank --------------------------------------------


def spearmanrank(x, y) -> CorrelationResult:
    """Spearman rank correlation.

    Parameters
    ----------
    x, y : 1D arrays of equal length

    Returns
    -------
    CorrelationResult
    """
    xf, yf = _check_pair(x, y, "spearmanrank")
    # Use scipy for tie handling if available, else naive rank + Pearson.
    try:
        from scipy.stats import spearmanr

        rho, p = spearmanr(xf, yf)
        return CorrelationResult(
            coef=float(rho),
            p_value=float(p),
            n=int(xf.size),
            method="spearman-rho",
        )
    except ImportError:
        from scipy.stats import rankdata  # scipy.stats is usually available

        rx = rankdata(xf)
        ry = rankdata(yf)
        rho = float(np.corrcoef(rx, ry)[0, 1])
        # Asymptotic t-based p-value.
        n = xf.size
        if abs(rho) >= 1.0 - 1e-12:
            p = 0.0
        else:
            t = rho * np.sqrt((n - 2) / (1 - rho**2))
            # t distribution with (n-2) df; approximate via normal for large n.
            p = 2.0 * _norm_sf(t)
        return CorrelationResult(
            coef=float(rho),
            p_value=float(p),
            n=int(n),
            method="spearman-rho",
        )


# ---------- distance correlation -------------------------------------


def distancecorr(x, y) -> CorrelationResult:
    """Szekely-Rizzo distance correlation.

    Zero iff X and Y are independent (unlike Pearson, which can be
    zero for dependent nonlinear pairs).

    Parameters
    ----------
    x, y : 1D arrays of equal length. Capped at 2000 rows internally
        (O(n^2) memory).

    Returns
    -------
    CorrelationResult
        `p_value` set to NaN (permutation testing is deferred to the
        caller via `kuant.nulltest.permtest`).

    References
    ----------
    Szekely & Rizzo 2007, "Measuring and testing dependence by
    correlation of distances."
    """
    xf, yf = _check_pair(x, y, "distancecorr")
    if xf.size > 2000:
        # O(n^2) memory ceiling.
        xf = xf[-2000:]
        yf = yf[-2000:]
    n = xf.size

    # Pairwise absolute distance matrices.
    a = np.abs(xf[:, None] - xf[None, :])
    b = np.abs(yf[:, None] - yf[None, :])
    # Double centering.
    a_row = a.mean(axis=1, keepdims=True)
    a_col = a.mean(axis=0, keepdims=True)
    a_grand = a.mean()
    b_row = b.mean(axis=1, keepdims=True)
    b_col = b.mean(axis=0, keepdims=True)
    b_grand = b.mean()
    A = a - a_row - a_col + a_grand
    B = b - b_row - b_col + b_grand
    dcov2 = float(np.mean(A * B))
    dvar_x = float(np.mean(A * A))
    dvar_y = float(np.mean(B * B))
    if dvar_x <= 0 or dvar_y <= 0 or dcov2 < 0:
        return CorrelationResult(
            coef=0.0,
            p_value=float("nan"),
            n=int(n),
            method="distance-corr",
        )
    dcor = np.sqrt(dcov2 / np.sqrt(dvar_x * dvar_y))
    return CorrelationResult(
        coef=float(dcor),
        p_value=float("nan"),
        n=int(n),
        method="distance-corr",
    )


# ---------- Chatterjee's xi ------------------------------------------


def chatterjeexi(x, y) -> CorrelationResult:
    """Chatterjee 2020 rank-based coefficient of correlation.

    A simple statistic that catches nonlinear dependence and equals 1
    when Y is a measurable function of X. Bounded in `[-0.5, 1]`.

    Parameters
    ----------
    x, y : 1D arrays of equal length

    Returns
    -------
    CorrelationResult
        `p_value` from Chatterjee's asymptotic null distribution.

    References
    ----------
    Chatterjee 2020, "A new coefficient of correlation."
    """
    xf, yf = _check_pair(x, y, "chatterjeexi")
    n = xf.size
    # Sort y by x.
    order = np.argsort(xf, kind="stable")
    y_sorted = yf[order]
    try:
        from scipy.stats import rankdata

        r = rankdata(y_sorted)
    except ImportError:
        # Naive rank if scipy missing.
        r = np.zeros(n)
        r[np.argsort(y_sorted, kind="stable")] = np.arange(1, n + 1)
    xi = 1.0 - 3.0 * np.sum(np.abs(np.diff(r))) / (n * n - 1)
    # Asymptotic distribution under H0: xi ~ N(0, 2/5 / n).
    z = xi * np.sqrt(5 * n / 2.0)
    p = 2.0 * _norm_sf(z)
    return CorrelationResult(
        coef=float(xi),
        p_value=float(p),
        n=int(n),
        method="chatterjee-xi",
    )


# ---------- Downside correlation -------------------------------------


def downsidecorr(x, y, *, threshold: float = 0.0) -> CorrelationResult:
    """Pearson correlation conditional on BOTH series being below
    `threshold`. Captures tail co-movement / crash-correlation.

    Parameters
    ----------
    x, y : 1D arrays of equal length
    threshold : float, default 0.0
        Both x_i < threshold AND y_i < threshold required for inclusion.

    Returns
    -------
    CorrelationResult
        `p_value` is the two-sided asymptotic Pearson test on the
        downside subsample.
    """
    xf, yf = _check_pair(x, y, "downsidecorr")
    mask = (xf < float(threshold)) & (yf < float(threshold))
    n_down = int(mask.sum())
    if n_down < 20:
        return CorrelationResult(
            coef=float("nan"),
            p_value=float("nan"),
            n=int(n_down),
            method="downside-corr",
        )
    xd = xf[mask]
    yd = yf[mask]
    coef = float(np.corrcoef(xd, yd)[0, 1])
    if abs(coef) >= 1.0 - 1e-12:
        p = 0.0
    else:
        t = coef * np.sqrt((n_down - 2) / (1 - coef**2))
        p = 2.0 * _norm_sf(t)
    return CorrelationResult(
        coef=float(coef),
        p_value=float(p),
        n=int(n_down),
        method="downside-corr",
    )


__all__ = [
    "CorrelationResult",
    "kendalltau",
    "spearmanrank",
    "distancecorr",
    "chatterjeexi",
    "downsidecorr",
]
