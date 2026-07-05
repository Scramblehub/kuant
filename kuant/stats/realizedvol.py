"""OHLC realized-volatility estimators and true-range primitive.

Given an OHLC bar series, five estimators from the microstructure
literature. All return per-bar volatility values that annualize by
multiplying by `sqrt(ann_factor)`, same convention as rollstd.

- `atr`: Average True Range. Not a vol estimator per se; it's the
  moving-average of `max(H-L, |H-C_prev|, |L-C_prev|)`. Used by
  practitioners as a stop-loss and position-sizing input.
- `parkinson`: uses only H and L. Very efficient under a driftless
  geometric Brownian motion.
- `garmanklass`: uses OHLC. Efficient improvement on Parkinson.
- `rogerssatchell`: uses OHLC. Robust to drift (unlike GK).
- `yangzhang`: OHLC + overnight gap. Combines opening jumps with
  intra-day variance. The estimator most robust to both drift and
  opening gaps.

All estimators return per-bar standard deviation. Multiply by
`sqrt(ann_factor)` for annualized figures.

Design: docs/kernels/stats/realizedvol.md.
"""

from __future__ import annotations

import numpy as np

from kuant._validation import (
    require_1d,
    require_equal_length,
    require_ohlc_ordering,
    require_positive,
    warn_window_exceeds_data,
)
from kuant.errors import KuantValueError


def _prepare_ohlc(open_, high, low, close, *, kernel: str):
    """Coerce OHLC arrays to a common shape and validate."""
    O_ = np.asarray(open_, dtype=np.float64)
    H = np.asarray(high, dtype=np.float64)
    L = np.asarray(low, dtype=np.float64)
    C = np.asarray(close, dtype=np.float64)
    require_1d(O_, "open", kernel=kernel)
    require_1d(H, "high", kernel=kernel)
    require_1d(L, "low", kernel=kernel)
    require_1d(C, "close", kernel=kernel)
    require_equal_length(O_, "open", H, "high", kernel=kernel)
    require_equal_length(O_, "open", L, "low", kernel=kernel)
    require_equal_length(O_, "open", C, "close", kernel=kernel)
    return O_, H, L, C


def atr(high, low, close, window: int = 14) -> np.ndarray:
    """Average True Range.

    Parameters
    ----------
    high, low, close : 1D array
    window : int, default 14

    Returns
    -------
    1D np.ndarray of the same length. First `window - 1` entries are NaN.

    Notes
    -----
    Wilder's original ATR uses an EMA-like recursive smoothing; the
    industry-standard closed-form used by TA-Lib and pandas-ta is a
    simple moving average of the True Range. We implement the latter
    for reproducibility.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> close = 100 + np.cumsum(rng.standard_normal(500))
    >>> high = close + np.abs(rng.standard_normal(500))
    >>> low = close - np.abs(rng.standard_normal(500))
    >>> a = atr(high, low, close, window=14)
    >>> np.isnan(a[:13]).all()
    True
    """
    H = np.asarray(high, dtype=np.float64)
    L = np.asarray(low, dtype=np.float64)
    C = np.asarray(close, dtype=np.float64)
    require_1d(H, "high", kernel="atr")
    require_1d(L, "low", kernel="atr")
    require_1d(C, "close", kernel="atr")
    require_equal_length(H, "high", L, "low", kernel="atr")
    require_equal_length(H, "high", C, "close", kernel="atr")
    require_positive(window, "window", kernel="atr", kind="int")

    n = H.size
    prev_close = np.empty(n, dtype=np.float64)
    prev_close[0] = C[0]
    prev_close[1:] = C[:-1]
    tr = np.maximum.reduce([H - L, np.abs(H - prev_close), np.abs(L - prev_close)])
    tr[0] = H[0] - L[0]

    out = np.full(n, np.nan)
    w = int(window)
    if w > n:
        warn_window_exceeds_data(w, n, kernel="atr")
        return out
    # Simple moving average.
    csum = np.concatenate([[0.0], np.cumsum(np.nan_to_num(tr, nan=0.0))])
    nan_count = np.concatenate([[0], np.cumsum(np.isnan(tr).astype(np.int64))])
    for t in range(w - 1, n):
        span_nan = nan_count[t + 1] - nan_count[t - w + 1]
        if span_nan > 0:
            continue
        out[t] = (csum[t + 1] - csum[t - w + 1]) / w
    return out


def parkinson(high, low) -> float:
    """Parkinson volatility estimator using only H and L.

    Returns the ESTIMATED per-bar standard deviation of returns. Assumes
    driftless GBM; biased downward when the underlying has drift.

    Formula: `sigma = sqrt(sum((ln(H/L))^2) / (4 * ln(2) * n))`.
    """
    H = np.asarray(high, dtype=np.float64)
    L = np.asarray(low, dtype=np.float64)
    require_1d(H, "high", kernel="parkinson")
    require_1d(L, "low", kernel="parkinson")
    require_equal_length(H, "high", L, "low", kernel="parkinson")
    if not bool((H > 0).all()) or not bool((L > 0).all()):
        raise KuantValueError(
            "kuant.parkinson: 'high' and 'low' must be strictly positive.  "
            "[KE-VAL-POSITIVE]\n"
            "  → Fix: this estimator is defined on price levels, not returns"
        )
    finite = np.isfinite(H) & np.isfinite(L)
    bad = np.where(finite & (H < L))[0]
    if bad.size:
        i = int(bad[0])
        raise KuantValueError(
            f"kuant.parkinson: OHLC ordering violated at index {i}: "
            f"H={H[i]:.6g} < L={L[i]:.6g}. Parkinson is not defined on "
            f"inverted bars.  [KE-VAL-RANGE]\n"
            f"  → Fix: verify H >= L on every bar before calling"
        )
    log_hl = np.log(H / L)
    finite = np.isfinite(log_hl)
    if not bool(finite.any()):
        return float("nan")
    return float(np.sqrt(np.sum(log_hl[finite] ** 2) / (4.0 * np.log(2.0) * finite.sum())))


def garmanklass(open_, high, low, close) -> float:
    """Garman-Klass volatility estimator using OHLC.

    Improvement on Parkinson under GBM; still biased under drift.
    """
    O_, H, L, C = _prepare_ohlc(open_, high, low, close, kernel="garmanklass")
    if not bool((O_ > 0).all()) or not bool((C > 0).all()):
        raise KuantValueError(
            "kuant.garmanklass: prices must be strictly positive.  "
            "[KE-VAL-POSITIVE]\n"
            "  → Fix: pass OHLC price levels"
        )
    require_ohlc_ordering(O_, H, L, C, kernel="garmanklass")
    hl = np.log(H / L)
    co = np.log(C / O_)
    contrib = 0.5 * hl * hl - (2.0 * np.log(2.0) - 1.0) * co * co
    finite = np.isfinite(contrib)
    if not bool(finite.any()):
        return float("nan")
    return float(np.sqrt(contrib[finite].sum() / finite.sum()))


def rogerssatchell(open_, high, low, close) -> float:
    """Rogers-Satchell volatility estimator using OHLC.

    Unlike Parkinson and Garman-Klass, unbiased under a NON-zero drift.
    """
    O_, H, L, C = _prepare_ohlc(open_, high, low, close, kernel="rogerssatchell")
    if not bool((O_ > 0).all()) or not bool((C > 0).all()):
        raise KuantValueError(
            "kuant.rogerssatchell: prices must be strictly positive.  "
            "[KE-VAL-POSITIVE]\n"
            "  → Fix: pass OHLC price levels"
        )
    require_ohlc_ordering(O_, H, L, C, kernel="rogerssatchell")
    log_ho = np.log(H / O_)
    log_hc = np.log(H / C)
    log_lo = np.log(L / O_)
    log_lc = np.log(L / C)
    contrib = log_ho * log_hc + log_lo * log_lc
    finite = np.isfinite(contrib)
    if not bool(finite.any()):
        return float("nan")
    return float(np.sqrt(contrib[finite].sum() / finite.sum()))


def yangzhang(open_, high, low, close, prev_close=None) -> float:
    """Yang-Zhang volatility estimator.

    Combines Rogers-Satchell with the overnight-return and open-to-open
    variance terms. Best-in-class for equity data with opening gaps.

    Parameters
    ----------
    open_, high, low, close : 1D array
    prev_close : 1D array, optional
        Previous close, aligned to `open_`. If omitted, the previous
        row's close is used and the first bar is dropped from the sum.
    """
    O_, H, L, C = _prepare_ohlc(open_, high, low, close, kernel="yangzhang")
    n = O_.size
    if n < 2:
        raise KuantValueError(
            "kuant.yangzhang: need at least 2 bars to compute the "
            "overnight variance term.  [KE-VAL-RANGE]\n"
            "  → Fix: provide more data"
        )
    if not bool((O_ > 0).all()) or not bool((C > 0).all()):
        raise KuantValueError(
            "kuant.yangzhang: prices must be strictly positive.  "
            "[KE-VAL-POSITIVE]\n"
            "  → Fix: pass OHLC price levels"
        )
    require_ohlc_ordering(O_, H, L, C, kernel="yangzhang")
    if prev_close is None:
        pc = np.empty(n)
        pc[0] = np.nan
        pc[1:] = C[:-1]
    else:
        pc = np.asarray(prev_close, dtype=np.float64)
        require_1d(pc, "prev_close", kernel="yangzhang")
        require_equal_length(O_, "open", pc, "prev_close", kernel="yangzhang")

    # Rogers-Satchell contribution per bar.
    log_ho = np.log(H / O_)
    log_hc = np.log(H / C)
    log_lo = np.log(L / O_)
    log_lc = np.log(L / C)
    rs = log_ho * log_hc + log_lo * log_lc

    log_op = np.log(O_ / pc)  # overnight return
    log_oc = np.log(C / O_)  # open-to-close return

    mask = np.isfinite(rs) & np.isfinite(log_op) & np.isfinite(log_oc)
    if int(mask.sum()) < 2:
        return float("nan")
    n_eff = int(mask.sum())
    k = 0.34 / (1.34 + (n_eff + 1) / (n_eff - 1))
    var_on = float(log_op[mask].var(ddof=1))
    var_oc = float(log_oc[mask].var(ddof=1))
    var_rs = float(rs[mask].mean())
    var_yz = var_on + k * var_oc + (1 - k) * var_rs
    return float(np.sqrt(max(var_yz, 0.0)))


__all__ = ["atr", "parkinson", "garmanklass", "rogerssatchell", "yangzhang"]
