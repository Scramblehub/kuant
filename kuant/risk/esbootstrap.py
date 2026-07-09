"""Bootstrap Expected Shortfall with confidence interval.

Expected Shortfall (ES) is the average loss GIVEN that a loss exceeds
the VaR threshold. Because ES is a tail statistic, its sample estimate
has high variance; a bootstrap CI quantifies that uncertainty.

Uses (moving-block or IID) bootstrap resamples of the return series,
each computing ES at the requested confidence level. The empirical
percentile of the bootstrap ES distribution gives the CI.

Sign convention: ES reported as POSITIVE loss magnitude.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kuant._validation import require_1d, require_positive, require_range
from kuant.errors import KuantValueError


@dataclass
class EsBootstrapResult:
    es_point: float
    es_ci_low: float
    es_ci_high: float
    var_point: float
    ci_alpha: float
    conf_alpha: float
    n_boot: int
    block_size: int
    n: int

    def summary(self) -> str:
        return (
            "=== EsBootstrapResult ===\n"
            f"ES (point):         {self.es_point:+.6f}\n"
            f"ES CI [{100 * self.ci_alpha:.0f}%]: "
            f"[{self.es_ci_low:+.6f}, {self.es_ci_high:+.6f}]\n"
            f"VaR (point):        {self.var_point:+.6f}\n"
            f"conf alpha:         {self.conf_alpha}\n"
            f"n bootstrap:        {self.n_boot}\n"
            f"block size:         {self.block_size}\n"
            f"n:                  {self.n}"
        )


def _es_from_sample(arr: np.ndarray, alpha: float) -> tuple[float, float]:
    """Return (VaR, ES) both as positive loss magnitudes."""
    losses = -arr
    q = np.quantile(losses, alpha)
    tail = losses[losses >= q]
    if tail.size == 0:
        return float(q), float(q)
    return float(q), float(tail.mean())


def esbootstrap(
    returns,
    *,
    conf_alpha: float = 0.95,
    ci_alpha: float = 0.95,
    n_boot: int = 500,
    block_size: int = 21,
    seed: int = 0,
) -> EsBootstrapResult:
    """Bootstrap Expected Shortfall.

    Parameters
    ----------
    returns : 1D array
    conf_alpha : float, default 0.95
        VaR / ES confidence level.
    ci_alpha : float, default 0.95
        Bootstrap CI level.
    n_boot : int, default 500
    block_size : int, default 21
        Moving-block bootstrap block length. Set to 1 for iid.
    seed : int, default 0

    Returns
    -------
    EsBootstrapResult

    References
    ----------
    Kunsch 1989 (moving-block bootstrap); Politis-Romano 1994
    (stationary bootstrap); Cont-Deguest-Scandolo 2010 (risk-measure
    robustness).
    """
    arr = np.asarray(returns, dtype=np.float64)
    require_1d(arr, "returns", kernel="esbootstrap")
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n < 100:
        raise KuantValueError(
            f"kuant.esbootstrap: only {n} finite values; need at least " f"100.  [KE-VAL-MIN-CLEAN]"
        )
    require_range(conf_alpha, "conf_alpha", kernel="esbootstrap", lo=0.5, hi=0.9999)
    require_range(ci_alpha, "ci_alpha", kernel="esbootstrap", lo=0.5, hi=0.9999)
    require_positive(n_boot, "n_boot", kernel="esbootstrap", kind="int")
    require_positive(block_size, "block_size", kernel="esbootstrap", kind="int")
    if block_size > n // 2:
        raise KuantValueError(
            f"kuant.esbootstrap: block_size ({block_size}) must be <= n/2 "
            f"({n // 2}).  [KE-VAL-RANGE]"
        )

    var_point, es_point = _es_from_sample(arr, conf_alpha)

    rng = np.random.default_rng(seed)
    es_boot = np.empty(int(n_boot), dtype=np.float64)
    n_blocks = int(np.ceil(n / block_size))
    for b in range(int(n_boot)):
        starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        idx_pieces = [np.arange(s, s + block_size) for s in starts]
        idx = np.concatenate(idx_pieces)[:n]
        sample = arr[idx]
        _, es_b = _es_from_sample(sample, conf_alpha)
        es_boot[b] = es_b

    lo_q = (1.0 - ci_alpha) / 2.0
    hi_q = 1.0 - lo_q
    ci_low = float(np.quantile(es_boot, lo_q))
    ci_high = float(np.quantile(es_boot, hi_q))
    return EsBootstrapResult(
        es_point=float(es_point),
        es_ci_low=ci_low,
        es_ci_high=ci_high,
        var_point=float(var_point),
        ci_alpha=float(ci_alpha),
        conf_alpha=float(conf_alpha),
        n_boot=int(n_boot),
        block_size=int(block_size),
        n=int(n),
    )


__all__ = ["EsBootstrapResult", "esbootstrap"]
