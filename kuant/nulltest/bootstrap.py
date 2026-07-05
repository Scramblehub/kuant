"""Block-bootstrap tools for serially-correlated financial data.

Standard i.i.d. bootstrap breaks on time series because return
sequences have serial correlation. Two block resampling schemes
preserve short-range dependence:

- **Moving Block Bootstrap** (Kunsch 1989): fixed-length blocks
  drawn with replacement.
- **Stationary Bootstrap** (Politis & Romano 1994): block lengths
  drawn from a geometric distribution; the resampled series is
  itself strictly stationary.

Design: docs/kernels/nulltest/bootstrap.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kuant._validation import (
    require_1d,
    require_equal_length,
    require_positive,
    warn_kuant,
)
from kuant.errors import KuantNumericWarning, KuantValueError


def stationary_bootstrap(series, mean_block_length: float, seed: int = 0) -> np.ndarray:
    """One stationary-block bootstrap draw from a 1D series.

    Parameters
    ----------
    series : 1D array
    mean_block_length : float
        Expected length of each block. Set larger for series with
        stronger serial correlation. Rule of thumb: `T^(1/3)` for
        weakly-dependent series.
    seed : int
        RNG seed.

    Returns
    -------
    1D np.ndarray of the same length as `series`.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> x = rng.standard_normal(200)
    >>> resample = stationary_bootstrap(x, mean_block_length=5, seed=42)
    >>> resample.shape == x.shape
    True
    """
    arr = np.asarray(series, dtype=np.float64)
    require_1d(arr, "series", kernel="stationary_bootstrap")
    require_positive(mean_block_length, "mean_block_length", kernel="stationary_bootstrap")
    n = arr.size
    if n < 2:
        raise KuantValueError(
            "kuant.stationary_bootstrap: need at least 2 elements.  "
            "[KE-VAL-RANGE]\n"
            "  → Fix: provide more data"
        )
    if float(mean_block_length) >= n:
        warn_kuant(
            kernel="stationary_bootstrap",
            code="KW-BOOT-BLOCK-TOO-LONG",
            what=(
                f"mean_block_length ({mean_block_length:g}) is not smaller "
                f"than n ({n}); the resample degenerates to near-perfect "
                f"copies of the input starting at a random offset"
            ),
            fix=(
                "set mean_block_length between 1 and about n^(1/3) for " "weakly-dependent series"
            ),
            category=KuantNumericWarning,
        )
    p = 1.0 / float(mean_block_length)
    if not 0.0 < p <= 1.0:
        raise KuantValueError(
            f"kuant.stationary_bootstrap: 'mean_block_length' produces "
            f"invalid probability {p}.  [KE-VAL-RANGE]\n"
            f"  → Fix: mean_block_length must be >= 1"
        )
    rng = np.random.default_rng(seed)
    out = np.empty(n, dtype=arr.dtype)
    # Start at a random position; each subsequent bar either extends the
    # current block (probability 1-p) or jumps to a new random start.
    i = rng.integers(0, n)
    for t in range(n):
        out[t] = arr[i]
        if rng.random() < p:
            i = int(rng.integers(0, n))
        else:
            i = (i + 1) % n
    return out


@dataclass
class BootstrapICResult:
    """Distribution of a signal's IC under block-bootstrap resampling.

    Attributes
    ----------
    point_estimate : float
        IC on the un-resampled data.
    bootstrap_distribution : 1D np.ndarray, length n_boot
        IC computed on each bootstrap draw.
    p_value : float
        Fraction of bootstrap draws >= `point_estimate` (for a
        one-sided upper test) OR <= it (lower). We report the two-sided
        p-value: `min(fraction_above, fraction_below) * 2`.
    ci_low : float
    ci_high : float
        95% percentile confidence interval on the IC.
    n_boot : int
    mean_block_length : float
    """

    point_estimate: float
    bootstrap_distribution: np.ndarray
    p_value: float
    ci_low: float
    ci_high: float
    n_boot: int
    mean_block_length: float

    def summary(self) -> str:
        return (
            "=== BootstrapICResult ===\n"
            f"point IC:            {self.point_estimate:+.4f}\n"
            f"95% CI:              [{self.ci_low:+.4f}, {self.ci_high:+.4f}]\n"
            f"two-sided p-value:   {self.p_value:.4f}\n"
            f"n_boot:              {self.n_boot}\n"
            f"mean_block_length:   {self.mean_block_length:g}"
        )


def bootstrap_ic(
    signal,
    forward_returns,
    n_boot: int = 1000,
    mean_block_length: float = 5.0,
    seed: int = 0,
) -> BootstrapICResult:
    """Block-bootstrap confidence interval + p-value for a signal's IC.

    Uses stationary-block resampling on `(signal, forward_returns)`
    JOINTLY (same block indices applied to both) so the correlation
    structure survives.

    Parameters
    ----------
    signal : 1D array
    forward_returns : 1D array
        Same length as `signal`.
    n_boot : int, default 1000
    mean_block_length : float, default 5.0
    seed : int, default 0

    Returns
    -------
    BootstrapICResult

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> sig = rng.standard_normal(500)
    >>> ret = 0.1 * sig + 0.3 * rng.standard_normal(500)   # real IC ~0.3
    >>> res = bootstrap_ic(sig, ret, n_boot=300)
    >>> res.p_value < 0.05                                  # detects signal
    True
    """
    sig = np.asarray(signal, dtype=np.float64)
    ret = np.asarray(forward_returns, dtype=np.float64)
    require_1d(sig, "signal", kernel="bootstrap_ic")
    require_1d(ret, "forward_returns", kernel="bootstrap_ic")
    require_equal_length(sig, "signal", ret, "forward_returns", kernel="bootstrap_ic")
    require_positive(n_boot, "n_boot", kernel="bootstrap_ic", kind="int")
    require_positive(mean_block_length, "mean_block_length", kernel="bootstrap_ic")
    if int(n_boot) < 100:
        warn_kuant(
            kernel="bootstrap_ic",
            code="KW-BOOT-LOW-N-BOOT",
            what=(
                f"n_boot={int(n_boot)} is below 100; the resulting p-value "
                f"has resolution worse than 1% and percentile CIs are "
                f"unstable"
            ),
            fix=(
                "use at least 1000 draws for a reasonable p-value; 100 is "
                "the bare minimum, and only for quick diagnostics"
            ),
            category=KuantNumericWarning,
        )

    mask = np.isfinite(sig) & np.isfinite(ret)
    sig_c = sig[mask]
    ret_c = ret[mask]
    if sig_c.size < 10:
        raise KuantValueError(
            f"kuant.bootstrap_ic: only {sig_c.size} clean rows.  "
            f"[KE-VAL-MIN-CLEAN]\n"
            f"  → Fix: provide more data or fewer NaNs"
        )

    def _corr(x, y):
        # Fast Pearson (not Spearman) to keep the bootstrap loop cheap;
        # for Spearman use kuant.signals.factor_ic then bootstrap the
        # per-period series.
        xc = x - x.mean()
        yc = y - y.mean()
        num = float((xc * yc).sum())
        den = float(np.sqrt((xc * xc).sum() * (yc * yc).sum()))
        return num / den if den > 0 else 0.0

    point = _corr(sig_c, ret_c)

    p = 1.0 / float(mean_block_length)
    rng = np.random.default_rng(seed)
    n = sig_c.size
    boot = np.empty(int(n_boot))
    for b in range(int(n_boot)):
        idx = np.empty(n, dtype=np.int64)
        i = int(rng.integers(0, n))
        for t in range(n):
            idx[t] = i
            if rng.random() < p:
                i = int(rng.integers(0, n))
            else:
                i = (i + 1) % n
        boot[b] = _corr(sig_c[idx], ret_c[idx])

    # p-value under the null IC = 0: what fraction of bootstrap draws
    # have the OPPOSITE sign to the observed point estimate (i.e. would
    # be as-or-more-extreme in the wrong direction if the true IC is zero).
    # Multiply by 2 for a two-sided test; cap at 1.
    if point > 0:
        pval = min(2.0 * float((boot <= 0).sum() / n_boot), 1.0)
    elif point < 0:
        pval = min(2.0 * float((boot >= 0).sum() / n_boot), 1.0)
    else:
        pval = 1.0
    ci_low = float(np.percentile(boot, 2.5))
    ci_high = float(np.percentile(boot, 97.5))

    return BootstrapICResult(
        point_estimate=float(point),
        bootstrap_distribution=boot,
        p_value=pval,
        ci_low=ci_low,
        ci_high=ci_high,
        n_boot=int(n_boot),
        mean_block_length=float(mean_block_length),
    )


__all__ = ["stationary_bootstrap", "bootstrap_ic", "BootstrapICResult"]
