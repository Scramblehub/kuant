"""Hansen's Superior Predictive Ability test and Model Confidence Set.

Both tests answer variants of:

    "Given a benchmark strategy AND a set of alternative strategies,
    is any alternative genuinely better than the benchmark once we
    correct for the number of alternatives tried?"

- **`spa_test`** (Hansen 2005): tests the null "no alternative is
  better than the benchmark." A small p rejects → at least one
  alternative is genuinely superior.
- **`mcs_test`** (Hansen, Lunde & Nason 2011): iteratively eliminates
  strategies that are provably worse until only a "confidence set"
  of statistically indistinguishable strategies remains.

Both use block-bootstrap to handle serial correlation.

Design: docs/kernels/nulltest/spa_test.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from kuant._validation import require_positive, require_probability
from kuant.errors import KuantShapeError, KuantValueError


@dataclass
class SPAResult:
    """Hansen SPA test result.

    Attributes
    ----------
    p_value : float
        Bootstrap p-value under the null "no alternative is better".
    max_t_stat : float
        The observed max standardized loss-difference across alternatives.
    n_alternatives : int
    n_boot : int
    survivors : list[int]
        For MCS: indices of strategies in the confidence set. Empty
        for SPA (which just tests the null).
    """

    p_value: float
    max_t_stat: float
    n_alternatives: int
    n_boot: int
    survivors: list = field(default_factory=list)

    def summary(self) -> str:
        return (
            "=== SPAResult ===\n"
            f"p-value:         {self.p_value:.4f}\n"
            f"max t-stat:      {self.max_t_stat:+.4f}\n"
            f"n_alternatives:  {self.n_alternatives}\n"
            f"n_boot:          {self.n_boot}\n"
            f"survivors:       {self.survivors if self.survivors else '(SPA-only)'}"
        )


def spa_test(
    benchmark_returns,
    alternative_returns,
    n_boot: int = 1000,
    mean_block_length: float = 5.0,
    seed: int = 0,
) -> SPAResult:
    """Hansen Superior Predictive Ability test.

    Parameters
    ----------
    benchmark_returns : 1D array of length T
    alternative_returns : 2D array of shape (T, K)
        Return series for K candidate strategies aligned to the benchmark.
    n_boot : int
    mean_block_length : float
    seed : int

    Returns
    -------
    SPAResult

    Notes
    -----
    Small p (< 0.05) rejects "no alternative beats the benchmark" →
    at least one alternative is genuinely superior.
    """
    bench = np.asarray(benchmark_returns, dtype=np.float64)
    alts = np.asarray(alternative_returns, dtype=np.float64)
    if bench.ndim != 1:
        raise KuantShapeError(
            f"kuant.spa_test: 'benchmark_returns' must be 1D, got shape "
            f"{bench.shape}.  [KE-SHAPE-EXPECTED]\n"
            f"  → Fix: pass a 1D benchmark series"
        )
    if alts.ndim != 2:
        raise KuantShapeError(
            f"kuant.spa_test: 'alternative_returns' must be 2D (T, K), "
            f"got shape {alts.shape}.  [KE-SHAPE-EXPECTED]\n"
            f"  → Fix: pass a (T, K) matrix of alternative return series"
        )
    T, K = alts.shape
    if bench.size != T:
        raise KuantShapeError(
            f"kuant.spa_test: 'benchmark_returns' length {bench.size} "
            f"does not match T={T} in 'alternative_returns'.  "
            f"[KE-SHAPE-EQUAL-LEN]\n"
            f"  → Fix: align both to the same time index"
        )
    require_positive(n_boot, "n_boot", kernel="spa_test", kind="int")
    require_positive(mean_block_length, "mean_block_length", kernel="spa_test")

    # Loss differential: benchmark loss minus alternative loss.
    # Here loss = -return (we want positive when the alternative outperforms).
    d = alts - bench[:, None]  # shape (T, K)
    # Standardize per-alternative.
    mu = d.mean(axis=0)
    sd = d.std(axis=0, ddof=1)
    sd = np.where(sd > 1e-15, sd, 1e-15)
    t_stat = np.sqrt(T) * mu / sd
    observed_max = float(t_stat.max())

    rng = np.random.default_rng(seed)
    counts = 0
    p = 1.0 / float(mean_block_length)
    for b in range(int(n_boot)):
        idx = np.empty(T, dtype=np.int64)
        i = int(rng.integers(0, T))
        for t in range(T):
            idx[t] = i
            if rng.random() < p:
                i = int(rng.integers(0, T))
            else:
                i = (i + 1) % T
        d_boot = d[idx]
        # Recentered under the null: subtract observed mean.
        mu_b = d_boot.mean(axis=0) - mu
        sd_b = d_boot.std(axis=0, ddof=1)
        sd_b = np.where(sd_b > 1e-15, sd_b, 1e-15)
        t_b = np.sqrt(T) * mu_b / sd_b
        if t_b.max() >= observed_max:
            counts += 1

    return SPAResult(
        p_value=counts / n_boot,
        max_t_stat=observed_max,
        n_alternatives=K,
        n_boot=int(n_boot),
        survivors=[],
    )


def mcs_test(
    strategy_returns,
    alpha: float = 0.05,
    n_boot: int = 1000,
    mean_block_length: float = 5.0,
    seed: int = 0,
) -> SPAResult:
    """Model Confidence Set: iteratively drop provably-worse strategies.

    Parameters
    ----------
    strategy_returns : 2D array of shape (T, K)
        Each column is a strategy's return series.
    alpha : float, default 0.05
        Confidence level. `1 - alpha` is the coverage probability.

    Returns
    -------
    SPAResult
        `.survivors` holds the indices of strategies in the confidence
        set at level `alpha`.

    Notes
    -----
    Iterative procedure:
      1. Test "all strategies are equally good."
      2. If rejected, drop the worst-performing one.
      3. Repeat until we cannot reject the joint null.

    The remaining set contains all strategies that are statistically
    indistinguishable from the best; the actual best is in there with
    probability `1 - alpha`.
    """
    R = np.asarray(strategy_returns, dtype=np.float64)
    if R.ndim != 2:
        raise KuantShapeError(
            f"kuant.mcs_test: 'strategy_returns' must be 2D (T, K), got "
            f"shape {R.shape}.  [KE-SHAPE-EXPECTED]\n"
            f"  → Fix: pass a (T, K) matrix"
        )
    T, K = R.shape
    if K < 2:
        raise KuantValueError(
            f"kuant.mcs_test: need at least 2 strategies, got {K}.  "
            f"[KE-VAL-RANGE]\n"
            f"  → Fix: pass a 2D array with at least 2 columns"
        )
    require_probability(alpha, "alpha", kernel="mcs_test")
    require_positive(n_boot, "n_boot", kernel="mcs_test", kind="int")
    require_positive(mean_block_length, "mean_block_length", kernel="mcs_test")

    survivors = list(range(K))
    last_pval = 1.0
    last_max = 0.0

    while len(survivors) > 1:
        sub = R[:, survivors]
        # Iteration test: is the WORST-in-sample strategy significantly
        # worse than the BEST? If yes, drop the worst. Equivalent MCS
        # semantics to Hansen-Lunde-Nason range-statistic test at the
        # single-drop level, with block-bootstrap p-value.
        means = sub.mean(axis=0)
        best_local = int(np.argmax(means))
        worst_local = int(np.argmin(means))
        if best_local == worst_local:
            break  # only one column left after previous drops

        # SPA compares alternatives (worst) against a benchmark (best),
        # asking "is worst better than best?" — we want the OPPOSITE:
        # "is worst SIGNIFICANTLY worse than best?" So flip and test
        # whether best beats worst.
        best_series = sub[:, best_local]
        worst_series = sub[:, worst_local]
        r = spa_test(
            worst_series,  # treat worst as "benchmark"
            best_series.reshape(-1, 1),  # test whether best beats it
            n_boot=n_boot,
            mean_block_length=mean_block_length,
            seed=seed,
        )
        last_pval = r.p_value
        last_max = r.max_t_stat

        if r.p_value > alpha:
            # Cannot reject "worst is not worse than best" → stop.
            break

        # Drop the worst.
        drop = survivors[worst_local]
        survivors.remove(drop)

    return SPAResult(
        p_value=last_pval,
        max_t_stat=last_max,
        n_alternatives=K,
        n_boot=int(n_boot),
        survivors=sorted(survivors),
    )


__all__ = ["SPAResult", "spa_test", "mcs_test"]
