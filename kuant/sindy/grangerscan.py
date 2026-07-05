"""Bonferroni-corrected Granger-causality scan over many candidates.

Motivation. Given a target time series and a library of N candidate
predictors × H horizons, test which candidates Granger-cause the
target. Bonferroni correction (α / (N·H)) controls the family-wise
error rate at the standard 0.05.

Classic use: scan a few dozen macro factors across a handful of
horizons for candidates that might inform a strategy's leverage or
gating decisions. Typical outcome — a handful of candidates pass
Bonferroni; most turn out to be variants of the same underlying
macro variable, leaving one or two genuinely orthogonal findings
worth chasing.

Cheap to run (statsmodels' grangercausalitytests is compiled),
useful even when the ultimate gate ends up being borderline.

Design: docs/tools/grangerscan.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from kuant._validation import require_dep
from kuant.errors import KuantValueError


@dataclass
class GrangerHit:
    """A single (candidate, horizon) pass at the Bonferroni threshold."""

    candidate: str
    horizon: int
    f_stat: float
    p_value: float


@dataclass
class GrangerScanResult:
    hits: list[GrangerHit] = field(default_factory=list)
    n_tests: int = 0
    bonferroni_alpha: float = 0.0

    def summary(self) -> str:
        lines = [
            "=== Granger causality scan ===",
            f"Tests run:                {self.n_tests}",
            f"Bonferroni α threshold:   {self.bonferroni_alpha:.6f}",
            f"Candidates passing:       {len(self.hits)}",
        ]
        if self.hits:
            lines.append("")
            lines.append(f'{"Candidate":<30s} {"Horizon":>7s} {"F":>8s} {"p":>12s}')
            for h in sorted(self.hits, key=lambda x: x.p_value):
                lines.append(
                    f"{h.candidate:<30s} {h.horizon:>7d} {h.f_stat:>8.3f} {h.p_value:>12.2e}"
                )
        return "\n".join(lines)


def _require_statsmodels():
    try:
        from statsmodels.tsa.stattools import grangercausalitytests

        return grangercausalitytests
    except ImportError as e:
        require_dep(
            "statsmodels",
            kernel="grangerscan",
            install="pip install statsmodels",
            cause=e,
        )


def grangerscan(
    target: np.ndarray,
    candidates: dict[str, np.ndarray],
    horizons: Optional[list[int]] = None,
    alpha: float = 0.05,
    verbose: bool = False,
) -> GrangerScanResult:
    """Bonferroni-corrected Granger F-test scan.

    For each (candidate, horizon) pair, fit a bivariate VAR and F-test
    whether adding the candidate's `horizon`-lagged values reduces the
    target's residual variance significantly.

    Parameters
    ----------
    target : 1D np.ndarray
        Target series (e.g. forward returns of a strategy).
    candidates : dict[str, 1D np.ndarray]
        Named candidate predictors of the same length as `target`.
    horizons : list[int], optional
        Lag horizons to test. Default [1, 2, 5].
    alpha : float, default 0.05
        Family-wise error rate. Bonferroni threshold =
        alpha / (n_candidates * n_horizons).
    verbose : bool, default False
        If True, print each test result to stdout.

    Returns
    -------
    GrangerScanResult with the list of hits.

    Notes
    -----
    Uses `statsmodels.tsa.stattools.grangercausalitytests`. Lazy import.

    Interpretation warning: many hits will be redundant with each other
    (e.g. VIX_level, VIX_change, VIX_pct all pass together). Post-filter
    for orthogonality against your existing signal set before shipping.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> x = rng.normal(size=500)
    >>> y = np.roll(x, 1) + rng.normal(scale=0.5, size=500)  # x Granger-causes y at lag 1
    >>> result = grangerscan(y, {'x': x, 'noise': rng.normal(size=500)})
    >>> 'x' in [h.candidate for h in result.hits]
    True
    """
    grangercausalitytests = _require_statsmodels()

    if horizons is None:
        horizons = [1, 2, 5]

    for h in horizons:
        if not isinstance(h, (int, np.integer)) or int(h) <= 0:
            raise KuantValueError(
                f"kuant.grangerscan: 'horizons' must be strictly positive "
                f"ints; got {h}.  [KE-VAL-POSITIVE]\n"
                f"  → Fix: pass positive horizon lags"
            )
    n_tests = len(candidates) * len(horizons)
    bonferroni_alpha = alpha / n_tests
    result = GrangerScanResult(n_tests=n_tests, bonferroni_alpha=bonferroni_alpha)

    target_arr = np.asarray(target, dtype=np.float64)

    for name, cand in candidates.items():
        cand_arr = np.asarray(cand, dtype=np.float64)
        # statsmodels wants [target, cause] 2D
        data = np.column_stack([target_arr, cand_arr])
        # Drop rows with NaN
        mask = np.isfinite(data).all(axis=1)
        data_clean = data[mask]
        if len(data_clean) < 30:
            if verbose:
                print(f"{name}: too few clean observations ({len(data_clean)}), skipping")
            continue

        for h in horizons:
            try:
                # `verbose=False` is deprecated in statsmodels >=0.14; the
                # function no longer prints internally, so we omit the kwarg.
                # Silence any remaining warnings from statsmodels internals.
                import warnings

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=FutureWarning)
                    res = grangercausalitytests(data_clean, maxlag=[h])
                f_stat = res[h][0]["ssr_ftest"][0]
                p_value = res[h][0]["ssr_ftest"][1]
                if verbose:
                    print(f"{name:<30s} h={h} F={f_stat:.3f} p={p_value:.2e}")
                if p_value < bonferroni_alpha:
                    result.hits.append(
                        GrangerHit(
                            candidate=name,
                            horizon=h,
                            f_stat=f_stat,
                            p_value=p_value,
                        )
                    )
            except Exception as exc:
                if verbose:
                    print(f"{name} h={h} failed: {exc}")
                continue

    return result
