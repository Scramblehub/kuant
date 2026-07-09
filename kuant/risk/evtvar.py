"""Extreme value theory (POT) VaR and Expected Shortfall.

Peaks-over-Threshold (POT) fits a Generalized Pareto Distribution
(GPD) to the empirical excesses above a threshold `u`. Under the
Pickands-Balkema-de Haan theorem, tail excesses of most distributions
converge to a GPD as the threshold rises. The GPD parameters (xi,
sigma) are estimated via method-of-moments (fast, robust).

Given fitted (xi, sigma) and threshold u with N_u excesses out of n
total observations, VaR at confidence alpha:

    VaR_alpha = u + (sigma / xi) * (((n / N_u) * (1 - alpha))^(-xi) - 1)

and ES:

    ES_alpha = VaR_alpha / (1 - xi) + (sigma - xi * u) / (1 - xi)

for xi < 1 (otherwise ES is infinite in expectation).

Sign convention: VaR and ES both reported as POSITIVE loss magnitudes.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from kuant._validation import require_1d, require_range
from kuant.errors import KuantNumericWarning, KuantValueError


@dataclass
class EvtVarResult:
    var: float
    es: float
    threshold: float
    xi: float
    sigma: float
    n_exceedances: int
    n_total: int
    alpha: float

    def summary(self) -> str:
        return (
            "=== EvtVarResult ===\n"
            f"VaR (loss):     {self.var:+.6f}\n"
            f"ES (loss):      {self.es:+.6f}\n"
            f"threshold u:    {self.threshold:+.6f}\n"
            f"xi (shape):     {self.xi:+.4f}\n"
            f"sigma (scale):  {self.sigma:+.6f}\n"
            f"exceedances:    {self.n_exceedances} / {self.n_total} "
            f"({100 * self.n_exceedances / self.n_total:.1f}%)\n"
            f"alpha:          {self.alpha}"
        )


def evtvar(
    returns,
    *,
    alpha: float = 0.99,
    threshold_pct: float = 0.90,
) -> EvtVarResult:
    """Peaks-over-Threshold GPD VaR and ES.

    Parameters
    ----------
    returns : 1D array
        Return series. LOSSES are treated as positive: internally the
        function fits the tail of `-returns` above a threshold, since
        that is where losses live.
    alpha : float, default 0.99
        Confidence level.
    threshold_pct : float, default 0.90
        Empirical quantile of |returns| to use as the GPD threshold.
        0.90 to 0.95 are standard practice.

    Returns
    -------
    EvtVarResult

    References
    ----------
    Balkema & de Haan 1974; Pickands 1975; McNeil-Frey-Embrechts 2015
    ("Quantitative Risk Management") for the practical estimator.
    """
    arr = np.asarray(returns, dtype=np.float64)
    require_1d(arr, "returns", kernel="evtvar")
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n < 250:
        raise KuantValueError(
            f"kuant.evtvar: only {n} finite values; need at least 250 "
            f"for a stable GPD tail fit.  [KE-VAL-MIN-CLEAN]"
        )
    require_range(alpha, "alpha", kernel="evtvar", lo=0.5, hi=0.9999)
    require_range(threshold_pct, "threshold_pct", kernel="evtvar", lo=0.5, hi=0.99)

    losses = -arr
    u = float(np.quantile(losses, threshold_pct))
    excesses = losses[losses > u] - u
    n_exc = excesses.size
    if n_exc < 20:
        raise KuantValueError(
            f"kuant.evtvar: only {n_exc} excesses above threshold; need "
            f"at least 20. Lower threshold_pct or provide more data.  "
            f"[KE-VAL-MIN-CLEAN]"
        )

    # Method-of-moments GPD fit (Hosking-Wallis 1987).
    mean_exc = float(excesses.mean())
    var_exc = float(excesses.var(ddof=1))
    if var_exc < 1e-15:
        return EvtVarResult(
            var=u,
            es=u,
            threshold=u,
            xi=0.0,
            sigma=0.0,
            n_exceedances=int(n_exc),
            n_total=int(n),
            alpha=float(alpha),
        )
    xi = 0.5 * (1.0 - mean_exc**2 / var_exc)
    sigma = 0.5 * mean_exc * (1.0 + mean_exc**2 / var_exc)
    if sigma < 1e-15:
        sigma = 1e-15
    # MOM is only consistent for xi < 0.5 (finite variance of excesses).
    # Beyond that, MOM biases xi downward; users should switch to PWM or MLE.
    if xi >= 0.4:
        warnings.warn(
            f"kuant.evtvar: MOM xi estimate ({xi:.3f}) is near or above "
            f"the 0.5 validity boundary. MOM assumes finite variance of "
            f"excesses and biases downward for heavier tails; prefer "
            f"probability-weighted moments (PWM) or MLE for a more "
            f"reliable estimate.  [KW-EVT-MOM-INVALID]",
            KuantNumericWarning,
            stacklevel=2,
        )

    # VaR / ES formulas from McNeil-Frey.
    prob_tail = 1.0 - alpha
    ratio = (n / n_exc) * prob_tail
    if abs(xi) < 1e-10:
        var = u + sigma * (-np.log(ratio))
    else:
        var = u + (sigma / xi) * (ratio ** (-xi) - 1)
    if xi < 1.0:
        es = (var + sigma - xi * u) / (1.0 - xi)
    else:
        es = float("inf")

    return EvtVarResult(
        var=float(var),
        es=float(es),
        threshold=float(u),
        xi=float(xi),
        sigma=float(sigma),
        n_exceedances=int(n_exc),
        n_total=int(n),
        alpha=float(alpha),
    )


__all__ = ["EvtVarResult", "evtvar"]
