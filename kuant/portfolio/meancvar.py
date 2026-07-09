"""Mean-CVaR portfolio optimization (Rockafellar-Uryasev 2000).

Solves:
    max_w  w' * mu
    s.t.   CVaR_alpha(w' * R) <= gamma
           sum(w) = 1, w >= 0

Rockafellar-Uryasev showed CVaR is linearly programmable via an
auxiliary variable eta:

    CVaR_alpha = eta + (1 / ((1 - alpha) * T)) * sum(max(0, -w'r_t - eta))

so the whole problem becomes an LP over (w, eta, z_t).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kuant._validation import require_range
from kuant.errors import KuantShapeError, KuantValueError


@dataclass
class MeanCvarResult:
    weights: np.ndarray
    expected_return: float
    cvar: float
    var: float
    alpha: float
    n_scenarios: int
    status: str

    def summary(self) -> str:
        return (
            "=== MeanCvarResult ===\n"
            f"weights:            {self.weights}\n"
            f"expected return:    {self.expected_return:+.6f}\n"
            f"VaR (alpha):        {self.var:.6f}\n"
            f"CVaR:               {self.cvar:.6f}\n"
            f"alpha:              {self.alpha}\n"
            f"scenarios:          {self.n_scenarios}\n"
            f"solver status:      {self.status}"
        )


def meancvar(
    returns,
    *,
    alpha: float = 0.95,
    cvar_limit: float | None = None,
    long_only: bool = True,
) -> MeanCvarResult:
    """Mean-CVaR portfolio optimization.

    Parameters
    ----------
    returns : 2D array, shape (T, n)
        Historical scenarios. T time steps, n assets.
    alpha : float, default 0.95
        CVaR confidence level.
    cvar_limit : float, optional
        Upper bound on CVaR. If None, minimizes CVaR subject to sum-to-1
        (no return objective; equivalent to minimum-CVaR portfolio).
    long_only : bool, default True
        If True, weights are constrained non-negative.

    Returns
    -------
    MeanCvarResult

    References
    ----------
    Rockafellar & Uryasev 2000, "Optimization of conditional value-
    at-risk."
    """
    R = np.asarray(returns, dtype=np.float64)
    if R.ndim != 2:
        raise KuantShapeError(
            f"kuant.meancvar: 'returns' must be 2D (T, n); got shape " f"{R.shape}.  [KE-SHAPE-2D]"
        )
    T, n = R.shape
    if T < 20:
        raise KuantValueError(
            f"kuant.meancvar: only {T} scenarios; need at least 20.  " f"[KE-VAL-MIN-CLEAN]"
        )
    require_range(alpha, "alpha", kernel="meancvar", lo=0.5, hi=0.999)

    mu = R.mean(axis=0)

    try:
        from scipy.optimize import linprog
    except ImportError as e:
        raise KuantValueError(
            "kuant.meancvar: requires scipy.optimize.  [KE-DEP-MISSING]\n"
            "  -> Fix: pip install scipy"
        ) from e

    # Variables: w (n), eta (1), z (T). Total: n + 1 + T.
    n_vars = n + 1 + T

    if cvar_limit is None:
        # Minimize CVaR: c = [0]*n + [1] + [1/((1-alpha)*T)] * T
        c = np.zeros(n_vars)
        c[n] = 1.0
        c[n + 1 :] = 1.0 / ((1 - alpha) * T)
    else:
        # Maximize expected return: minimize -mu' w
        c = np.zeros(n_vars)
        c[:n] = -mu

    # Inequality: z_t >= -R_t' w - eta  ->  -R_t' w - eta - z_t <= 0
    A_ub_lines = []
    b_ub_lines = []
    for t in range(T):
        row = np.zeros(n_vars)
        row[:n] = -R[t]
        row[n] = -1.0
        row[n + 1 + t] = -1.0
        A_ub_lines.append(row)
        b_ub_lines.append(0.0)

    if cvar_limit is not None:
        # CVaR constraint: eta + (1/((1-alpha)T)) sum(z_t) <= cvar_limit
        row = np.zeros(n_vars)
        row[n] = 1.0
        row[n + 1 :] = 1.0 / ((1 - alpha) * T)
        A_ub_lines.append(row)
        b_ub_lines.append(float(cvar_limit))

    A_ub = np.array(A_ub_lines)
    b_ub = np.array(b_ub_lines)

    # Equality: sum(w) = 1.
    A_eq = np.zeros((1, n_vars))
    A_eq[0, :n] = 1.0
    b_eq = np.array([1.0])

    # Bounds: w in [0, inf) if long_only else (-inf, inf); eta free; z >= 0.
    bounds = []
    if long_only:
        bounds.extend([(0, None)] * n)
    else:
        bounds.extend([(None, None)] * n)
    bounds.append((None, None))  # eta
    bounds.extend([(0, None)] * T)  # z

    res = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if not res.success:
        raise KuantValueError(
            f"kuant.meancvar: LP failed with status: {res.message}.  " f"[KE-LP-FAILED]"
        )

    w = res.x[:n]
    eta = float(res.x[n])
    z = res.x[n + 1 :]
    cvar_val = eta + (1.0 / ((1 - alpha) * T)) * float(z.sum())
    port_return = float(mu @ w)
    return MeanCvarResult(
        weights=w,
        expected_return=port_return,
        cvar=cvar_val,
        var=eta,
        alpha=float(alpha),
        n_scenarios=int(T),
        status=res.message if res.message else "OK",
    )


__all__ = ["MeanCvarResult", "meancvar"]
