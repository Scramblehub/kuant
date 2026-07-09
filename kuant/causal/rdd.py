"""Sharp Regression Discontinuity Design.

Estimates the causal jump in Y at a threshold `cutoff` in the running
variable `x`. Treatment is deterministic in x (sharp RDD): D = 1[x >=
cutoff]. Under smoothness of E[Y | X] on either side of `cutoff`, the
discontinuity IS the local average treatment effect.

Standard estimator: local linear regression on either side of the
cutoff within a bandwidth `h`. Uses a triangular kernel (Imbens-Kalyanaraman
recommendation) that gives full weight at the cutoff and zero weight at
+/-h. Default bandwidth follows a simple rule of thumb; for production
use pass an Imbens-Kalyanaraman or Calonico-Cattaneo-Titiunik bandwidth.

Sign convention: `tau` = E[Y | x -> cutoff+] - E[Y | x -> cutoff-]
(post-jump minus pre-jump). Positive tau means outcome JUMPS UP at the
cutoff.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kuant._validation import require_1d, require_positive
from kuant.errors import KuantValueError


@dataclass
class RddResult:
    tau: float
    tau_se: float
    tau_t_stat: float
    n_left: int
    n_right: int
    intercept_left: float
    intercept_right: float
    slope_left: float
    slope_right: float
    bandwidth: float
    cutoff: float

    def summary(self) -> str:
        return (
            "=== RddResult ===\n"
            f"tau (jump):           {self.tau:+.6f}\n"
            f"tau SE:               {self.tau_se:.6f}\n"
            f"tau t-stat:           {self.tau_t_stat:+.2f}\n"
            f"cutoff / bandwidth:   {self.cutoff:.4f} / {self.bandwidth:.4f}\n"
            f"n (left / right):     {self.n_left} / {self.n_right}"
        )


def _triangular_kernel(u):
    return np.maximum(1.0 - np.abs(u), 0.0)


def _wls(x, y, w):
    """Weighted least squares for y = a + b*x with weights w.

    Degrees of freedom use the unweighted-count convention `n - 2`
    (Calonico-Cattaneo-Titiunik 2014 / statsmodels WLS), not the
    kernel-weighted effective sample size, so tau_se matches published
    RDD SE conventions.
    """
    W = np.diag(w)
    X = np.column_stack([np.ones_like(x), x])
    XtWX = X.T @ W @ X
    XtWy = X.T @ W @ y
    beta = np.linalg.solve(XtWX, XtWy)
    resid = y - X @ beta
    dof = max(x.size - 2, 1)
    sigma2 = float(np.sum(w * resid**2) / dof)
    cov = sigma2 * np.linalg.pinv(XtWX)
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    return beta[0], beta[1], se[0], se[1]


def rdd(x, y, cutoff: float, *, bandwidth: float = None) -> RddResult:
    """Sharp regression discontinuity via local linear regression.

    Parameters
    ----------
    x : 1D array
        Running variable.
    y : 1D array
        Outcome.
    cutoff : float
        Threshold at which treatment starts (D = 1[x >= cutoff]).
    bandwidth : float, optional
        Half-width of the local window around `cutoff`. If None, uses
        a simple rule-of-thumb bandwidth 1.5 * std(x) * n^{-0.2}.

    Returns
    -------
    RddResult

    References
    ----------
    Imbens & Lemieux 2008; Imbens & Kalyanaraman 2012 for optimal
    bandwidth; Calonico, Cattaneo, Titiunik 2014 for the modern
    bias-corrected variant.
    """
    x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    require_1d(x_arr, "x", kernel="rdd")
    require_1d(y_arr, "y", kernel="rdd")
    if x_arr.size != y_arr.size:
        raise KuantValueError(
            f"kuant.rdd: 'x' and 'y' must be equal length; got "
            f"{x_arr.size} and {y_arr.size}.  [KE-SHAPE-EQUAL-LEN]"
        )
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    xc = x_arr[mask]
    yc = y_arr[mask]
    n = xc.size
    if n < 40:
        raise KuantValueError(
            f"kuant.rdd: only {n} clean rows; need at least 40.  " f"[KE-VAL-MIN-CLEAN]"
        )

    if bandwidth is None:
        bandwidth = float(1.5 * np.std(xc) * n ** (-0.2))
    require_positive(bandwidth, "bandwidth", kernel="rdd", kind="value")

    xc_c = xc - cutoff
    # Left / right subsamples inside bandwidth.
    left = (xc_c >= -bandwidth) & (xc_c < 0)
    right = (xc_c >= 0) & (xc_c <= bandwidth)
    n_l = int(left.sum())
    n_r = int(right.sum())
    if n_l < 5 or n_r < 5:
        raise KuantValueError(
            f"kuant.rdd: fewer than 5 observations on one side of the "
            f"cutoff within the bandwidth (left={n_l}, right={n_r}). "
            f"Widen bandwidth.  [KE-VAL-MIN-CLEAN]"
        )

    w_l = _triangular_kernel(xc_c[left] / bandwidth)
    w_r = _triangular_kernel(xc_c[right] / bandwidth)
    a_l, b_l, se_a_l, _ = _wls(xc_c[left], yc[left], w_l)
    a_r, b_r, se_a_r, _ = _wls(xc_c[right], yc[right], w_r)

    tau = a_r - a_l
    tau_se = float(np.sqrt(se_a_l**2 + se_a_r**2))
    tau_t = float(tau / tau_se) if tau_se > 0 else float("inf")

    return RddResult(
        tau=float(tau),
        tau_se=tau_se,
        tau_t_stat=tau_t,
        n_left=n_l,
        n_right=n_r,
        intercept_left=float(a_l),
        intercept_right=float(a_r),
        slope_left=float(b_l),
        slope_right=float(b_r),
        bandwidth=float(bandwidth),
        cutoff=float(cutoff),
    )


__all__ = ["RddResult", "rdd"]
