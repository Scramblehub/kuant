"""Heteroskedasticity-and-autocorrelation-consistent (HAC) standard errors.

Two workhorses for regression inference under time-series dependence:

- `neweywestse`: Newey-West 1987 HAC estimator. Bartlett kernel with a
  data-driven or user-specified bandwidth. The most-cited HAC.
- `andrewsse`: Andrews 1991 automatic-bandwidth HAC using the quadratic
  spectral kernel. Better small-sample properties than Newey-West;
  slower to compute.

Both return the HAC covariance matrix of `beta = (X'X)^{-1} X'y` under
the model `y = X beta + u`, with standard errors on the diagonal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kuant._validation import require_positive
from kuant.errors import KuantShapeError, KuantValueError


@dataclass
class HacResult:
    beta: np.ndarray
    se: np.ndarray
    cov: np.ndarray
    residuals: np.ndarray
    n: int
    k: int
    bandwidth: int
    kernel: str

    def summary(self) -> str:
        return (
            "=== HacResult ===\n"
            f"kernel:      {self.kernel}\n"
            f"bandwidth:   {self.bandwidth}\n"
            f"n / k:       {self.n} / {self.k}\n"
            f"beta:        {self.beta}\n"
            f"se:          {self.se}\n"
            f"t-stats:     {self.beta / self.se}"
        )


def _check_xy(y, X, kernel: str):
    y = np.asarray(y, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    if y.ndim != 1:
        raise KuantShapeError(
            f"kuant.{kernel}: 'y' must be 1D, got {y.ndim}D.  [KE-SHAPE-EXPECTED]"
        )
    if X.ndim != 2:
        raise KuantShapeError(
            f"kuant.{kernel}: 'X' must be 2D, got {X.ndim}D.  [KE-SHAPE-EXPECTED]"
        )
    if X.shape[0] != y.size:
        raise KuantShapeError(
            f"kuant.{kernel}: X.shape[0] ({X.shape[0]}) != len(y) ({y.size}).  "
            f"[KE-SHAPE-EQUAL-LEN]"
        )
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    y = y[mask]
    X = X[mask]
    if y.size < X.shape[1] + 10:
        raise KuantValueError(
            f"kuant.{kernel}: after dropping non-finite rows, {y.size} rows "
            f"and {X.shape[1]} regressors; need at least k+10 rows.  "
            f"[KE-VAL-MIN-CLEAN]"
        )
    return y, X


def _ols(y, X):
    XtX = X.T @ X
    XtXinv = np.linalg.inv(XtX)
    beta = XtXinv @ (X.T @ y)
    resid = y - X @ beta
    return beta, resid, XtXinv


def neweywestse(y, X, *, bandwidth: int | None = None) -> HacResult:
    """Newey-West HAC standard errors (Bartlett kernel).

    Parameters
    ----------
    y : 1D array, length n
    X : 2D array, shape (n, k)
    bandwidth : int, optional
        Truncation lag. If None, uses `floor(4 * (n/100) ** (2/9))`
        (Newey-West 1994 rule-of-thumb).

    Returns
    -------
    HacResult

    References
    ----------
    Newey & West 1987, "A simple positive semi-definite covariance
    matrix." Newey & West 1994 for the automatic bandwidth.
    """
    y, X = _check_xy(y, X, "neweywestse")
    n, k = X.shape
    if bandwidth is None:
        bandwidth = int(np.floor(4 * (n / 100.0) ** (2 / 9.0)))
    bandwidth = max(int(bandwidth), 1)
    require_positive(bandwidth, "bandwidth", kernel="neweywestse", kind="int")

    beta, resid, XtXinv = _ols(y, X)
    u = resid[:, None] * X  # score contributions

    # HAC middle: sum of autocovariances weighted by Bartlett kernel.
    S = u.T @ u
    for lag in range(1, bandwidth + 1):
        w = 1.0 - lag / (bandwidth + 1.0)
        Gamma = u[lag:].T @ u[:-lag]
        S += w * (Gamma + Gamma.T)
    cov = XtXinv @ S @ XtXinv
    se = np.sqrt(np.diag(cov))
    return HacResult(
        beta=beta,
        se=se,
        cov=cov,
        residuals=resid,
        n=int(n),
        k=int(k),
        bandwidth=int(bandwidth),
        kernel="newey-west",
    )


def andrewsse(y, X, *, bandwidth: int | None = None) -> HacResult:
    """Andrews HAC standard errors (quadratic spectral kernel).

    Parameters
    ----------
    y : 1D array, length n
    X : 2D array, shape (n, k)
    bandwidth : int, optional
        If None, uses Andrews 1991 AR(1) plug-in bandwidth.

    Returns
    -------
    HacResult

    References
    ----------
    Andrews 1991, "Heteroskedasticity and autocorrelation consistent
    covariance matrix estimation."
    """
    y, X = _check_xy(y, X, "andrewsse")
    n, k = X.shape
    beta, resid, XtXinv = _ols(y, X)

    if bandwidth is None:
        # Andrews AR(1) plug-in on the sum of scores.
        u = resid[:, None] * X
        s = u.sum(axis=1)
        # AR(1) coefficient on s.
        rho = float(np.corrcoef(s[:-1], s[1:])[0, 1]) if s.size > 2 else 0.0
        rho = max(min(rho, 0.97), -0.97)
        # Andrews 1991 Table 1: alpha(2) = 4 rho^2 / (1 - rho)^4 for the
        # QS / Parzen / Tukey-Hanning kernels under a scalar AR(1) plug-in.
        alpha2 = 4 * rho**2 / (1 - rho) ** 4 if rho != 0 else 0.0
        bandwidth = int(np.ceil(1.3221 * (alpha2 * n) ** (1 / 5.0))) if alpha2 > 0 else 1
    bandwidth = max(int(bandwidth), 1)

    u = resid[:, None] * X

    def qs_kernel(z):
        # Quadratic spectral kernel; z is the scaled lag.
        if abs(z) < 1e-12:
            return 1.0
        x = 6 * np.pi * z / 5.0
        return (25.0 / (12.0 * (np.pi * z) ** 2)) * (np.sin(x) / x - np.cos(x))

    S = u.T @ u
    # QS decays quickly; sum lags up to `n // 3` conservatively.
    max_lag = min(int(n) - 1, 3 * bandwidth)
    for lag in range(1, max_lag + 1):
        w = qs_kernel(lag / bandwidth)
        if abs(w) < 1e-8:
            continue
        Gamma = u[lag:].T @ u[:-lag]
        S += w * (Gamma + Gamma.T)
    cov = XtXinv @ S @ XtXinv
    se = np.sqrt(np.diag(cov))
    return HacResult(
        beta=beta,
        se=se,
        cov=cov,
        residuals=resid,
        n=int(n),
        k=int(k),
        bandwidth=int(bandwidth),
        kernel="andrews-qs",
    )


__all__ = ["HacResult", "neweywestse", "andrewsse"]
