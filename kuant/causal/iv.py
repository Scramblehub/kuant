"""Two-stage least squares Instrumental Variables (2SLS).

Estimates the causal effect of endogenous regressor(s) X on outcome
Y using an instrument Z that (i) correlates with X (relevance) and
(ii) affects Y only through X (exclusion). Solves the endogeneity bias
that plagues naive OLS when X is co-determined with unobservables.

Just-identified case: dim(Z) == dim(X). Over-identified: dim(Z) > dim(X)
(Sargan test recommended but not implemented here: v0.6 scope).

Sign convention: `beta` reports the second-stage coefficient(s) on X.
`f_stat_stage1` gauges instrument strength: first-stage F < 10 is the
Staiger-Stock weak-instrument threshold, and the reported estimate is
unreliable in that regime (KW-IV-WEAK-INSTRUMENT).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from kuant._validation import require_2d
from kuant.errors import KuantNumericWarning, KuantValueError


@dataclass
class IvResult:
    beta: np.ndarray
    se: np.ndarray
    t_stat: np.ndarray
    f_stat_stage1: float
    r2_stage1: float
    r2_stage2: float
    n: int
    k_endog: int
    k_instr: int

    def summary(self) -> str:
        coefs = ", ".join(
            f"{b:+.4f}(t={t:+.2f})" for b, t in zip(self.beta, self.t_stat, strict=False)
        )
        return (
            "=== IvResult ===\n"
            f"2SLS beta:            {coefs}\n"
            f"stage-1 F stat:       {self.f_stat_stage1:.2f} "
            f"({'weak' if self.f_stat_stage1 < 10 else 'strong'})\n"
            f"stage-1 R2:           {self.r2_stage1:.4f}\n"
            f"stage-2 R2:           {self.r2_stage2:.4f}\n"
            f"n / k_endog / k_instr:{self.n} / {self.k_endog} / {self.k_instr}"
        )


def _add_intercept(X: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(X.shape[0]), X])


def iv(y, x_endog, z_instr, *, add_intercept: bool = True) -> IvResult:
    """Two-stage least squares IV.

    Parameters
    ----------
    y : 1D array of length n
    x_endog : 2D array (n, k_endog)
        Endogenous regressor(s).
    z_instr : 2D array (n, k_instr)
        Instrument(s). Must satisfy k_instr >= k_endog.
    add_intercept : bool, default True

    Returns
    -------
    IvResult

    References
    ----------
    Wright 1928 (original IV); Wooldridge Chapter 5 for the modern
    treatment; Staiger & Stock 1997 for the weak-instrument F<10 rule.
    """
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    x_arr = np.atleast_2d(np.asarray(x_endog, dtype=np.float64))
    z_arr = np.atleast_2d(np.asarray(z_instr, dtype=np.float64))
    if x_arr.shape[0] == 1 and x_arr.shape[1] == y_arr.size:
        x_arr = x_arr.T
    if z_arr.shape[0] == 1 and z_arr.shape[1] == y_arr.size:
        z_arr = z_arr.T
    require_2d(x_arr, "x_endog", kernel="iv")
    require_2d(z_arr, "z_instr", kernel="iv")
    n = y_arr.size
    if x_arr.shape[0] != n or z_arr.shape[0] != n:
        raise KuantValueError(
            f"kuant.iv: 'y', 'x_endog', 'z_instr' must all have length n; "
            f"got {n}, {x_arr.shape[0]}, {z_arr.shape[0]}.  "
            f"[KE-SHAPE-EQUAL-LEN]"
        )
    k_endog = x_arr.shape[1]
    k_instr = z_arr.shape[1]
    if k_instr < k_endog:
        raise KuantValueError(
            f"kuant.iv: under-identified: need k_instr ({k_instr}) >= "
            f"k_endog ({k_endog}).  [KE-VAL-RANGE]"
        )
    mask = (
        np.isfinite(y_arr) & np.all(np.isfinite(x_arr), axis=1) & np.all(np.isfinite(z_arr), axis=1)
    )
    if mask.sum() < k_endog + k_instr + 5:
        raise KuantValueError(
            f"kuant.iv: only {int(mask.sum())} clean rows; need "
            f"more than k_endog + k_instr + 5.  [KE-VAL-MIN-CLEAN]"
        )
    y_c = y_arr[mask]
    x_c = x_arr[mask, :]
    z_c = z_arr[mask, :]

    Z = _add_intercept(z_c) if add_intercept else z_c
    # Stage 1: regress each endogenous var on Z; get fitted values X_hat.
    beta_stage1, *_ = np.linalg.lstsq(Z, x_c, rcond=None)
    x_hat = Z @ beta_stage1
    resid1 = x_c - x_hat
    ss_res1 = float(np.sum(resid1**2))
    ss_tot1 = float(np.sum((x_c - x_c.mean(axis=0)) ** 2))
    r2_1 = 1.0 - ss_res1 / (ss_tot1 + 1e-30)

    # Stage-1 F on the excluded instruments (in add_intercept path: cols 1: onwards).
    # Approximation: F = R2/(1-R2) * (n-k_instr-1)/k_instr for single endog case.
    n_c = y_c.size
    if k_endog == 1:
        f_stat = float(r2_1 / max(1.0 - r2_1, 1e-30) * (n_c - k_instr - 1) / k_instr)
    else:
        f_stat = float("nan")

    # Stage 2: regress y on [1, X_hat] to get the 2SLS point estimate.
    X_hat_full = _add_intercept(x_hat) if add_intercept else x_hat
    X_full = _add_intercept(x_c) if add_intercept else x_c
    beta2, *_ = np.linalg.lstsq(X_hat_full, y_c, rcond=None)

    # Correct 2SLS variance: sigma^2 must be computed from residuals
    # against the ORIGINAL X (not X_hat), otherwise SE / t-stats are
    # systematically understated. See Wooldridge Ch. 5 or Hayashi
    # Econometrics Ch. 3.5.
    resid2 = y_c - X_full @ beta2
    dof2 = max(n_c - X_hat_full.shape[1], 1)
    sigma2 = float(np.sum(resid2**2) / dof2)
    XtX_inv = np.linalg.pinv(X_hat_full.T @ X_hat_full)
    var_beta2 = sigma2 * np.diag(XtX_inv)
    se2 = np.sqrt(np.maximum(var_beta2, 0.0))

    ss_tot2 = float(np.sum((y_c - y_c.mean()) ** 2))
    ss_res2 = float(np.sum(resid2**2))
    r2_2 = 1.0 - ss_res2 / (ss_tot2 + 1e-30)

    # Trim intercept from reported beta / t if it was added.
    if add_intercept:
        beta_out = beta2[1:]
        se_out = se2[1:]
    else:
        beta_out = beta2
        se_out = se2
    t_out = beta_out / np.where(se_out > 0, se_out, np.inf)

    if k_endog == 1 and f_stat < 10:
        warnings.warn(
            f"kuant.iv: stage-1 F={f_stat:.2f} below Staiger-Stock 10 "
            f"threshold: instrument is weak; 2SLS estimate is biased "
            f"and its standard errors are unreliable.  "
            f"[KW-IV-WEAK-INSTRUMENT]",
            KuantNumericWarning,
            stacklevel=2,
        )

    return IvResult(
        beta=beta_out.astype(np.float64),
        se=se_out.astype(np.float64),
        t_stat=t_out.astype(np.float64),
        f_stat_stage1=float(f_stat),
        r2_stage1=float(r2_1),
        r2_stage2=float(r2_2),
        n=int(n_c),
        k_endog=int(k_endog),
        k_instr=int(k_instr),
    )


__all__ = ["IvResult", "iv"]
