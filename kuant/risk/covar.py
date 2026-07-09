"""Adrian-Brunnermeier CoVaR (conditional VaR).

CoVaR measures the VaR of asset X given that asset Y is at its own
VaR level. When X = "the system" and Y = "an individual institution",
CoVaR quantifies how much the system loses when the institution is
in tail distress -- the workhorse systemic-risk measure.

Implementation: rolling quantile regression (Koenker-Bassett 1978) of
X on Y evaluated at Y = Q_alpha(Y). Sign convention: CoVaR reported
as POSITIVE loss magnitude.

Two variants are commonly reported:
- CoVaR: VaR of X | Y at its VaR (a stress-conditional VaR)
- delta CoVaR: CoVaR minus the unconditional VaR of X (i.e. the
  incremental tail exposure created by the tail dependence)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kuant._validation import require_1d, require_range
from kuant.errors import KuantValueError


@dataclass
class CoVarResult:
    covar: float
    delta_covar: float
    var_x_uncond: float
    var_y_uncond: float
    q_regression_slope: float
    alpha: float
    n: int

    def summary(self) -> str:
        return (
            "=== CoVarResult ===\n"
            f"CoVaR (X | Y=q_Y):    {self.covar:+.6f}\n"
            f"delta CoVaR:          {self.delta_covar:+.6f}\n"
            f"VaR_X (uncond):       {self.var_x_uncond:+.6f}\n"
            f"VaR_Y (uncond):       {self.var_y_uncond:+.6f}\n"
            f"q-reg slope:          {self.q_regression_slope:+.4f}\n"
            f"alpha:                {self.alpha}\n"
            f"n:                    {self.n}"
        )


def _quantile_regression_1d(x, y, tau: float, n_iter: int = 2000):
    """Simple quantile regression y = a + b*x via subgradient descent.

    Minimizes sum(rho_tau(y - a - b*x)) where rho_tau(u) is the check
    loss. Standardizes x and y internally so a single learning rate
    generalizes across problem scales, then undoes the transformation.
    For production accuracy prefer scipy.optimize or statsmodels.
    """
    mx, my = float(x.mean()), float(y.mean())
    sx = float(x.std(ddof=0)) or 1.0
    sy = float(y.std(ddof=0)) or 1.0
    xs = (x - mx) / sx
    ys = (y - my) / sy

    a = float(np.quantile(ys, tau))
    b = 0.0
    lr = 0.05
    for k in range(int(n_iter)):
        u = ys - a - b * xs
        w = np.where(u < 0, tau - 1, tau)
        grad_a = -np.mean(w)
        grad_b = -np.mean(w * xs)
        step = lr / (1.0 + k / 200.0)
        a -= step * grad_a
        b -= step * grad_b

    # Undo standardization: y = a_std*sy + my + (b_std*sy/sx)*(x - mx)
    b_orig = b * sy / sx
    a_orig = a * sy + my - b_orig * mx
    return a_orig, b_orig


def covar(returns_x, returns_y, *, alpha: float = 0.95) -> CoVarResult:
    """Adrian-Brunnermeier CoVaR via quantile regression.

    Parameters
    ----------
    returns_x, returns_y : 1D arrays of equal length
        `X` is the "system" and `Y` is the individual asset (or vice
        versa depending on framing).
    alpha : float, default 0.95

    Returns
    -------
    CoVarResult

    References
    ----------
    Adrian & Brunnermeier 2016, "CoVaR." American Economic Review.
    """
    arr_x = np.asarray(returns_x, dtype=np.float64)
    arr_y = np.asarray(returns_y, dtype=np.float64)
    require_1d(arr_x, "returns_x", kernel="covar")
    require_1d(arr_y, "returns_y", kernel="covar")
    if arr_x.size != arr_y.size:
        raise KuantValueError(
            f"kuant.covar: 'returns_x' and 'returns_y' must be equal "
            f"length; got {arr_x.size} and {arr_y.size}.  "
            f"[KE-SHAPE-EQUAL-LEN]"
        )
    mask = np.isfinite(arr_x) & np.isfinite(arr_y)
    xf = arr_x[mask]
    yf = arr_y[mask]
    if xf.size < 100:
        raise KuantValueError(
            f"kuant.covar: only {xf.size} paired finite values; need at "
            f"least 100.  [KE-VAL-MIN-CLEAN]"
        )
    require_range(alpha, "alpha", kernel="covar", lo=0.5, hi=0.9999)

    losses_x = -xf
    losses_y = -yf
    var_x_uncond = float(np.quantile(losses_x, alpha))
    var_y_uncond = float(np.quantile(losses_y, alpha))
    var_y_median = float(np.quantile(losses_y, 0.5))

    # Quantile regression of loss_X on loss_Y at level alpha.
    a, b = _quantile_regression_1d(losses_y, losses_x, alpha)

    # CoVaR = predicted loss_X when loss_Y = VaR_Y.
    covar_val = a + b * var_y_uncond
    # Median-conditional counterpart for delta.
    covar_median = a + b * var_y_median
    delta = covar_val - covar_median

    return CoVarResult(
        covar=float(covar_val),
        delta_covar=float(delta),
        var_x_uncond=float(var_x_uncond),
        var_y_uncond=float(var_y_uncond),
        q_regression_slope=float(b),
        alpha=float(alpha),
        n=int(xf.size),
    )


__all__ = ["CoVarResult", "covar"]
