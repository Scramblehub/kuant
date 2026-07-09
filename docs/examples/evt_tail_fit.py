"""evt_tail_fit.py — POT tail fit with Student-t and GPD primitives.

Demonstrates:
  - `kuant.core.tcdf` / `tpdf` for Student-t fat-tail modeling
  - `kuant.core.gpdcdf` / `gpdpdf` for Generalized Pareto POT tail fit
  - `kuant.core.gpdppf` for quantile extrapolation (VaR / tail-loss)
  - Comparison of tail modeling under different distributional assumptions

Peaks-Over-Threshold (POT) is the workhorse of extreme value theory:
pick a high threshold, fit GPD to exceedances above it, extrapolate to
tail quantiles the sample alone can't estimate.

Run:
    python docs/examples/evt_tail_fit.py
"""

from __future__ import annotations

import numpy as np

from kuant.core import gpdpdf, gpdppf, tcdf


def fit_gpd_mle(exceedances: np.ndarray, xi_grid=None, scale_grid=None) -> tuple[float, float]:
    """Grid-search MLE for GPD (xi, scale). Fast and dependency-free for demo.

    Real use should use scipy.stats.genpareto.fit or a proper optimizer; the
    grid keeps this example self-contained inside kuant + numpy.
    """
    if xi_grid is None:
        xi_grid = np.linspace(-0.2, 0.9, 56)
    if scale_grid is None:
        # Auto-scale the search around the median exceedance.
        median_exc = float(np.median(exceedances))
        scale_grid = np.linspace(0.1 * median_exc, 4.0 * median_exc, 80)
    best_ll = -np.inf
    best_xi = 0.0
    best_scale = 1.0
    for xi in xi_grid:
        for scale in scale_grid:
            if scale <= 0:
                continue
            f = gpdpdf(exceedances, xi, scale)
            # log-likelihood; skip if any zero (outside support)
            if np.any(f <= 0):
                continue
            ll = float(np.sum(np.log(f)))
            if ll > best_ll:
                best_ll = ll
                best_xi = float(xi)
                best_scale = float(scale)
    return best_xi, best_scale


def main() -> None:
    rng = np.random.default_rng(1)

    # 1) Simulate returns from a fat-tailed distribution (Student-t with df=5).
    df_true = 5.0
    n = 5000
    # Sample from Student-t; direct via numpy for simplicity.
    returns = rng.standard_t(df_true, size=n) * 0.01

    # 2) Empirical tail: pick the top 5% |return| threshold; work on losses.
    losses = -returns[returns < 0]
    threshold = float(np.quantile(losses, 0.90))  # top 10% of losses
    exceedances = losses[losses > threshold] - threshold
    print(f"n = {n} bars, negative returns = {int(np.sum(returns < 0))}")
    print(f"POT threshold (90th percentile of losses): {threshold:.4f}")
    print(f"n exceedances above threshold: {len(exceedances)}")
    print()

    # 3) Fit GPD to exceedances via grid-MLE.
    xi_hat, scale_hat = fit_gpd_mle(exceedances)
    print(f"GPD fit:  xi = {xi_hat:+.3f}   scale = {scale_hat:.4f}")
    print("          (heavy tail signature is xi > 0)")
    print()

    # 4) Return-period tail estimates from the fitted GPD.
    #    P(loss > threshold) = fraction of losses above threshold =
    #        exceedance_rate = len(exceedances) / n = |negatives| / n × 0.10
    #    For a target unconditional "1-in-K" loss, the conditional
    #    probability of exceeding it is:  p_cond = 1 - 1/(K * exceedance_rate)
    exceedance_rate = len(exceedances) / n
    return_periods = np.array([20, 50, 100, 500, 1000])
    # Conditional-CDF value such that (1 - cdf) * exceedance_rate = 1/K
    p_cond = 1.0 - 1.0 / (return_periods * exceedance_rate)
    # Only valid when p_cond > 0
    valid = p_cond > 0
    losses_at_period = threshold + gpdppf(p_cond[valid], xi_hat, scale_hat)

    print("Return-period tail losses (via GPD extrapolation):")
    print(f"  {'1-in-K bars':>12s}  {'loss threshold':>16s}")
    for K, x in zip(return_periods[valid], losses_at_period):
        print(f"  {K:>12d}  {x:>16.5f}")
    print()

    # 5) Compare with a fitted Student-t (parametric alternative).
    #    Use variance-of-returns to estimate scale, then compute tail quantile.
    var = float(np.var(returns))
    # For Student-t with df dof, var(X) = df/(df-2)*scale^2 for a scaled t.
    # We approximate; a proper fit would use MLE via scipy. Here we just
    # illustrate that tcdf / tpdf are available for tail work.
    df_guess = 5.0
    scale_t = float(np.sqrt(var * (df_guess - 2) / df_guess))
    x_grid = np.array([0.02, 0.03, 0.05, 0.08])
    p_student_t = 1.0 - tcdf(x_grid / scale_t, df_guess)
    print("Student-t (df=5) tail probability of loss > x:")
    print(f"  {'x':>8s}  {'P(loss > x)':>14s}")
    for xi_val, pi_val in zip(x_grid, p_student_t):
        print(f"  {xi_val:>8.4f}  {pi_val:>14.6f}")


if __name__ == "__main__":
    main()
