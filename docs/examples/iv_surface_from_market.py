"""iv_surface_from_market.py — invert market prices to implied volatility surface.

Demonstrates:
  - Vectorized `kuant.options.impvol` (Newton-Raphson) for full-chain inversion
  - `kuant.options.impvolbisection` fallback for pathologically flat-vega points
  - Cross-check via `kuant.core.bsput`: re-price with recovered IV, match input

Simulates a market: prices come from a known "true" IV surface, then we invert.
In a real workflow, `market_price` comes from your data feed.

Run:
    python docs/examples/iv_surface_from_market.py
"""

from __future__ import annotations

import numpy as np

from kuant.core import bsput
from kuant.options import impvol, impvolbisection


def make_synthetic_surface(strikes: np.ndarray, tenors: np.ndarray) -> np.ndarray:
    """Toy skew: quadratic in log-moneyness, mild term structure."""
    K = strikes[:, None]
    T = tenors[None, :]
    S = 100.0
    m = np.log(K / S)  # log-moneyness
    base_iv = 0.20
    smile_curvature = 0.15  # bigger = more smile
    term_slope = 0.05  # bigger = more term-structure tilt
    iv = base_iv + smile_curvature * m * m + term_slope * np.sqrt(T)
    return iv


def main() -> None:
    S = 100.0
    r = 0.05
    q = 0.02

    strikes = np.linspace(85, 115, 13)
    tenors = np.array([1 / 12, 3 / 12, 6 / 12, 1.0, 2.0])

    # 1) Simulate a market with a known IV surface, price puts across the grid.
    true_iv = make_synthetic_surface(strikes, tenors)
    K = strikes[:, None]
    T = tenors[None, :]
    market_price = bsput(S, K, T, r, true_iv, q)

    # 2) Invert every point in one call — Newton-Raphson is fast and vectorized.
    recovered_iv = impvol(market_price, S, K, T, r, is_call=False, q=q)

    # 3) Sanity: re-price with the recovered IV — should match the market price.
    reprice = bsput(S, K, T, r, recovered_iv, q)
    reprice_error = np.max(np.abs(reprice - market_price))

    iv_error = np.max(np.abs(recovered_iv - true_iv))
    print("=== IV inversion (Newton) ===")
    print(f"grid shape:          {recovered_iv.shape} (strikes × tenors)")
    print(f"max IV error:        {iv_error:.2e}")
    print(f"max reprice error:   {reprice_error:.2e}")
    print()

    # 4) Show the recovered smile at the 6-month tenor.
    j = int(np.argmin(np.abs(tenors - 0.5)))
    print("6-month smile (recovered vs true IV):")
    print(f"  {'strike':>7s}  {'true_iv':>8s}  {'recovered':>10s}  {'error':>10s}")
    for i, strike in enumerate(strikes):
        print(
            f"  {strike:>7.1f}  {true_iv[i, j]:>8.5f}  {recovered_iv[i, j]:>10.5f}  "
            f"{abs(recovered_iv[i, j] - true_iv[i, j]):>10.2e}"
        )

    # 5) When Newton might struggle (deep OTM, near-expiry — flat vega),
    #    bisection is the robust fallback. Here true_iv = 0.30.
    otm_short = bsput(100.0, 60.0, 0.02, 0.05, 0.30, 0.02)  # deep OTM 1-wk put
    iv_newton = impvol(otm_short, 100.0, 60.0, 0.02, 0.05, is_call=False, q=0.02)
    iv_bisect = impvolbisection(otm_short, 100.0, 60.0, 0.02, 0.05, is_call=False, q=0.02)
    print()
    print("Deep-OTM 1-week put — Newton vs bisection (true IV = 0.30):")
    print(f"  Newton:    {iv_newton:.6f}   (may drift on flat-vega regions)")
    print(f"  bisection: {iv_bisect:.6f}   (bracketed, guaranteed to converge)")
    print()
    print("Rule of thumb:")
    print("  - Newton (impvol) — fast, use for typical liquid-strike solves")
    print("  - Bisection (impvolbisection) — bulletproof, use when vega is flat")


if __name__ == "__main__":
    main()
