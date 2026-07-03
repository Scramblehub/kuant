"""bs_price_surface.py — vectorized Black-Scholes on a (strike, tenor) grid.

Demonstrates:
  - Broadcasting `kuant.core.bscall` / `bsput` across a 2D grid in one call
  - Batched Greeks (`bscalldelta`, `bsgamma`, `bsvega`) share the same grid
  - How a full option surface is one function call, not a nested loop

Run:
    python docs/examples/bs_price_surface.py
"""
from __future__ import annotations

import numpy as np

from kuant.core import bscall, bsput
from kuant.options import bscalldelta, bsgamma, bsputdelta, bsvega


def main() -> None:
    # A (strike, tenor) grid — 21 strikes × 8 tenors = 168 options
    S = 100.0
    r = 0.05
    q = 0.02
    sigma = 0.25

    strikes = np.linspace(80, 120, 21)        # (21,)
    tenors = np.array([1/52, 1/12, 3/12, 6/12, 1, 2, 3, 5])  # (8,)

    # Broadcasting: strikes goes to shape (21, 1), tenors to (1, 8) → grid (21, 8)
    K = strikes[:, None]
    T = tenors[None, :]

    # One call each: entire surface computed with vectorized ops
    call_price = bscall(S, K, T, r, sigma, q)
    put_price = bsput(S, K, T, r, sigma, q)
    call_delta = bscalldelta(S, K, T, r, sigma, q)
    put_delta = bsputdelta(S, K, T, r, sigma, q)
    gamma = bsgamma(S, K, T, r, sigma, q)
    vega = bsvega(S, K, T, r, sigma, q)

    print(f"Grid shape: strikes={strikes.shape}, tenors={tenors.shape}")
    print(f"Call price surface shape: {call_price.shape}")
    print()

    # ATM slice at each tenor
    atm_idx = np.argmin(np.abs(strikes - S))
    print("ATM slice (K=100):")
    print(f"  {'tenor':>8s}  {'call':>7s}  {'put':>7s}  {'delta_c':>8s}  "
          f"{'gamma':>8s}  {'vega':>7s}")
    for j, tenor in enumerate(tenors):
        print(
            f"  {tenor:>8.4f}  {call_price[atm_idx, j]:>7.3f}  {put_price[atm_idx, j]:>7.3f}  "
            f"{call_delta[atm_idx, j]:>8.4f}  {gamma[atm_idx, j]:>8.4f}  {vega[atm_idx, j]:>7.2f}"
        )

    print()
    # Put-call parity check: call - put = S·e^(-qT) - K·e^(-rT)
    parity_lhs = call_price - put_price
    parity_rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
    max_parity_error = float(np.max(np.abs(parity_lhs - parity_rhs)))
    print(f"Put-call parity max error across grid: {max_parity_error:.2e}")

    # Delta parity: delta_c - delta_p = e^(-qT)
    delta_parity_lhs = call_delta - put_delta
    delta_parity_rhs = np.exp(-q * T)
    max_delta_error = float(np.max(np.abs(delta_parity_lhs - delta_parity_rhs)))
    print(f"Delta parity max error: {max_delta_error:.2e}")


if __name__ == "__main__":
    main()
