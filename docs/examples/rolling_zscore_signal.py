"""rolling_zscore_signal.py — mean-reversion signal from a rolling z-score.

Demonstrates:
  - `kuant.stats.zscore` — rolling standardization in one call
  - How rolling primitives compose (mean + std under the hood)
  - Turning a raw price series into a bounded signal ready for gating

The z-score of returns is a classic mean-reversion primitive: extreme
values above/below zero identify overbought/oversold conditions.

Run:
    python docs/examples/rolling_zscore_signal.py
"""
from __future__ import annotations

import numpy as np

from kuant.stats import rollmean, rollstd, zscore


def main() -> None:
    # Simulate a mean-reverting return series with occasional dislocations.
    rng = np.random.default_rng(42)
    n = 1000
    mu = 0.0002              # tiny drift
    sigma = 0.012            # daily-return-scale vol
    returns = rng.normal(mu, sigma, n)

    # Inject three "dislocation" bursts to see if the signal picks them up.
    for anchor in (250, 500, 750):
        returns[anchor:anchor + 5] += 3.0 * sigma  # 3σ dislocation over 5 days

    # 1) One-liner rolling z-score. Window 63 (~3 trading months).
    z = zscore(returns, window=63)

    # 2) Same computation manually — shows what zscore composes.
    manual_z = (returns - rollmean(returns, 63)) / rollstd(returns, 63, ddof=1)
    max_diff = float(np.nanmax(np.abs(z - manual_z)))
    print(f"zscore() matches (r - rollmean) / rollstd to {max_diff:.2e}")
    print()

    # 3) Report signal shape: how often is the score outside ±2?
    valid = ~np.isnan(z)
    n_valid = int(valid.sum())
    n_extreme = int(np.sum(np.abs(z[valid]) > 2))
    print(f"Bar count: {n} total, {n_valid} valid after warm-up")
    print(f"|z| > 2 fires: {n_extreme} bars ({100 * n_extreme / n_valid:.2f}%)")
    print(f"|z| > 3 fires: {int(np.sum(np.abs(z[valid]) > 3))} bars "
          f"({100 * np.sum(np.abs(z[valid]) > 3) / n_valid:.2f}%)")
    print()

    # 4) For each injection, show max |z| over the 20 bars around it.
    #    (5 bars of +3σ over 63-day window doesn't peak dramatically —
    #     realistic: markets need bigger shocks to cross 2σ signal.)
    print("Peak |z| in a 20-bar window around each injection:")
    for anchor in (250, 500, 750):
        window = z[anchor: anchor + 20]
        window = window[~np.isnan(window)]
        if len(window):
            peak_z = window[np.argmax(np.abs(window))]
            print(f"  bar {anchor}-{anchor+20}: peak z = {peak_z:+.2f}")

    # 5) Composition sanity: for a mean-reversion trade, you'd
    #    trade against extreme z. Example signal generation (no execution):
    signal = np.where(np.isnan(z), 0.0, -np.clip(z / 3.0, -1.0, 1.0))
    print()
    print(f"Sample signal (last 5 bars): {signal[-5:]}")


if __name__ == "__main__":
    main()
