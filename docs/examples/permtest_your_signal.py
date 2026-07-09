"""permtest_your_signal.py — universal permutation null-test for any signal.

Demonstrates:
  - `kuant.sindy.permtest` — the workhorse null-test primitive
  - Applying it to arbitrary predictor / target / metric combinations
  - How to interpret the p-value (fraction of shuffled trials beating real)

The pattern: whenever you observe a metric M on a real (X, y) pair, shuffle
y N times, recompute M, and see how often the shuffled version beats the
real. High p ⇒ the observed signal isn't distinguishable from noise.

Run:
    python docs/examples/permtest_your_signal.py
"""

from __future__ import annotations

import numpy as np

from kuant.sindy import permtest


def main() -> None:
    rng = np.random.default_rng(7)

    # 1) A real signal: y depends linearly on x plus noise.
    n = 400
    x = rng.normal(size=n)
    y_real = 0.4 * x + rng.normal(scale=0.5, size=n)

    # 2) A null case: y is pure noise, no dependence on x.
    y_null = rng.normal(size=n)

    # 3) Metric: correlation coefficient (higher magnitude → stronger signal).
    def abs_corr(x, y):
        return abs(float(np.corrcoef(x, y)[0, 1]))

    real_metric_signal = abs_corr(x, y_real)
    real_metric_null = abs_corr(x, y_null)
    print("Observed metrics:")
    print(f"  real signal  |corr| = {real_metric_signal:.4f}")
    print(f"  null case    |corr| = {real_metric_null:.4f}")
    print()

    # 4) Permutation test on the real signal.
    result_real = permtest(
        real_metric_signal,
        abs_corr,
        x,
        y_real,
        n_perms=1000,
        seed=0,
        higher_is_better=True,
    )

    # 5) Permutation test on the null case.
    result_null = permtest(
        real_metric_null,
        abs_corr,
        x,
        y_null,
        n_perms=1000,
        seed=1,
        higher_is_better=True,
    )

    print("Permutation p-values:")
    print(
        f"  real signal  p = {result_real.p_value:.4f}   "
        f"({int(result_real.at_least_as_extreme)}/{result_real.n_perms} shuffles beat real)"
    )
    print(
        f"  null case    p = {result_null.p_value:.4f}   "
        f"({int(result_null.at_least_as_extreme)}/{result_null.n_perms} shuffles beat real)"
    )
    print()

    # 6) Sanity: distribution of null-case shuffles centered near real,
    #    while real-signal shuffles are far below real.
    print("Interpretation:")
    print("  p < 0.05  ⇒  observed signal is significantly stronger than random")
    print("  p ~ 0.5   ⇒  observed signal is indistinguishable from noise")
    print()
    print(
        f"  real signal verdict: {'REAL' if result_real.p_value < 0.05 else 'not distinguishable'}"
    )
    print(
        f"  null case verdict:   {'REAL' if result_null.p_value < 0.05 else 'not distinguishable'}"
    )


if __name__ == "__main__":
    main()
