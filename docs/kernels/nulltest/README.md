# kuant.nulltest: Prove this signal isn't noise

Tools for the "is this alpha real?" question that every backtest
eventually has to answer.

Each kernel targets a specific null hypothesis that a plausible-
looking result could hide behind:

- **`stationary_bootstrap`**: resample a serially-correlated series
  while preserving short-range dependence. The building block for
  every other test in this subpackage.
- **`bootstrap_ic`**: CI and two-sided p-value for a signal's
  Information Coefficient against forward returns. Null: true IC is
  zero.
- **`spa_test`**: Hansen (2005) Superior Predictive Ability. Null:
  no alternative strategy beats the benchmark. Corrects for the
  number of alternatives tested.
- **`mcs_test`**: Hansen, Lunde & Nason (2011) Model Confidence Set.
  Iteratively eliminates provably-worse strategies until only a
  statistically-indistinguishable "survivor" set remains.
- **`mht_correction`**: Bonferroni / Holm / BH adjustment for a
  bulk vector of raw p-values from a screen.

## When to use which

- **Single signal, one IC number.** `bootstrap_ic` for CI and p-value.
- **Family of strategies vs a benchmark.** `spa_test` for a single
  joint p-value.
- **Family of strategies, want the survivor set.** `mcs_test`.
- **Raw p-values from an existing screen.** `mht_correction`.
- **Sharpe-specific selection-bias adjustment.** Cross to
  `kuant.portfolio.deflated_sharpe`.

## Design theme

Every test uses stationary-block resampling under the hood (see
`stationary_bootstrap`) because return series are serially
correlated and i.i.d. bootstrap would understate the null variance.
Every dataclass returns a `.summary()` string alongside its raw
fields so results are printable straight into a tearsheet.

## Individual pages

- [`bootstrap.md`](bootstrap.md): `stationary_bootstrap`,
  `bootstrap_ic`, `BootstrapICResult`.
- [`spa_test.md`](spa_test.md): `spa_test`, `mcs_test`, `SPAResult`.
- [`mht_correction.md`](mht_correction.md): Bonferroni, Holm, BH.
