# mes — Marginal Expected Shortfall

## Purpose

Expected loss of an individual asset conditional on the SYSTEM being
in its worst `tau` fraction of days. Introduced by
Acharya-Pedersen-Philippon-Richardson 2017 as a systemic-risk
building block: the individual's contribution to system-wide tail
losses.

```math
\text{MES}_i = \mathbb{E}\left[ -r_i \mid r_{\text{system}} \in \text{worst-}\tau \right]
```

reported as a positive loss magnitude.

Complements `covar`: CoVaR conditions on the system being at a
specific quantile POINT; MES averages across a tail SLAB. Same
information, different summary.

## Public API

```python
from kuant.risk import mes

result = mes(returns_asset, returns_system, tau=0.05)
print(result.summary())
print(result.mes, result.n_tail_days)
```

- `returns_asset`, `returns_system`: 1D arrays of equal length.
  Non-finite pairs stripped jointly.
- `tau`: tail fraction in `[0.001, 0.5]`. Default `0.05` (worst 5%
  of system days).

Returns `MesResult` with fields `mes`, `system_var`, `n_tail_days`,
`n`, `tau`.

## Design decisions

### 1. Empirical tail selection

`threshold = quantile(returns_system, tau)` (a NEGATIVE number since
`tau` is small). Tail mask is `returns_system <= threshold`. MES is
`-mean(returns_asset[tail_mask])`, sign-flipped to a positive loss.
`system_var` is `-threshold`, also positive.

No parametric model. If the sample is too small to trust this
empirical average, escalate to a parametric conditional model; for
the sample sizes we permit (`n >= 100`), the direct empirical
estimator is standard.

### 2. Tail fraction, not confidence level

`mes` takes `tau` (worst fraction), matching the systemic-risk
literature convention. `tau = 0.05` = worst 5% of system days. The
rest of `kuant.risk` uses `alpha` (confidence level) where `1 -
alpha` plays the same role; the naming split is intentional and
follows Acharya-Pedersen-Philippon-Richardson 2017.

### 3. Positive-loss sign convention

`mes > 0` means the asset LOSES money on average during system tail
days. `mes < 0` means the asset GAINS during system tail days, that
is, a natural hedge. The kernel does not clamp sign.

### 4. Minimum sample and length parity

- `n_finite_pairs < 100` raises `KE-VAL-MIN-CLEAN`.
- `len(returns_asset) != len(returns_system)` raises
  `KE-SHAPE-EQUAL-LEN` before NaN masking.
- Empty tail (`n_tail == 0` after quantile, only reachable at
  extreme edge conditions) raises `KE-VAL-MIN-CLEAN`.

## Edge cases

| Condition | Behavior |
| --- | --- |
| Length mismatch | `KuantValueError` with `KE-SHAPE-EQUAL-LEN` |
| Fewer than 100 paired finite values | `KuantValueError` with `KE-VAL-MIN-CLEAN` |
| `tau` out of `[0.001, 0.5]` | `KuantValueError` with `KE-VAL-RANGE` |
| Empty tail after quantile | `KuantValueError` with `KE-VAL-MIN-CLEAN` |
| Asset independent of system | `mes ~ 0` |
| Asset perfectly tracks system | `mes ~ system_var` |
| Asset anti-correlated with system | `mes < 0` (natural hedge) |
| Non-finite entries in either input | paired stripping |

## Cross-check tests

- `test_mes_positive_when_asset_moves_with_system`: 2k draws,
  `asset = 0.8 * sys + 0.2 * noise`. `mes > 0`, `n_tail_days > 50`.
- `test_mes_near_zero_when_independent`: 5k independent Gaussians.
  `|mes| < 0.003`.
- `test_mes_length_mismatch`: differing lengths raise.
- `test_mes_min_clean_gate`: only 2 finite pairs raises with
  `KE-VAL-MIN-CLEAN`.

## Direct usage in kuant

Position-level systemic scan. Sweep `mes` across every position in a
book against a common system return (SPY, or the book's own gross
exposure). Positive MES concentrates the book's systemic beta;
negative MES flags natural hedges. Delta-MES over time is a stress
signal.

## Related kernels

- `kuant.risk.covar`: quantile-point counterpart. CoVaR fits a
  quantile regression and reads off a single conditional VaR; MES
  averages losses across the tail slab.
- `kuant.risk.esbootstrap`: `esbootstrap` on the asset returns
  restricted to system tail days would give a CI on MES; queued as
  a downstream composition, not a kernel today.

## References

- Acharya, V., Pedersen, L., Philippon, T., Richardson, M. 2017.
  "Measuring Systemic Risk." *Review of Financial Studies* 30(1):
  2-47.
