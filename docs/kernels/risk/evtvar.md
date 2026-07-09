# evtvar — Peaks-over-Threshold GPD VaR and Expected Shortfall

## Purpose

Extreme value theory VaR and ES via the Peaks-over-Threshold (POT)
method. Fits a Generalized Pareto Distribution to the empirical
excesses above a high threshold, then reads VaR and ES off the
fitted tail. This is the correct estimator once returns are
genuinely heavy-tailed (Student-t with low df, equity crash regimes,
credit losses).

Under Pickands 1975 and Balkema-de Haan 1974, the tail excesses of
essentially any distribution converge to a GPD as the threshold
rises. That gives POT a rigorous asymptotic footing that neither
Gaussian VaR nor `cornishfishervar` can claim in the tail.

## Public API

```python
from kuant.risk import evtvar

result = evtvar(returns, alpha=0.99, threshold_pct=0.90)
print(result.summary())
print(result.var, result.es)  # POSITIVE loss magnitudes
```

- `returns`: 1D array-like. Non-finite entries stripped.
- `alpha`: confidence level in `[0.5, 0.9999]`. Default `0.99`.
- `threshold_pct`: empirical quantile of `|returns|` used as the GPD
  threshold, in `[0.5, 0.99]`. Default `0.90`. Standard practice
  sits in `[0.90, 0.95]`.

Returns `EvtVarResult` with fields `var`, `es`, `threshold`, `xi`,
`sigma`, `n_exceedances`, `n_total`, `alpha`.

## Design decisions

### 1. Fit `-returns`, not `returns`

Losses live in the LEFT tail of returns. The kernel internally flips
sign, so all downstream statistics (threshold, excesses, VaR, ES)
are in loss space and reported positive.

### 2. Method of moments (Hosking-Wallis 1987)

```math
\hat\xi     = \frac{1}{2} \left( 1 - \frac{\bar y^2}{s^2_y} \right)
\hat\sigma  = \frac{1}{2} \bar y \left( 1 + \frac{\bar y^2}{s^2_y} \right)
```

with `y` = excesses above threshold `u`. Closed form, no optimizer,
robust for `xi < 0.5`. For heavier tails (`xi >= 0.5`), MOM biases
downward and PWM or MLE is the correct escalation. A
`KuantNumericWarning` with code `KW-EVT-MOM-INVALID` fires when
`xi >= 0.4` (buffer inside the `0.5` boundary).

### 3. McNeil-Frey VaR / ES closed forms

Once `(xi, sigma, u)` are fitted with `N_u` excesses out of `n`:

```math
\text{VaR}_\alpha = u + \frac{\sigma}{\xi}
                    \left(
                      \left( \frac{n}{N_u} (1 - \alpha) \right)^{-\xi} - 1
                    \right)
\text{ES}_\alpha  = \frac{\text{VaR}_\alpha + \sigma - \xi u}{1 - \xi}
                    \quad \text{for } \xi < 1
```

For `|xi| < 1e-10` the VaR formula degenerates and uses the
exponential limit `u + sigma * (-log(ratio))`. For `xi >= 1` the ES
is infinite in expectation and reported as `inf`.

### 4. Minimum samples and minimum exceedances

Two hard gates:

- `n_finite < 250` raises `KE-VAL-MIN-CLEAN`. Below that, tail
  inference is unreliable regardless of threshold.
- `n_exceedances < 20` raises `KE-VAL-MIN-CLEAN`. Below that, the
  MOM moments of the excesses are too noisy.

Users hitting the second gate should lower `threshold_pct` or feed
more data.

### 5. Positive-loss sign convention

Same as the rest of `kuant.risk`. VaR and ES are positive numbers.
Larger means worse.

## Edge cases

| Condition | Behavior |
| --- | --- |
| `var_exc < 1e-15` (degenerate excesses) | returns `xi = 0`, `sigma = 0`, `var = es = u` |
| `xi >= 1` | ES reported as `inf` (mean of GPD tail diverges) |
| `xi` near `0.5` | `KuantNumericWarning` with `KW-EVT-MOM-INVALID` |
| `|xi| < 1e-10` | uses exponential-limit VaR formula |
| `threshold_pct` too high (few exceedances) | `KuantValueError` with `KE-VAL-MIN-CLEAN` |
| `alpha` or `threshold_pct` out of range | `KuantValueError` with `KE-VAL-RANGE` |
| `n_finite < 250` | `KuantValueError` with `KE-VAL-MIN-CLEAN` |
| Non-finite entries | stripped before fitting |

## Cross-check tests

- `test_evtvar_recovers_var_and_es_ordering`: 5k Student-t (df=4)
  draws. `es >= var > threshold`, at least 100 exceedances at
  `threshold_pct = 0.90`.
- `test_evtvar_rejects_short_series`: `n = 100` raises the min-clean
  gate.
- `test_evtvar_rejects_too_high_threshold`: `threshold_pct = 0.99`
  on `n = 300` leaves too few exceedances and raises.

## Direct usage in kuant

Primary tail-risk estimator once `cornishfishervar` warns.
Downstream position-sizing consumes `.var` and `.es` as positive
loss numbers. `.xi` doubles as a heaviness diagnostic: track it
over a rolling window and treat sharp jumps as tail-regime changes.

## Related kernels

- `kuant.risk.cornishfishervar`: cheaper alternative for near-Gaussian
  series; this kernel is the escalation path.
- `kuant.risk.esbootstrap`: gives a CI on the historical ES point
  estimate. The two are complementary: `evtvar` extrapolates INTO
  the tail via a fitted GPD; `esbootstrap` quantifies noise on the
  empirical ES you already have.

## References

- Balkema, A., de Haan, L. 1974. "Residual life time at great age."
  *Annals of Probability*.
- Hosking, J., Wallis, J. 1987. "Parameter and quantile estimation
  for the generalized Pareto distribution." *Technometrics*.
- Pickands, J. 1975. "Statistical inference using extreme order
  statistics." *Annals of Statistics*.
- McNeil, A., Frey, R., Embrechts, P. 2015. *Quantitative Risk
  Management*, 2nd ed., ch. 7 on POT.
