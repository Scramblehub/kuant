# kuant.risk

Tail-risk and systemic-risk measures. Shipped in v0.6.0 batch 9.

Five kernels split cleanly into two families:

**Univariate tail (single return series)**

1. [`cornishfishervar`](cornishfishervar.md): Cornish-Fisher 1937
   expansion VaR. Adjusts the Gaussian quantile for sample skew and
   excess kurtosis. Cheap, closed form, degrades gracefully for
   mild non-normality. Use for near-Gaussian series; switch to
   `evtvar` past `|skew| > 1` or `|excess kurt| > 7`.
2. [`evtvar`](evtvar.md): Peaks-over-Threshold GPD VaR and Expected
   Shortfall. Fits a Generalized Pareto tail via method of moments
   (Hosking-Wallis 1987) on excesses above an empirical quantile.
   The right tool for genuinely heavy-tailed series (t with `df` in
   the single digits, equity crash losses, etc.).
3. [`esbootstrap`](esbootstrap.md): Moving-block bootstrap CI on
   Expected Shortfall. Quantifies the sample-variance of an ES point
   estimate (Kunsch 1989 block bootstrap). Answers "is this ES change
   real, or is it inside the noise band."

**Bivariate systemic (asset conditional on system)**

4. [`covar`](covar.md): Adrian-Brunnermeier 2016 CoVaR via quantile
   regression. VaR of one series conditional on the other being at
   its own VaR. Also reports delta-CoVaR (incremental tail exposure
   from tail dependence).
5. [`mes`](mes.md): Marginal Expected Shortfall
   (Acharya-Pedersen-Philippon-Richardson 2017). Expected loss of an
   individual asset averaged over the worst `tau` fraction of system
   days. Complements CoVaR: CoVaR conditions on a quantile point,
   MES averages across a tail slab.

## Conventions across the subpackage

**Sign.** All VaR / ES / CoVaR / MES outputs are POSITIVE loss
magnitudes. A larger number means a worse loss. This matches
industry reporting and lets downstream code stack results without
sign gymnastics.

**Confidence level.** `alpha` is the CONFIDENCE level, not the tail
probability. `alpha = 0.95` targets the worst 5% of returns.
`cornishfishervar`, `evtvar`, `esbootstrap`, and `covar` all use
`alpha` this way. `mes` uses `tau` for the tail FRACTION (`tau =
0.05` = worst 5%), matching the systemic-risk literature.

**Backend.** All five are numpy-only (float64). Bootstrap and
quantile regression carry inner loops; a cupy port is queued for a
later batch if profiling shows real hot-loop cost.

**Return type.** Each kernel returns a frozen dataclass with a
`.summary()` method that prints the diagnostic block. Point
estimates plus the ancillary state (sample moments, tail counts,
regression slope) needed to reason about the estimate.

**Error / warning codes.** All shared:

- `KE-VAL-MIN-CLEAN`: not enough finite observations. Minimums are
  30 (`cornishfishervar`), 100 (`esbootstrap`, `covar`, `mes`),
  250 (`evtvar`), plus 20 GPD exceedances for `evtvar`.
- `KE-VAL-RANGE`: parameter out of allowed range (`alpha`,
  `tau`, `threshold_pct`, `block_size`).
- `KE-SHAPE-EQUAL-LEN`: bivariate kernels require matching lengths.
- `KW-CF-EXPANSION-INVALID`: Cornish-Fisher expansion outside its
  safe region.
- `KW-EVT-MOM-INVALID`: MOM GPD fit near its validity boundary.

## When to reach for what

| Situation | Kernel |
| --- | --- |
| Near-Gaussian returns, need cheap VaR | `cornishfishervar` |
| Heavy tails, need VaR AND ES | `evtvar` |
| Need a CI band on ES | `esbootstrap` |
| Systemic: "how much do I lose when the system tanks" | `mes` |
| Systemic: "VaR of X given Y at ITS VaR" | `covar` |

## Related kernels

- [`kuant.stats`](../stats/README.md): `rollmean`, `rollstd`,
  `zscore` are the rolling-window building blocks that feed
  time-varying versions of these estimators.
- [`kuant.portfolio`](../portfolio/README.md): downstream position
  sizing consumes VaR / ES point estimates.

## References

- Acharya, V., Pedersen, L., Philippon, T., Richardson, M. 2017.
  "Measuring Systemic Risk." *Review of Financial Studies*.
- Adrian, T., Brunnermeier, M. 2016. "CoVaR." *American Economic
  Review*.
- Balkema, A., de Haan, L. 1974. "Residual life time at great age."
  *Annals of Probability*.
- Cont, R., Deguest, R., Scandolo, G. 2010. "Robustness and
  sensitivity analysis of risk measurement procedures."
  *Quantitative Finance*.
- Cornish, E., Fisher, R. 1937. "Moments and cumulants in the
  specification of distributions." *Revue de l'Institut International
  de Statistique*.
- Hosking, J., Wallis, J. 1987. "Parameter and quantile estimation
  for the generalized Pareto distribution." *Technometrics*.
- Koenker, R., Bassett, G. 1978. "Regression quantiles."
  *Econometrica*.
- Kunsch, H. 1989. "The jackknife and the bootstrap for general
  stationary observations." *Annals of Statistics*.
- McNeil, A., Frey, R., Embrechts, P. 2015. *Quantitative Risk
  Management*, 2nd ed. Princeton University Press.
- Pickands, J. 1975. "Statistical inference using extreme order
  statistics." *Annals of Statistics*.
- Politis, D., Romano, J. 1994. "The stationary bootstrap." *Journal
  of the American Statistical Association*.
