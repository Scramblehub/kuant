# nocloningscan — Multi-seed model variance (no-cloning theorem analog)

## Purpose

Runs your `(fit + predict) → (predictions, metrics)` pipeline across
N random seeds and compares:

- **Metric variance** — how much do headline numbers (Sharpe, R²)
  fluctuate seed-to-seed?
- **Prediction pair-correlation** — how similar are the actual
  time-series paths across seeds?

Together these answer: **is your model deterministic in outcome even
when internally stochastic in path?**

If yes → seed-averaging (ensemble) is safe and beneficial.
If no → your apparent skill is a random-seed instance; you're
overfitting.

The QM analog: the no-cloning theorem states that an arbitrary
quantum state cannot be perfectly copied. Two "runs" of the same
model may produce different internal states but converge on the
same measurement outcome.

## Public API

```python
from kuant.qm import nocloningscan

def fit_predict(seed):
    # your fit + predict logic, parameterized by seed
    return predictions_array, {'metric_name': metric_value}

result = nocloningscan(fit_predict, n_seeds=10, base_seed=0)
print(result.summary())
```

## Design decisions

### User supplies a single callable

`fit_predict_fn(seed) → (predictions, metrics)`. Keeps
`nocloningscan` unopinionated about model type, dataset, and CV
scheme. All the pipeline stays inside your callable.

### Coefficient of variation as the headline

`CV = std / |mean|`. Small CV = tight seed-to-seed metrics. The
summary tags a verdict based on:

- Pair-corr < 0.95 AND max_metric_CV < 5% → **DIFFERENT PATHS, SAME
  DESTINATION** (safe to ensemble)
- max_metric_CV ≥ 5% → **HIGH SEED VARIANCE** (overfitting warning)
- Otherwise → near-identical seeds (no ensemble benefit)

### Pair correlations across all C(N, 2) pairs

For `n_seeds` up to a few hundred, this is O(N² · T) — fine. For
larger N, sample pairs rather than exhaust them.

## Real-world use

V8 HMM sleeve with 10 seeds:

| Metric | Mean | Std | CV |
|---|---|---|---|
| Sharpe | 2.51 | 0.01 | 0.4% |
| CAGR | 171.6% | 0.3% | 0.2% |

Pair-corr on posterior time-series: **0.66** (very NOT identical).

Verdict: DIFFERENT PATHS, SAME DESTINATION. Seed-ensembling was
approved and shipped as robustness enhancement.

## Related tools

- `kuant.qm.zenoscan` — sibling model-variance diagnostic (over
  retrain frequency instead of seed)
- `kuant.qm.decoherencescan` — sibling model-variance diagnostic (over
  day-in-window instead of seed)
- `kuant.sindy.permtest` — different kind of null: shuffle target
  instead of varying seeds
