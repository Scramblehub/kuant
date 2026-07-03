# decoherencescan — Within-window skill decay diagnostic

## Purpose

Walk-forward decomposition of model skill by **day-in-window** — that
is, how correlation between prediction and realized target evolves
across the prediction horizon after each fit.

If skill is monotonically decaying, everything is fine. If skill is
NON-MONOTONIC (bad first, then peaks, then decays), you have a
"warm-up" period. Your retraining schedule may be resetting the model
right as it's becoming useful.

This is the diagnostic that motivated our 21d → 126d retrain-frequency
change on an HMM-based sleeve.

## Public API

```python
from kuant.qm import decoherencescan

result = decoherencescan(
    fit_fn, predict_fn,
    X, y,
    train_window=252,
    predict_window=252,
    buckets=None,   # default: 5 equal segments of predict_window
)
print(result.summary())
```

## Design decisions

### Walk-forward scheme

At each `t = train_window`, `train_window + predict_window`, ...:
fit once on `[t-train_window, t)`, predict for `[t, t+predict_window)`
without retraining. Repeat until end.

Then group all predictions by `day - t` (their offset from the fit
time) and compute correlation with realized target per bucket.

### User supplies fit / predict callables

Unopinionated about model type. Uses the same interface as `zenoscan`
so both tools can share user code.

### Default 5-bucket split

`buckets = [(0, predict_window/5), (predict_window/5, 2·predict_window/5), ...]`.
Users can override for finer resolution or targeted probes.

### Non-monotonic flag

`is_monotonic = True` iff `bucket_corr[i] ≥ bucket_corr[i+1]` for all
i. If False, a non-monotonic decay pattern was detected — treat as a
warning that retrain frequency might be over-tuned.

## Canonical non-monotonic pattern

The pattern that motivates this tool: a walk-forward bucketed
correlation series that is NEGATIVE for the first ~20 days after each
fit (recency bias from the training tail overpowers signal), PEAKS
around days 20–40, then decays through the rest of the prediction
window. If your retrain cadence is close to the "peak skill" bucket
boundary, you are resetting the model during its most useful window.
Extending the retrain cadence past the peak-skill bucket is a
frequently-shipping remedy — see `kuant.qm.zenoscan` for the natural
follow-up.

## Related tools

- `kuant.qm.zenoscan` — the natural follow-up: given a warm-up
  finding, scan retrain frequencies to find the sweet spot
- `kuant.qm.nocloningscan` — different variance axis (seed vs
  day-in-window)
