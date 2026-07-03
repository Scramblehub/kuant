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
change on the V8 HMM sleeve.

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

## Real-world use

V8 HMM sleeve, 252d prediction window:

| Day-in-window | Correlation | Notes |
|---|---|---|
| 0..20 | **–0.156** | Actively wrong (recency bias from train tail) |
| 20..40 | **+0.203** | Peak skill |
| 40..60 | +0.111 | Decaying |
| 60..100 | +0.108 | Slow decay |
| 100..150 | –0.013 | Noise |
| 150..252 | +0.078 | Partial recovery |

Non-monotonic. Peak at 20..40 days. Since we were retraining every
21 days, we were RESETTING the model right at peak skill. Increased
retrain freq to 126d → shipped as a production win.

## Related tools

- `kuant.qm.zenoscan` — the natural follow-up: given a warm-up
  finding, scan retrain frequencies to find the sweet spot
- `kuant.qm.nocloningscan` — different variance axis (seed vs
  day-in-window)
