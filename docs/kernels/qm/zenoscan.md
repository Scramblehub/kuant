# zenoscan — Retrain-frequency scan (Zeno-effect check)

## Purpose

Walk-forward retrain-frequency scan. Compare `metric_fn` values across
several retrain frequencies to detect the quantum-Zeno-effect analog:
frequent retraining "freezes" the model in its current state,
preventing it from developing skill on newer data.

Motivated by a real V8 HMM finding: retraining every 21 days was
worse on every metric than every 126 days, AND used 6× the compute.
The mechanism was a non-monotonic "warm-up" period where the model
was actively wrong for the first ~20 days after each retrain.

## Public API

```python
from kuant.qm import zenoscan

result = zenoscan(
    fit_fn, predict_fn, metric_fn,
    X, y,
    retrain_freqs=[21, 63, 126, 252],
    train_window=252,
)
print(result.summary())
```

## Design decisions

### Walk-forward, out-of-sample metrics

At each time step `t >= train_window`, if `(t - train_window) %
retrain_freq == 0`, retrain the model on `X[t-train_window:t]`. In
between retrains, the same fitted model predicts each new bar. This
matches the operational pattern of a production model with a
scheduled retrain cadence.

### User supplies fit / predict / metric callables

No sklearn dependency in this module — the user provides their own
model interface. This keeps `zenoscan` unopinionated about model type
(any linear model, tree ensemble, HMM, or neural net can be fed in).

### Retrain count reported per frequency

The `retrain_counts` field records how many retrains actually
happened. Useful for comparing compute cost, not just metrics.

## Real-world use in our research

V8 HMM sleeve: compared 21d, 63d, 126d, 252d. Result:

| Freq | CAGR | Sharpe | MDD | Retrains |
|---|---|---|---|---|
| 21d | 171.57% | 2.509 | –24.19% | 145 |
| 63d | 168.97% | 2.503 | –24.95% | 49 |
| **126d** | **172.77%** | **2.523** | **–23.83%** | 25 |
| 252d | 171.93% | 2.530 | –27.09% | 13 |

**126d beat 21d on every metric and used 6× less compute.** Shipped as
a production change.

## Related tools

- `kuant.qm.hmm` — the model class we validated this on
- `kuant.qm.belltest` — companion QM-inspired experiment tool
