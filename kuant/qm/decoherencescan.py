'''Within-window decoherence scan — how model skill decays vs day-in-window.

Motivation. If you train a model on `[t-train_window, t)` and then use
it to predict for `[t, t + predict_window)` without re-training, the
model's skill vs actual outcomes need not be monotone in the day-in-
window. On an HMM-based sleeve in prior research, we found:

  Day in window     Correlation of prediction with realized
  0..20             -0.156 (actively wrong)
  20..40            +0.203 (peak skill)
  40..60            +0.111 (decaying)
  60..100           +0.108 (slow decay)
  100..150          -0.013 (noise)
  150..252          +0.078 (partial recovery)

Non-monotonic. The initial-window recency bias made the model *worse*
than random for the first 20 days after each retrain — which explained
why frequent retraining was worse than infrequent retraining. This tool
is the diagnostic that produces that table for any (fit_fn, predict_fn)
combination.

Design: docs/kernels/qm/decoherencescan.md.
'''
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


@dataclass
class DecoherenceScanResult:
    '''Per-bucket correlation of prediction vs realized target.'''
    bucket_bounds: list[tuple[int, int]]
    bucket_corr: list[float]
    bucket_n: list[int]
    is_monotonic: bool
    peak_bucket_idx: int
    peak_corr: float

    def summary(self) -> str:
        lines = [
            '=== Within-window decoherence scan ===',
            f'Monotonic decay: {self.is_monotonic}',
            f'Peak bucket:     {self.bucket_bounds[self.peak_bucket_idx]} '
            f'(corr {self.peak_corr:+.4f})',
            '',
            f'{"Day-in-window":<15s} {"corr":>10s} {"n":>10s}',
        ]
        for (lo, hi), c, n in zip(self.bucket_bounds, self.bucket_corr, self.bucket_n):
            marker = ' <- peak' if (lo, hi) == self.bucket_bounds[self.peak_bucket_idx] else ''
            lines.append(f'{f"{lo:>3d}..{hi:<3d}":<15s} {c:>+10.4f} {n:>10d}{marker}')
        return '\n'.join(lines)


def decoherencescan(
    fit_fn: Callable[[np.ndarray, np.ndarray], Any],
    predict_fn: Callable[[Any, np.ndarray], np.ndarray],
    X: np.ndarray,
    y: np.ndarray,
    train_window: int,
    predict_window: int,
    buckets: list[tuple[int, int]] | None = None,
) -> DecoherenceScanResult:
    '''Bucket predictions by day-in-window and measure correlation with target.

    Parameters
    ----------
    fit_fn : callable
        `fit_fn(X_train, y_train) -> model`.
    predict_fn : callable
        `predict_fn(model, X_bar) -> y_pred` for a single bar or slice.
    X : (T, F) np.ndarray
        Feature matrix.
    y : (T,) np.ndarray
        Target.
    train_window : int
        Training window size.
    predict_window : int
        Number of bars predicted per fit (no retrain within this window).
    buckets : list[(int, int)], optional
        Day-in-window buckets. Default divides `predict_window` into 5
        equal segments.

    Returns
    -------
    DecoherenceScanResult

    Notes
    -----
    Walk-forward scheme: at each `t = train_window, train_window +
    predict_window, ...`, fit once and predict the next `predict_window`
    bars. Then group predictions by `day - t` (the offset from the
    fit time) and compute correlation with realized `y` per bucket.

    A non-monotonic decay pattern is a warning sign that retraining
    frequency should be re-tuned — usually LESS frequent retraining
    beats MORE frequent (see zenoscan).
    '''
    T = len(y)
    if predict_window < 2:
        raise ValueError(f'predict_window must be >= 2, got {predict_window}')

    if buckets is None:
        step = predict_window // 5 if predict_window >= 5 else 1
        buckets = [(i, min(i + step, predict_window))
                   for i in range(0, predict_window, step)]

    bucket_days = [np.arange(lo, hi) for lo, hi in buckets]

    # Walk-forward: fit once per stride.
    y_pred_all = np.full(T, np.nan)
    day_in_win_all = np.full(T, -1, dtype=np.int64)

    t = train_window
    while t + predict_window <= T:
        model = fit_fn(X[t - train_window:t], y[t - train_window:t])
        pred = predict_fn(model, X[t:t + predict_window])
        pred = np.asarray(pred).ravel()
        y_pred_all[t:t + predict_window] = pred
        day_in_win_all[t:t + predict_window] = np.arange(predict_window)
        t += predict_window

    valid = ~np.isnan(y_pred_all)
    bucket_corr = []
    bucket_n = []
    for days in bucket_days:
        mask = valid & np.isin(day_in_win_all, days)
        n = int(mask.sum())
        if n >= 2:
            c = float(np.corrcoef(y_pred_all[mask], y[mask])[0, 1])
            if not np.isfinite(c):
                c = 0.0
        else:
            c = 0.0
        bucket_corr.append(c)
        bucket_n.append(n)

    peak_idx = int(np.argmax(bucket_corr))
    is_monotonic = all(
        bucket_corr[i] >= bucket_corr[i + 1] - 1e-12
        for i in range(len(bucket_corr) - 1)
    )

    return DecoherenceScanResult(
        bucket_bounds=list(buckets),
        bucket_corr=bucket_corr,
        bucket_n=bucket_n,
        is_monotonic=is_monotonic,
        peak_bucket_idx=peak_idx,
        peak_corr=float(bucket_corr[peak_idx]),
    )
