"""Zeno-effect scan: does less-frequent retraining help?

Motivation. The quantum Zeno effect: repeated measurement "freezes"
a system's state, preventing evolution. In finance, a model that's
retrained very frequently might be "frozen" in its most recent
training-window state — never getting the chance to develop skill on
newer data before being reset.

Real-world motivation: a shipped HMM-based regime sleeve was retrained
every 21 trading days. Testing 21d / 63d / 126d / 252d revealed that
126d beat 21d on every metric AND used 6× less compute. Explanation:
the model had a "warm-up" period (days 0–20 in a window were actively
wrong, days 20–40 were peak skill). Frequent retraining kept resetting
the model right as it became skillful.

This tool automates that test. Give it a fit function, a predict
function, and a list of retrain frequencies, and it returns metrics
per frequency.

Design: docs/tools/zenoscan.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from kuant.errors import KuantShapeError, KuantValueError


@dataclass
class ZenoScanResult:
    retrain_freqs: list[int]
    metrics: dict[int, dict[str, float]]  # freq -> metric name -> value
    retrain_counts: dict[int, int]

    def summary(self, metric_order: list[str] | None = None) -> str:
        if not self.retrain_freqs:
            return "(empty result)"
        keys = metric_order or list(next(iter(self.metrics.values())).keys())
        header = f'{"Freq":>6s} ' + " ".join(f"{k:>12s}" for k in keys) + " n_retrain"
        lines = ["=== Zeno-effect retrain-frequency scan ===", header, "-" * len(header)]
        for freq in self.retrain_freqs:
            row = self.metrics[freq]
            values = " ".join(f'{row.get(k, float("nan")):>12.4f}' for k in keys)
            lines.append(f"{freq:>6d} {values} {self.retrain_counts[freq]:>9d}")
        return "\n".join(lines)


def zenoscan(
    fit_fn: Callable[[np.ndarray, np.ndarray], Any],
    predict_fn: Callable[[Any, np.ndarray], np.ndarray],
    metric_fn: Callable[[np.ndarray, np.ndarray], dict[str, float]],
    X: np.ndarray,
    y: np.ndarray,
    retrain_freqs: list[int],
    train_window: int,
) -> ZenoScanResult:
    """Walk-forward retrain-frequency scan.

    Parameters
    ----------
    fit_fn : callable
        `fit_fn(X_train, y_train) -> model`.
    predict_fn : callable
        `predict_fn(model, X_predict) -> y_pred`.
    metric_fn : callable
        `metric_fn(y_true, y_pred) -> dict[str, float]`. Return whatever
        metrics you care about (Sharpe, R², hit rate, ...).
    X : (T, F) np.ndarray
        Feature matrix over T time steps.
    y : (T,) np.ndarray
        Target series.
    retrain_freqs : list of int
        Retrain frequencies (in same time units as X and y) to compare.
    train_window : int
        How many past observations to use per training call.

    Returns
    -------
    ZenoScanResult with per-frequency metrics.

    Examples
    --------
    Compare retraining every 21 days vs every 126 days:

    >>> from sklearn.linear_model import LinearRegression
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(1000, 3))
    >>> y = X @ [0.5, -0.2, 0.1] + rng.normal(scale=0.5, size=1000)
    >>> def fit_fn(Xt, yt):
    ...     return LinearRegression().fit(Xt, yt)
    >>> def predict_fn(m, Xt):
    ...     return m.predict(Xt)
    >>> def metric_fn(y_true, y_pred):
    ...     from numpy import corrcoef
    ...     return {'corr': float(corrcoef(y_true, y_pred)[0, 1])}
    >>> result = zenoscan(
    ...     fit_fn, predict_fn, metric_fn,
    ...     X, y, retrain_freqs=[21, 63, 126], train_window=252,
    ... )
    >>> len(result.metrics) == 3
    True
    """
    if len(X) != len(y):
        raise KuantShapeError(
            f"kuant.zenoscan: 'X' and 'y' must have equal length along "
            f"the time axis; got len(X)={len(X)}, len(y)={len(y)}.  "
            f"[KE-SHAPE-EQUAL-LEN]\n"
            f"  → Fix: align X and y to the same time index before calling"
        )
    T = len(y)
    if train_window >= T:
        raise KuantValueError(
            f"kuant.zenoscan: 'train_window' ({train_window}) is >= "
            f"len(y) ({T}); no walk-forward predictions can be made.  "
            f"[KE-VAL-RANGE]\n"
            f"  → Fix: lower train_window, or provide a longer series"
        )
    for freq in retrain_freqs:
        if not isinstance(freq, (int, np.integer)) or int(freq) <= 0:
            raise KuantValueError(
                f"kuant.zenoscan: 'retrain_freqs' must be strictly "
                f"positive ints; got {freq}.  [KE-VAL-POSITIVE]\n"
                f"  → Fix: pass positive integer retrain frequencies "
                f"(e.g. [21, 63, 126])"
            )
    result = ZenoScanResult(retrain_freqs=list(retrain_freqs), metrics={}, retrain_counts={})

    for freq in retrain_freqs:
        y_pred_full = np.full(T, np.nan)
        n_retrain = 0
        current_model = None

        for t in range(train_window, T):
            # Retrain at boundaries divisible by freq (measured from train_window).
            if (t - train_window) % freq == 0 or current_model is None:
                train_start = t - train_window
                current_model = fit_fn(X[train_start:t], y[train_start:t])
                n_retrain += 1

            # Predict at t using the currently-active model.
            y_pred_full[t] = predict_fn(current_model, X[t : t + 1])[0]

        # Evaluate on the range where we made predictions.
        valid = ~np.isnan(y_pred_full)
        y_true = y[valid]
        y_pred = y_pred_full[valid]
        result.metrics[freq] = metric_fn(y_true, y_pred)
        result.retrain_counts[freq] = n_retrain

    return result
