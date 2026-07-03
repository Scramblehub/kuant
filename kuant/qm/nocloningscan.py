"""Multi-seed model variance — a "no-cloning theorem" analog.

The QM no-cloning theorem: an arbitrary quantum state cannot be
perfectly copied. Applied to stochastic models: two runs of the same
"model" trained on the same data with different random seeds may
produce meaningfully DIFFERENT internal states (paths / posteriors)
even when their aggregate METRICS are indistinguishable.

Practical implication: if per-seed metric variance is small but
per-seed prediction correlation is <1, you have "different paths,
same destination." That's a green light to average across seeds for
robustness — you're removing seed-dependent noise without losing skill.

If per-seed metric variance is LARGE, seed diversity IS the source of
apparent skill: your model is not robust and you're overfitting to a
random-seed instance.

Practical example: 10 seeds of an HMM-based regime sleeve produced
posterior time-series correlated at ~0.66 across seeds (very NOT
identical paths) while headline metrics were tight (sub-1% CV).
Verdict from this tool: different paths, same destination — seed-
ensembling shipped as a robustness enhancement.

Design: docs/kernels/qm/nocloningscan.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from kuant._validation import require_range

import numpy as np


@dataclass
class NoCloningScanResult:
    """Multi-seed model variance analysis."""

    seed_metrics: dict[int, dict[str, float]]  # seed -> metric name -> value
    metric_stats: dict[str, dict[str, float]]  # metric name -> {mean, std, cv}
    prediction_pair_corr_mean: float  # mean corr across seed pairs
    prediction_pair_corr_std: float
    n_seeds: int

    def summary(self) -> str:
        lines = [
            "=== Multi-seed model variance ===",
            f"Seeds:                        {self.n_seeds}",
            f"Prediction pair-corr mean:    {self.prediction_pair_corr_mean:.4f}",
            f"Prediction pair-corr std:     {self.prediction_pair_corr_std:.4f}",
            "",
        ]
        lines.append(f'{"Metric":<20s} {"mean":>10s} {"std":>10s} {"CV":>10s}')
        for name, stats in self.metric_stats.items():
            lines.append(
                f'{name:<20s} {stats["mean"]:>10.4f} {stats["std"]:>10.4f} {stats["cv"]:>10.4f}'
            )

        pcorr = self.prediction_pair_corr_mean
        max_cv = max(s["cv"] for s in self.metric_stats.values()) if self.metric_stats else 0.0
        lines.append("")
        if pcorr < 0.95 and max_cv < 0.05:
            lines.append("Verdict: DIFFERENT PATHS, SAME DESTINATION — seed-averaging safe.")
        elif max_cv >= 0.05:
            lines.append(
                "Verdict: HIGH SEED VARIANCE — model may be overfitting to a random-seed instance."
            )
        else:
            lines.append("Verdict: near-identical seeds — no ensembling benefit.")
        return "\n".join(lines)


def nocloningscan(
    fit_predict_fn: Callable[[int], tuple[np.ndarray, dict[str, float]]],
    n_seeds: int,
    base_seed: int = 0,
) -> NoCloningScanResult:
    """Run `fit_predict_fn(seed)` across N seeds and summarize variance.

    Parameters
    ----------
    fit_predict_fn : callable
        `fit_predict_fn(seed) -> (predictions_array, metrics_dict)`.
        User implements this to encapsulate their fit+predict logic
        parameterized by a random seed.
    n_seeds : int
        Number of seeds to run.
    base_seed : int, default 0
        Passed as `base_seed + i` for `i in range(n_seeds)`.

    Returns
    -------
    NoCloningScanResult

    Notes
    -----
    Cost is `n_seeds` sequential invocations of `fit_predict_fn`.
    Prediction pair-correlations are computed across all C(n_seeds, 2)
    pairs. For large n_seeds, cost is O(n_seeds²) in the correlation
    computation but negligible vs the fit calls.

    Examples
    --------
    >>> from sklearn.linear_model import LinearRegression
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X_all = rng.normal(size=(500, 3))
    >>> y_all = X_all @ [0.5, -0.2, 0.1] + rng.normal(scale=0.5, size=500)
    >>> def fit_predict(seed):
    ...     rng = np.random.default_rng(seed)
    ...     idx = rng.choice(500, size=300, replace=False)  # random subsample
    ...     m = LinearRegression().fit(X_all[idx], y_all[idx])
    ...     p = m.predict(X_all)
    ...     return p, {'r2': float(m.score(X_all, y_all))}
    >>> result = nocloningscan(fit_predict, n_seeds=5)
    >>> result.n_seeds
    5
    """
    require_range(
        n_seeds,
        "n_seeds",
        kernel="nocloningscan",
        lo=2,
        hi=float("inf"),
    )

    seed_predictions: dict[int, np.ndarray] = {}
    seed_metrics: dict[int, dict[str, float]] = {}

    for i in range(n_seeds):
        s = base_seed + i
        pred, metrics = fit_predict_fn(s)
        seed_predictions[s] = np.asarray(pred, dtype=np.float64)
        seed_metrics[s] = dict(metrics)

    # Metric statistics
    metric_names = list(next(iter(seed_metrics.values())).keys())
    metric_stats: dict[str, dict[str, float]] = {}
    for name in metric_names:
        values = np.array([seed_metrics[s][name] for s in seed_metrics])
        mean = float(values.mean())
        std = float(values.std())
        cv = float(std / abs(mean)) if abs(mean) > 1e-12 else float("inf")
        metric_stats[name] = {"mean": mean, "std": std, "cv": cv}

    # Prediction pair correlations
    seeds = list(seed_predictions.keys())
    pair_corrs = []
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            pi, pj = seed_predictions[seeds[i]], seed_predictions[seeds[j]]
            if pi.size >= 2 and pj.size >= 2:
                c = np.corrcoef(pi, pj)[0, 1]
                if np.isfinite(c):
                    pair_corrs.append(c)
    pair_corrs = np.array(pair_corrs) if pair_corrs else np.array([1.0])

    return NoCloningScanResult(
        seed_metrics=seed_metrics,
        metric_stats=metric_stats,
        prediction_pair_corr_mean=float(pair_corrs.mean()),
        prediction_pair_corr_std=float(pair_corrs.std()),
        n_seeds=n_seeds,
    )
