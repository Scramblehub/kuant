"""Bell-inequality-style test: does a joint model carry information
inaccessible to classical linear aggregation of the same features?

Motivation. QM's Bell inequality tests whether a joint quantum state
carries super-classical correlations. Applied to financial signal
aggregation: does an HMM's joint posterior over N features carry
predictive information about a target that beats the best classical
(linear or nonlinear) aggregation of the same N features?

If HMM_R² > classical_bound → "Bell violation" → super-classical
structure worth exploiting.
If HMM_R² ≤ classical_bound → feature-level regime detection has
hit its theoretical ceiling; only picker-level or unobservable
structure can add more.

This has been an important negative-result tool in our production
research: it confirmed our HMM sleeve was at the classical bound and
directed further work toward picker-level (not feature-level) alpha.

Design: docs/tools/belltest.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from kuant._validation import require_dep


def _require_sklearn():
    """Lazy import of sklearn — only needed when belltest is called."""
    try:
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.linear_model import LinearRegression, Ridge
        from sklearn.mixture import GaussianMixture
        from sklearn.model_selection import KFold

        return GradientBoostingRegressor, LinearRegression, Ridge, GaussianMixture, KFold
    except ImportError as e:
        require_dep(
            "scikit-learn",
            kernel="belltest",
            install="pip install scikit-learn",
            cause=e,
        )


@dataclass
class BellTestResult:
    """Outcome of a Bell-inequality-style aggregation test."""

    r2_per_feature: dict[str, float]
    r2_linear_all: float
    r2_ridge_all: float
    r2_nonlinear_all: float
    r2_joint_model: float
    classical_bound: float
    joint_beats_bound: bool
    margin_pp: float  # (r2_joint_model - classical_bound) * 100

    def summary(self) -> str:
        lines = ["=== Bell inequality test result ==="]
        lines.append(f"Classical bound (best single-feature R²): {self.classical_bound:.4f}")
        lines.append(f"Joint model R²:                            {self.r2_joint_model:.4f}")
        lines.append(f"Margin (joint - bound):                    {self.margin_pp:+.2f}pp")
        lines.append(f"Violation:                                 {self.joint_beats_bound}")
        lines.append("")
        lines.append("Per-feature R²:")
        for k, v in self.r2_per_feature.items():
            lines.append(f"  {k:<30s} {v:.4f}")
        lines.append(f'  {"OLS multi-linear":<30s} {self.r2_linear_all:.4f}')
        lines.append(f'  {"Ridge multi-linear":<30s} {self.r2_ridge_all:.4f}')
        lines.append(f'  {"GradientBoosting":<30s} {self.r2_nonlinear_all:.4f}')
        return "\n".join(lines)


def _cv_r2(model_fn: Callable, X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> float:
    """K-fold cross-validated R² using out-of-fold predictions."""
    _, _, _, _, KFold = _require_sklearn()
    kf = KFold(n_splits=n_splits, shuffle=False)
    oof = np.zeros_like(y, dtype=np.float64)
    for train_idx, test_idx in kf.split(X):
        m = model_fn()
        m.fit(X[train_idx], y[train_idx])
        oof[test_idx] = m.predict(X[test_idx])
    ss_res = np.sum((y - oof) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def belltest(
    features: dict[str, np.ndarray],
    target: np.ndarray,
    joint_model_fn: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
    n_splits: int = 5,
) -> BellTestResult:
    """Run the Bell-inequality-style aggregation test.

    Parameters
    ----------
    features : dict[str, np.ndarray]
        Named features. Each value is a 1D array of the same length.
    target : 1D np.ndarray
        Target variable (e.g. forward returns).
    joint_model_fn : callable, optional
        A function `(X, y) -> y_pred` that fits and predicts on the joint
        feature matrix. If None, defaults to fitting an HMM-like clustering
        via Gaussian Mixture and using cluster-posteriors as predictors.
        Users can pass their actual HMM here.
    n_splits : int, default 5
        Cross-validation folds for each R² computation.

    Returns
    -------
    BellTestResult

    Notes
    -----
    The "classical bound" is the max R² across:
      - Each single-feature linear regression
      - Multi-feature OLS
      - Multi-feature Ridge (regularized)
      - GradientBoosting (nonlinear)

    If the joint model can't beat this bound, feature-level regime work is
    likely spent — the joint model is just an efficient linear aggregator,
    not a source of new information.
    """
    GradientBoostingRegressor, LinearRegression, Ridge, GaussianMixture, KFold = _require_sklearn()

    feature_names = list(features.keys())
    X = np.column_stack([features[k].astype(np.float64) for k in feature_names])
    y = target.astype(np.float64)

    # Per-feature linear R²
    r2_per = {}
    for i, name in enumerate(feature_names):
        r2_per[name] = _cv_r2(LinearRegression, X[:, i : i + 1], y, n_splits=n_splits)

    # Multi-feature classical models
    r2_lin = _cv_r2(LinearRegression, X, y, n_splits=n_splits)
    r2_ridge = _cv_r2(lambda: Ridge(alpha=1.0), X, y, n_splits=n_splits)
    r2_gbr = _cv_r2(lambda: GradientBoostingRegressor(random_state=0), X, y, n_splits=n_splits)

    # Classical bound = max over all classical predictors
    classical_bound = max(list(r2_per.values()) + [r2_lin, r2_ridge, r2_gbr])

    # Joint model
    if joint_model_fn is None:
        # Default: Gaussian Mixture posterior as HMM-style joint aggregator
        def _default_joint(X_tr, y_tr, X_te):
            gm = GaussianMixture(n_components=3, random_state=0)
            gm.fit(X_tr)
            post = gm.predict_proba(X_te)  # (n_test, n_components)
            # Predict y from posterior using linear regression on train
            gm_post_train = gm.predict_proba(X_tr)
            lr = LinearRegression().fit(gm_post_train, y_tr)
            return lr.predict(post)

        # Manual K-fold for the compound fit
        kf = KFold(n_splits=n_splits, shuffle=False)
        oof = np.zeros_like(y)
        for tr, te in kf.split(X):
            oof[te] = _default_joint(X[tr], y[tr], X[te])
    else:
        kf = KFold(n_splits=n_splits, shuffle=False)
        oof = np.zeros_like(y)
        for tr, te in kf.split(X):
            oof[te] = (
                joint_model_fn(X[tr], y[tr])[te]
                if len(joint_model_fn(X[tr], y[tr])) == len(y)
                else joint_model_fn(X[tr], y[tr])
            )
        # Note: user-supplied model_fn is expected to return either full-length
        # out-of-fold predictions or fold-specific ones; we assume full-length.

    ss_res = np.sum((y - oof) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2_joint = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    margin_pp = (r2_joint - classical_bound) * 100
    return BellTestResult(
        r2_per_feature=r2_per,
        r2_linear_all=r2_lin,
        r2_ridge_all=r2_ridge,
        r2_nonlinear_all=r2_gbr,
        r2_joint_model=r2_joint,
        classical_bound=classical_bound,
        joint_beats_bound=r2_joint > classical_bound,
        margin_pp=margin_pp,
    )
