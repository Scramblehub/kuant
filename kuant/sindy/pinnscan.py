'''Nonlinear feature-library scan via GradientBoosting + permutation null.

Motivation. If a LASSO scan (`sindylasso`) came back empty, that only
rules out sparse LINEAR structure. Interactions and non-linearities
between library features could still carry real signal. This tool
tests the nonlinear case: fit a GradientBoostingRegressor on the whole
library, get out-of-fold predictions, and run a permutation test to
confirm the fit is above the noise floor.

Named after "physics-informed neural network"-adjacent modeling — the
idea is that when a rich hand-engineered feature library represents
your physical intuition, a nonlinear regressor should be able to
recover any real interaction. If GBR + permutation says "no", you can
retire the entire library.

Canonical example this catches: a rich library (~20 hand-engineered
features) gives a small but non-zero OOF correlation, a quintile gate
built on the OOF predictions looks meaningfully positive on headline
metrics, but the permutation p sits around 0.5 — half of shuffled-
target runs produce the same gate strength. Without the permutation
step this ships as a real signal; with it, the null is decisive.

Design: docs/kernels/sindy/pinnscan.md.
'''
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .permtest import permtest


@dataclass
class PinnScanResult:
    r2_oof: float
    corr_oof: float
    feature_importances: dict[str, float]
    permutation_p: float
    n_perms: int
    n_features: int

    def summary(self) -> str:
        lines = [
            '=== Nonlinear (PINN-lite) feature-library scan ===',
            f'Features in library:      {self.n_features}',
            f'OOF R²:                   {self.r2_oof:.4f}',
            f'OOF correlation:          {self.corr_oof:+.4f}',
            f'Permutation p-value:      {self.permutation_p:.4f} (n={self.n_perms})',
            f'Signal beats shuffle:     {self.permutation_p < 0.05}',
            '',
            'Top 10 feature importances:',
        ]
        top = sorted(self.feature_importances.items(),
                     key=lambda kv: -kv[1])[:10]
        for name, imp in top:
            lines.append(f'  {name:<30s} {imp:.4f}')

        if self.permutation_p >= 0.05:
            lines.append('')
            lines.append('DIAGNOSTIC: OOF fit does NOT survive permutation. The library')
            lines.append('carries no nonlinear signal above the noise floor. If sindylasso')
            lines.append('was also null on this library, retire it — the joint search space')
            lines.append('has been explored.')
        return '\n'.join(lines)


def _require_sklearn():
    try:
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.model_selection import KFold
        return GradientBoostingRegressor, KFold
    except ImportError as e:
        raise ImportError(
            'kuant.sindy.pinnscan requires scikit-learn. '
            'Install with: pip install scikit-learn'
        ) from e


def pinnscan(
    target: np.ndarray,
    library: dict[str, np.ndarray],
    n_splits: int = 5,
    n_perms: int = 200,
    n_estimators: int = 100,
    random_state: int = 0,
) -> PinnScanResult:
    '''Nonlinear feature-library scan with a permutation null.

    Fits a GradientBoostingRegressor over the whole library using
    KFold CV to get out-of-fold predictions, then runs a permutation
    test on the OOF R² to confirm the fit is above the noise floor.

    Parameters
    ----------
    target : 1D np.ndarray
        Target series.
    library : dict[str, 1D np.ndarray]
        Feature library (same length as target).
    n_splits : int, default 5
        CV folds. Time-series order preserved (no shuffle).
    n_perms : int, default 200
        Permutations for the null test.
    n_estimators : int, default 100
        Boosting rounds for the GBR.
    random_state : int, default 0
        For GBR reproducibility.

    Returns
    -------
    PinnScanResult
    '''
    GradientBoostingRegressor, KFold = _require_sklearn()

    feature_names = list(library.keys())
    X = np.column_stack([np.asarray(library[k], dtype=np.float64) for k in feature_names])
    y = np.asarray(target, dtype=np.float64)

    if X.shape[0] != y.size:
        raise ValueError(
            f'target length {y.size} != features length {X.shape[0]}'
        )

    mask = np.isfinite(np.column_stack([X, y[:, None]])).all(axis=1)
    X_clean, y_clean = X[mask], y[mask]
    if len(y_clean) < 30:
        raise ValueError(f'too few clean rows ({len(y_clean)}) after NaN drop')

    kf = KFold(n_splits=n_splits, shuffle=False)

    def _oof_r2(X_arg, y_arg):
        oof = np.zeros_like(y_arg)
        for tr, te in kf.split(X_arg):
            m = GradientBoostingRegressor(
                n_estimators=n_estimators, random_state=random_state,
            )
            m.fit(X_arg[tr], y_arg[tr])
            oof[te] = m.predict(X_arg[te])
        ss_res = float(np.sum((y_arg - oof) ** 2))
        ss_tot = float(np.sum((y_arg - y_arg.mean()) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0, oof

    real_r2, real_oof = _oof_r2(X_clean, y_clean)

    # Correlation from OOF predictions (more forgiving than R² on noisy targets).
    if np.std(real_oof) > 1e-12:
        corr_oof = float(np.corrcoef(real_oof, y_clean)[0, 1])
    else:
        corr_oof = 0.0

    # Feature importances from a single fit on the full clean data.
    full_model = GradientBoostingRegressor(
        n_estimators=n_estimators, random_state=random_state,
    ).fit(X_clean, y_clean)
    importances = {
        name: float(imp)
        for name, imp in zip(feature_names, full_model.feature_importances_)
    }

    # Permutation p-value on OOF R².
    def _r2_metric(X_arg, y_arg):
        r2, _ = _oof_r2(X_arg, y_arg)
        return r2

    perm_result = permtest(
        real_r2, _r2_metric, X_clean, y_clean,
        n_perms=n_perms, seed=random_state, higher_is_better=True,
    )

    return PinnScanResult(
        r2_oof=real_r2,
        corr_oof=corr_oof,
        feature_importances=importances,
        permutation_p=perm_result.p_value,
        n_perms=n_perms,
        n_features=len(feature_names),
    )
