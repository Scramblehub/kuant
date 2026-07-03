"""LASSO-with-CV feature-library scan.

Motivation. Given a target time series and a rich library of candidate
features (lagged versions of X, non-linear transforms, interactions,
derivatives), fit L1-regularized regression with cross-validated
regularization strength. The L1 penalty auto-selects a sparse subset;
the CV picks the right level of sparsity.

If CV picks the STRONGEST regularization at the top of your search
range (i.e., forces the LASSO to select zero features), that's an
unambiguous null-signal signature. A common example from prior
research on a residuals-after-multi-factor-wash target: LASSO selected
0 of 22 candidate features and picked the top-of-range alpha,
diagnosing the residual as unpredictable at daily frequency.

If it selects a few features and the R² beats a shuffled-target
baseline, you have candidate signals.

Design: docs/kernels/sindy/sindylasso.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from kuant._validation import require_dep, require_equal_length, require_min_clean


@dataclass
class SindyLassoResult:
    selected_features: dict[str, float]  # feature name -> coefficient (non-zero only)
    alpha_selected: float
    alpha_grid: np.ndarray
    r2: float  # OOF R² at chosen alpha
    n_features_in_library: int
    intercept: float

    def summary(self) -> str:
        n_sel = len(self.selected_features)
        lines = [
            "=== SINDy LASSO feature-library scan ===",
            f"Library size:            {self.n_features_in_library}",
            f"Selected (non-zero):     {n_sel}",
            f"Alpha chosen (by CV):    {self.alpha_selected:.4g}",
            f"Alpha grid range:        [{self.alpha_grid[0]:.4g}, {self.alpha_grid[-1]:.4g}]",
            f"OOF R² at chosen alpha:  {self.r2:.4f}",
            f"Intercept:               {self.intercept:+.4f}",
        ]

        # Null-signal warning: alpha at top of range + zero features selected
        if n_sel == 0:
            top = self.alpha_grid.max()
            if abs(self.alpha_selected - top) < 1e-12:
                lines.append("")
                lines.append("DIAGNOSTIC: CV picked the strongest regularization AND selected")
                lines.append("zero features. Signal-to-noise is below threshold across the entire")
                lines.append("library. This is a clean null result.")

        if n_sel > 0:
            lines.append("")
            lines.append("Selected features (largest |coef| first):")
            ordered = sorted(self.selected_features.items(), key=lambda kv: -abs(kv[1]))
            for name, coef in ordered:
                lines.append(f"  {name:<30s} {coef:+.4f}")
        return "\n".join(lines)


def _require_sklearn():
    try:
        from sklearn.linear_model import LassoCV
        from sklearn.model_selection import KFold

        return LassoCV, KFold
    except ImportError as e:
        require_dep(
            "scikit-learn",
            kernel="sindylasso",
            install="pip install scikit-learn",
            cause=e,
        )


def sindylasso(
    target: np.ndarray,
    library: dict[str, np.ndarray],
    alpha_grid: Optional[np.ndarray] = None,
    n_splits: int = 5,
    max_iter: int = 10000,
) -> SindyLassoResult:
    """LASSO-with-CV feature-library scan.

    Parameters
    ----------
    target : 1D np.ndarray
        Target series (e.g. next-day return residual).
    library : dict[str, 1D np.ndarray]
        Feature library. Values must be the same length as `target`.
    alpha_grid : 1D np.ndarray, optional
        L1 penalty grid to CV over. Default `np.logspace(-5, -1, 30)`.
    n_splits : int, default 5
        CV folds. Uses `KFold` in order (no shuffle) to respect the
        time-series structure of the data.
    max_iter : int, default 10000
        Max iterations per LASSO fit.

    Returns
    -------
    SindyLassoResult

    Notes
    -----
    Uses `sklearn.linear_model.LassoCV`. Lazy sklearn dep.

    The `library` values are stacked column-wise into an `(n_samples,
    n_features)` design matrix in the order given by `library.keys()`.
    NaN-containing rows are dropped before fitting.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> n = 500
    >>> x1 = rng.normal(size=n)
    >>> x2 = rng.normal(size=n)
    >>> noise = rng.normal(scale=0.3, size=n)
    >>> y = 0.5 * x1 + noise  # x2 has zero coefficient by construction
    >>> library = {'x1': x1, 'x2': x2, 'noise_col': rng.normal(size=n)}
    >>> result = sindylasso(y, library)
    >>> 'x1' in result.selected_features
    True
    """
    LassoCV, KFold = _require_sklearn()

    if alpha_grid is None:
        alpha_grid = np.logspace(-5, -1, 30)

    feature_names = list(library.keys())
    X = np.column_stack([np.asarray(library[k], dtype=np.float64) for k in feature_names])
    y = np.asarray(target, dtype=np.float64)

    require_equal_length(X, "library", y, "target", kernel="sindylasso")

    # Drop rows with any NaN.
    mask = np.isfinite(np.column_stack([X, y[:, None]])).all(axis=1)
    X_clean, y_clean = X[mask], y[mask]
    require_min_clean(
        y_clean, "target", kernel="sindylasso", min_count=30, purpose="fit LASSO with CV"
    )

    kf = KFold(n_splits=n_splits, shuffle=False)
    model = LassoCV(
        alphas=alpha_grid,
        cv=kf,
        max_iter=max_iter,
        n_jobs=1,
    )
    model.fit(X_clean, y_clean)

    # Selected features = non-zero coefficients
    selected = {
        name: float(coef) for name, coef in zip(feature_names, model.coef_) if abs(coef) > 1e-12
    }

    # OOF R² at chosen alpha
    oof = np.zeros_like(y_clean)
    from sklearn.linear_model import Lasso

    for tr, te in kf.split(X_clean):
        m = Lasso(alpha=float(model.alpha_), max_iter=max_iter)
        m.fit(X_clean[tr], y_clean[tr])
        oof[te] = m.predict(X_clean[te])
    ss_res = float(np.sum((y_clean - oof) ** 2))
    ss_tot = float(np.sum((y_clean - y_clean.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return SindyLassoResult(
        selected_features=selected,
        alpha_selected=float(model.alpha_),
        alpha_grid=np.asarray(alpha_grid),
        r2=r2,
        n_features_in_library=len(feature_names),
        intercept=float(model.intercept_),
    )
