'''Polynomial-symbolic regression scan.

Motivation. Between LASSO on raw features (`sindylasso`) and
GradientBoosting on the same library (`pinnscan`), there's a gap:
compact symbolic non-linear relationships. Squared terms and pairwise
interactions are the sweet spot — often physically motivated
("gamma effect", "convexity term") and cheap to fit.

This tool builds a polynomial feature expansion (default degree 2:
squares + pairwise interactions), then runs LASSO with CV on the
expanded library. Non-zero coefficients give you a compact
polynomial equation.

V8 SINDy #9 (symbolic regression null): polynomial degree-2 expansion
of a candidate library, then LASSO. Result: no polynomial improvement
over linear. Documented as a null; the mechanism is the same as
sindylasso, just with an interaction basis.

Design: docs/kernels/sindy/symbolicscan.md.
'''
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class SymbolicScanResult:
    selected_terms: dict[str, float]     # symbolic term string -> coefficient
    alpha_selected: float
    alpha_grid: np.ndarray
    r2: float
    n_terms_in_expansion: int
    degree: int
    intercept: float

    def summary(self) -> str:
        lines = [
            '=== Symbolic (polynomial) regression scan ===',
            f'Polynomial degree:        {self.degree}',
            f'Expanded library size:    {self.n_terms_in_expansion}',
            f'Selected (non-zero):      {len(self.selected_terms)}',
            f'Alpha chosen (by CV):     {self.alpha_selected:.4g}',
            f'OOF R²:                   {self.r2:.4f}',
            f'Intercept:                {self.intercept:+.4f}',
        ]

        if self.selected_terms:
            lines.append('')
            lines.append('Selected polynomial equation (largest |coef| first):')
            ordered = sorted(self.selected_terms.items(),
                             key=lambda kv: -abs(kv[1]))
            eqn_terms = [f'{self.intercept:+.4f}']
            for term, coef in ordered:
                sign = '+' if coef >= 0 else '-'
                eqn_terms.append(f'{sign} {abs(coef):.4f}·{term}')
                lines.append(f'  {term:<30s} {coef:+.4f}')
            lines.append('')
            lines.append('  y ≈ ' + ' '.join(eqn_terms))
        else:
            lines.append('')
            lines.append('DIAGNOSTIC: LASSO selected zero polynomial terms. No compact')
            lines.append('symbolic structure detected. Try pinnscan for a full nonlinear')
            lines.append('search, or accept the null.')
        return '\n'.join(lines)


def _require_sklearn():
    try:
        from sklearn.linear_model import Lasso, LassoCV
        from sklearn.model_selection import KFold
        from sklearn.preprocessing import PolynomialFeatures
        return Lasso, LassoCV, KFold, PolynomialFeatures
    except ImportError as e:
        raise ImportError(
            'kuant.sindy.symbolicscan requires scikit-learn. '
            'Install with: pip install scikit-learn'
        ) from e


def symbolicscan(
    target: np.ndarray,
    features: dict[str, np.ndarray],
    degree: int = 2,
    alpha_grid: Optional[np.ndarray] = None,
    n_splits: int = 5,
    max_iter: int = 10000,
    include_bias: bool = False,
) -> SymbolicScanResult:
    '''Polynomial-symbolic regression scan via LASSO on a polynomial expansion.

    Parameters
    ----------
    target : 1D np.ndarray
        Target series.
    features : dict[str, 1D np.ndarray]
        Base features to expand. Each value must be the same length as target.
    degree : int, default 2
        Polynomial degree. `2` gives linear + squares + pairwise
        interactions; `3` adds cubes and triple interactions.
    alpha_grid : 1D np.ndarray, optional
        LASSO alpha grid. Default `np.logspace(-5, -1, 30)`.
    n_splits : int, default 5
        CV folds (KFold, no shuffle).
    max_iter : int, default 10000
        Max LASSO iterations.
    include_bias : bool, default False
        Whether the polynomial expansion includes the intercept column.
        Default False since LASSO handles the intercept separately.

    Returns
    -------
    SymbolicScanResult

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> n = 500
    >>> x1 = rng.normal(size=n)
    >>> x2 = rng.normal(size=n)
    >>> noise = rng.normal(scale=0.3, size=n)
    >>> y = 0.5 * x1 * x2 + noise  # pure interaction term
    >>> result = symbolicscan(y, {'x1': x1, 'x2': x2})
    >>> 'x1 x2' in result.selected_terms or 'x1*x2' in result.selected_terms
    True
    '''
    Lasso, LassoCV, KFold, PolynomialFeatures = _require_sklearn()

    if degree < 1:
        raise ValueError(f'degree must be >= 1, got {degree}')

    if alpha_grid is None:
        alpha_grid = np.logspace(-5, -1, 30)

    feature_names = list(features.keys())
    X_raw = np.column_stack([np.asarray(features[k], dtype=np.float64) for k in feature_names])
    y = np.asarray(target, dtype=np.float64)

    if X_raw.shape[0] != y.size:
        raise ValueError(
            f'target length {y.size} != features length {X_raw.shape[0]}'
        )

    mask = np.isfinite(np.column_stack([X_raw, y[:, None]])).all(axis=1)
    X_raw_clean, y_clean = X_raw[mask], y[mask]
    if len(y_clean) < 30:
        raise ValueError(f'too few clean rows ({len(y_clean)}) after NaN drop')

    # Polynomial expansion
    poly = PolynomialFeatures(degree=degree, include_bias=include_bias, interaction_only=False)
    X_poly = poly.fit_transform(X_raw_clean)
    term_names = list(poly.get_feature_names_out(feature_names))

    kf = KFold(n_splits=n_splits, shuffle=False)
    model = LassoCV(alphas=alpha_grid, cv=kf, max_iter=max_iter, n_jobs=1)
    model.fit(X_poly, y_clean)

    selected = {
        name: float(coef)
        for name, coef in zip(term_names, model.coef_)
        if abs(coef) > 1e-12
    }

    # OOF R² at chosen alpha
    oof = np.zeros_like(y_clean)
    for tr, te in kf.split(X_poly):
        m = Lasso(alpha=float(model.alpha_), max_iter=max_iter)
        m.fit(X_poly[tr], y_clean[tr])
        oof[te] = m.predict(X_poly[te])
    ss_res = float(np.sum((y_clean - oof) ** 2))
    ss_tot = float(np.sum((y_clean - y_clean.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return SymbolicScanResult(
        selected_terms=selected,
        alpha_selected=float(model.alpha_),
        alpha_grid=np.asarray(alpha_grid),
        r2=r2,
        n_terms_in_expansion=len(term_names),
        degree=degree,
        intercept=float(model.intercept_),
    )
