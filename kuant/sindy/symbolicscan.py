"""Polynomial-symbolic regression scan.

Motivation. Between LASSO on raw features (`sindylasso`) and
GradientBoosting on the same library (`pinnscan`), there's a gap:
compact symbolic non-linear relationships. Squared terms and pairwise
interactions are the sweet spot — often physically motivated
("gamma effect", "convexity term") and cheap to fit.

This tool builds a polynomial feature expansion (default degree 2:
squares + pairwise interactions), then runs LASSO with CV on the
expanded library. Non-zero coefficients give you a compact
polynomial equation.

Typical null result on daily-frequency financial data: polynomial
degree-2 expansion of a candidate library, then LASSO, yields no
polynomial improvement over the pure linear scan. The mechanism is
the same as sindylasso, just with an interaction basis.

Design: docs/kernels/sindy/symbolicscan.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from kuant._validation import (
    require_dep,
    require_equal_length,
    require_min_clean,
    require_positive,
    require_range,
    warn_kuant,
)
from kuant.errors import KuantNumericWarning, KuantValueError


@dataclass
class SymbolicScanResult:
    selected_terms: dict[str, float]  # symbolic term string -> coefficient
    alpha_selected: float
    alpha_grid: np.ndarray
    r2: float
    n_terms_in_expansion: int
    degree: int
    intercept: float

    def summary(self) -> str:
        lines = [
            "=== Symbolic (polynomial) regression scan ===",
            f"Polynomial degree:        {self.degree}",
            f"Expanded library size:    {self.n_terms_in_expansion}",
            f"Selected (non-zero):      {len(self.selected_terms)}",
            f"Alpha chosen (by CV):     {self.alpha_selected:.4g}",
            f"OOF R²:                   {self.r2:.4f}",
            f"Intercept:                {self.intercept:+.4f}",
        ]

        if self.selected_terms:
            lines.append("")
            lines.append("Selected polynomial equation (largest |coef| first):")
            ordered = sorted(self.selected_terms.items(), key=lambda kv: -abs(kv[1]))
            eqn_terms = [f"{self.intercept:+.4f}"]
            for term, coef in ordered:
                sign = "+" if coef >= 0 else "-"
                eqn_terms.append(f"{sign} {abs(coef):.4f}·{term}")
                lines.append(f"  {term:<30s} {coef:+.4f}")
            lines.append("")
            lines.append("  y ≈ " + " ".join(eqn_terms))
        else:
            lines.append("")
            lines.append("DIAGNOSTIC: LASSO selected zero polynomial terms. No compact")
            lines.append("symbolic structure detected. Try pinnscan for a full nonlinear")
            lines.append("search, or accept the null.")
        return "\n".join(lines)


def _require_sklearn():
    try:
        from sklearn.linear_model import Lasso, LassoCV
        from sklearn.model_selection import KFold
        from sklearn.preprocessing import PolynomialFeatures

        return Lasso, LassoCV, KFold, PolynomialFeatures
    except ImportError as e:
        require_dep(
            "scikit-learn",
            kernel="symbolicscan",
            install="pip install scikit-learn",
            cause=e,
        )


def symbolicscan(
    target: np.ndarray,
    features: dict[str, np.ndarray],
    degree: int = 2,
    alpha_grid: Optional[np.ndarray] = None,
    n_splits: int = 5,
    max_iter: int = 10000,
    include_bias: bool = False,
) -> SymbolicScanResult:
    """Polynomial-symbolic regression scan via LASSO on a polynomial expansion.

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
    """
    Lasso, LassoCV, KFold, PolynomialFeatures = _require_sklearn()

    require_positive(degree, "degree", kernel="symbolicscan", kind="int")
    require_range(n_splits, "n_splits", kernel="symbolicscan", lo=2, hi=float("inf"))
    require_positive(max_iter, "max_iter", kernel="symbolicscan", kind="int")

    if alpha_grid is None:
        alpha_grid = np.logspace(-5, -1, 30)

    feature_names = list(features.keys())
    X_raw = np.column_stack([np.asarray(features[k], dtype=np.float64) for k in feature_names])
    y = np.asarray(target, dtype=np.float64)

    require_equal_length(X_raw, "features", y, "target", kernel="symbolicscan")

    mask = np.isfinite(np.column_stack([X_raw, y[:, None]])).all(axis=1)
    X_raw_clean, y_clean = X_raw[mask], y[mask]
    require_min_clean(
        y_clean,
        "target",
        kernel="symbolicscan",
        min_count=30,
        purpose="fit polynomial LASSO",
    )

    # Polynomial expansion
    poly = PolynomialFeatures(degree=degree, include_bias=include_bias, interaction_only=False)
    X_poly = poly.fit_transform(X_raw_clean)
    term_names = list(poly.get_feature_names_out(feature_names))

    # A5 — n < 2·p check on the EXPANDED library (polynomial widens the
    # design matrix; degree=2 on 10 features gives 65 columns).
    n_samp = X_poly.shape[0]
    n_expanded = X_poly.shape[1]
    if n_samp < 2 * n_expanded:
        raise KuantValueError(
            f"kuant.symbolicscan: only {n_samp} clean samples for "
            f"{n_expanded} polynomial terms (degree={degree} on "
            f"{len(feature_names)} base features); LASSO CV is unreliable "
            f"when n_samples < 2·n_expanded_terms.  [KE-VAL-UNDERDET]\n"
            f"  → Fix: raise n_samples above {2 * n_expanded}, lower "
            f"`degree`, or drop base features"
        )

    kf = KFold(n_splits=n_splits, shuffle=False)
    model = LassoCV(alphas=alpha_grid, cv=kf, max_iter=max_iter, n_jobs=1)
    model.fit(X_poly, y_clean)

    selected = {
        name: float(coef) for name, coef in zip(term_names, model.coef_) if abs(coef) > 1e-12
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

    # B2 — CV endpoint warnings
    a_lo, a_hi = float(alpha_grid.min()), float(alpha_grid.max())
    a_sel = float(model.alpha_)
    if abs(a_sel - a_lo) < 1e-12:
        warn_kuant(
            kernel="symbolicscan",
            code="KW-CV-ENDPOINT-LOW",
            what=(
                f"CV picked the weakest regularization α={a_sel:g} at the "
                f"bottom of the grid [{a_lo:g}, {a_hi:g}]"
            ),
            fix=(
                "expand alpha_grid downward — the true optimum may be smaller "
                "and selected polynomial terms may be overfit"
            ),
            category=KuantNumericWarning,
        )
    elif abs(a_sel - a_hi) < 1e-12:
        warn_kuant(
            kernel="symbolicscan",
            code="KW-CV-ENDPOINT-HIGH",
            what=(
                f"CV picked the strongest regularization α={a_sel:g} at the "
                f"top of the grid; {len(selected)} term(s) selected"
            ),
            fix=(
                "if 0 terms selected this is a clean null result; otherwise "
                "expand alpha_grid upward to confirm the plateau"
            ),
            category=KuantNumericWarning,
        )

    return SymbolicScanResult(
        selected_terms=selected,
        alpha_selected=float(model.alpha_),
        alpha_grid=np.asarray(alpha_grid),
        r2=r2,
        n_terms_in_expansion=len(term_names),
        degree=degree,
        intercept=float(model.intercept_),
    )
