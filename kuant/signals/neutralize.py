"""OLS residual after regressing a signal on factors.

Given a signal series and one or more factor exposure series (industry,
size, market beta, style factors), fit the OLS regression:

    signal_t = alpha + Σ_k beta_k · factor_k(t) + residual_t

and return the residual. That's the "factor-neutralized signal": the
part of the signal that isn't explained by the factors.

Use to strip incidental exposures out of an alpha signal, so a
long-short portfolio built from the residual doesn't accidentally
tilt into value/size/momentum just because the raw signal happened
to correlate with them.

Design: docs/kernels/signals/neutralize.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kuant._validation import require_1d, require_equal_length, warn_kuant
from kuant.errors import KuantNumericWarning, KuantShapeError, KuantValueError


@dataclass
class NeutralizeResult:
    """Residuals + fit metadata.

    Attributes
    ----------
    residuals : 1D np.ndarray, length T
        Signal minus the fitted linear combination of factors.
        NaN where the source signal or any factor was NaN.
    betas : dict[str, float]
        Fitted coefficient per factor (plus 'intercept' if added).
    r2 : float
        Fraction of signal variance explained by the factors, on the
        rows that survived the NaN drop.
    condition_number : float
        Condition number of `X^T X`. > 1e10 triggers a KuantNumericWarning
        (near-collinear factors → betas unstable).
    n_used : int
        Number of rows the regression fit on (after NaN drop).
    """

    residuals: np.ndarray
    betas: dict
    r2: float
    condition_number: float
    n_used: int

    def summary(self) -> str:
        parts = [
            "=== NeutralizeResult ===",
            f"n_used:              {self.n_used}",
            f"R² explained:        {self.r2:.4f}",
            f"condition number:    {self.condition_number:.2e}",
            f"betas:               {len(self.betas)} coefficients",
        ]
        for name, b in self.betas.items():
            parts.append(f"  {name:<20s} {b:+.6f}")
        return "\n".join(parts)


def neutralize(
    signal,
    factors,
    add_intercept: bool = True,
) -> NeutralizeResult:
    """Regress `signal` on `factors`, return the residual series.

    Parameters
    ----------
    signal : 1D array, length T
        The raw alpha signal.
    factors : 2D array (T, K), dict[str, 1D array], or list of 1D arrays
        Factor exposure series. dict form gives named coefficients in
        the result; 2D array yields names `factor0`, `factor1`, ...;
        list yields `factor0`, ... in the order given.
    add_intercept : bool, default True
        Adds a column of ones to the design matrix. Setting False
        forces the regression through zero — appropriate only if you
        know the signal is already mean-centred.

    Returns
    -------
    NeutralizeResult

    Warnings
    --------
    - `KuantNumericWarning` (`KW-COLLINEAR-FACTORS`) if the design
      matrix's condition number exceeds 1e10 — the betas are
      numerically unstable and the residuals should not be trusted.

    Notes
    -----
    - Rows where signal or any factor is NaN are DROPPED from the fit;
      the corresponding output residual is NaN. That's the "PIT-clean"
      choice — factor exposures at a bar with no signal don't
      participate.
    - Uses `np.linalg.lstsq` for numerical robustness.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> factor = rng.standard_normal(500)
    >>> signal = 0.5 * factor + rng.standard_normal(500) * 0.1
    >>> r = neutralize(signal, {'factor': factor})
    >>> abs(r.betas['factor'] - 0.5) < 0.05                # β near truth
    True
    >>> r.r2 > 0.9                                          # signal well explained
    True
    """
    signal_arr = np.asarray(signal, dtype=np.float64)
    require_1d(signal_arr, "signal", kernel="neutralize")
    T = signal_arr.size

    # Normalize `factors` to (T, K) plus a name list.
    if isinstance(factors, dict):
        factor_names = list(factors.keys())
        cols = [np.asarray(factors[k], dtype=np.float64) for k in factor_names]
    elif isinstance(factors, np.ndarray):
        if factors.ndim == 1:
            cols = [factors.astype(np.float64)]
            factor_names = ["factor0"]
        elif factors.ndim == 2:
            factor_names = [f"factor{i}" for i in range(factors.shape[1])]
            cols = [factors[:, i].astype(np.float64) for i in range(factors.shape[1])]
        else:
            raise KuantShapeError(
                f"kuant.neutralize: 'factors' array must be 1D or 2D, got "
                f"shape {factors.shape}.  [KE-SHAPE-EXPECTED]\n"
                f"  → Fix: pass a (T, K) design matrix or a dict of 1D series"
            )
    elif isinstance(factors, (list, tuple)):
        cols = [np.asarray(c, dtype=np.float64) for c in factors]
        factor_names = [f"factor{i}" for i in range(len(cols))]
    else:
        raise KuantValueError(
            f"kuant.neutralize: 'factors' must be a 2D array, dict of 1D "
            f"arrays, or list of 1D arrays; got {type(factors).__name__}.  "
            f"[KE-SHAPE-EXPECTED]\n"
            f"  → Fix: pass one of the three supported forms"
        )

    if not cols:
        raise KuantValueError(
            "kuant.neutralize: 'factors' has no columns; no regression to "
            "run.  [KE-VAL-RANGE]\n"
            "  → Fix: pass at least one factor"
        )
    for i, c in enumerate(cols):
        require_1d(c, f"factors[{factor_names[i]}]", kernel="neutralize")
        require_equal_length(
            signal_arr,
            "signal",
            c,
            f"factors[{factor_names[i]}]",
            kernel="neutralize",
        )

    # Design matrix.
    X = np.column_stack(cols)
    if add_intercept:
        X = np.column_stack([np.ones(T), X])
        design_names = ["intercept"] + factor_names
    else:
        design_names = list(factor_names)

    # Drop rows with any NaN in signal or design.
    row_mask = np.isfinite(signal_arr) & np.isfinite(X).all(axis=1)
    n_used = int(row_mask.sum())
    if n_used < X.shape[1]:
        raise KuantValueError(
            f"kuant.neutralize: only {n_used} clean rows after NaN drop, "
            f"but design has {X.shape[1]} parameters; regression is "
            f"underdetermined.  [KE-VAL-UNDERDET]\n"
            f"  → Fix: provide more data or reduce the number of factors"
        )

    X_clean = X[row_mask]
    y_clean = signal_arr[row_mask]

    # Condition number of X^T X (equivalently, squared cond of X).
    _, s, _ = np.linalg.svd(X_clean, full_matrices=False)
    if s.min() == 0.0:
        cond = float("inf")
    else:
        cond = float((s.max() / s.min()) ** 2)

    if cond > 1e10:
        warn_kuant(
            kernel="neutralize",
            code="KW-COLLINEAR-FACTORS",
            what=(
                f"design matrix condition number {cond:.2e} > 1e10; "
                f"factors are near-collinear, betas are numerically unstable"
            ),
            fix=(
                "drop or orthogonalize redundant factors; check for a "
                "constant-plus-intercept combination (e.g. a factor that "
                "is 1 - another factor)"
            ),
            category=KuantNumericWarning,
        )

    # OLS via lstsq (SVD-based, robust to near-singular X).
    beta, *_ = np.linalg.lstsq(X_clean, y_clean, rcond=None)

    fitted = X_clean @ beta
    ss_res = float(np.sum((y_clean - fitted) ** 2))
    ss_tot = float(np.sum((y_clean - y_clean.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Reinsert residuals into the T-length output with NaN where dropped.
    residuals = np.full(T, np.nan, dtype=np.float64)
    residuals[row_mask] = y_clean - fitted

    return NeutralizeResult(
        residuals=residuals,
        betas={n: float(b) for n, b in zip(design_names, beta)},
        r2=r2,
        condition_number=cond,
        n_used=n_used,
    )


__all__ = ["neutralize", "NeutralizeResult"]
