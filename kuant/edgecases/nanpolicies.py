"""Callable NaN-handling policies.

Every downstream kernel and pipeline has to decide what to do about
NaN in the input. There are five reasonable answers and users
re-implement all of them locally. `nanpolicies` ships them once as
composable callables so a pipeline can be configured with a policy
string or a policy callable:

    from kuant.edgecases import nanpolicies

    # Direct call
    y = nanpolicies.forwardfill(x)

    # Or looked up by name (config-driven code)
    policy = nanpolicies.get('forwardfill')
    y = policy(x)

The five policies:

- **`strict`** — raise `KuantValueError` on any NaN. Use when downstream
  math requires clean input and silent NaN propagation would hide bugs.
- **`skipna`** — return only the finite entries. 1D input drops NaN;
  2D drops rows where ANY column is NaN (all-or-nothing per row).
- **`forwardfill`** — replace NaN with the last-seen finite value on
  axis 0. Leading NaN (before the first finite value) stays NaN.
- **`interpolate`** — linear interpolation between the two nearest
  finite values. Leading and trailing NaN stay NaN. On 2D input,
  interpolates per column.
- **`dropcolumn`** — for 2D input only: drop columns whose finite-value
  fraction falls below `min_finite_frac` (default 0.5). Returns
  `(values_kept, col_mask)` so the caller can align a parallel
  column index.

Design: docs/kernels/edgecases/nanpolicies.md.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from kuant._validation import require_probability
from kuant.errors import KuantShapeError, KuantValueError


def strict(x):
    """Reject any NaN. Returns `x` unchanged if clean; raises otherwise."""
    arr = np.asarray(x)
    if arr.dtype.kind not in "fc":
        return arr
    if bool(np.isnan(arr).any()):
        n_nan = int(np.isnan(arr).sum())
        n_total = int(arr.size)
        raise KuantValueError(
            f"kuant.edgecases.nanpolicies.strict: input contains "
            f"{n_nan} NaN of {n_total} values.  [KE-VAL-NAN]\n"
            f"  → Fix: pick a different policy (skipna, forwardfill, "
            f"interpolate) or clean the input upstream"
        )
    return arr


def skipna(x):
    """Return only finite entries. Rows in 2D input are all-or-nothing."""
    arr = np.asarray(x)
    if arr.dtype.kind not in "fc":
        return arr
    if arr.ndim == 1:
        return arr[np.isfinite(arr)]
    if arr.ndim == 2:
        row_mask = np.isfinite(arr).all(axis=1)
        return arr[row_mask]
    raise KuantShapeError(
        f"kuant.edgecases.nanpolicies.skipna: input must be 1D or 2D, "
        f"got shape {arr.shape}.  [KE-SHAPE-EXPECTED]\n"
        f"  → Fix: pass a 1D series or 2D panel"
    )


def _forward_fill_1d(arr: np.ndarray) -> np.ndarray:
    """Vectorized forward-fill on a 1D array; leading NaN preserved."""
    mask = np.isfinite(arr)
    if not bool(mask.any()):
        return arr.copy()
    idx = np.where(mask, np.arange(arr.size), 0)
    idx = np.maximum.accumulate(idx)
    out = arr[idx]
    first_true = int(np.argmax(mask))
    out[:first_true] = np.nan
    return out


def forwardfill(x):
    """Replace NaN with the last-seen finite value on axis 0."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        return _forward_fill_1d(arr)
    if arr.ndim == 2:
        out = arr.copy()
        for c in range(arr.shape[1]):
            out[:, c] = _forward_fill_1d(arr[:, c])
        return out
    raise KuantShapeError(
        f"kuant.edgecases.nanpolicies.forwardfill: input must be 1D or "
        f"2D, got shape {arr.shape}.  [KE-SHAPE-EXPECTED]\n"
        f"  → Fix: pass a 1D series or 2D panel"
    )


def _interpolate_1d(arr: np.ndarray) -> np.ndarray:
    """Linear interpolation between the nearest finite values on a 1D array.

    Leading and trailing NaN stay NaN. If there is only one finite value,
    the array is returned unchanged (no way to interpolate direction).
    """
    mask = np.isfinite(arr)
    n_finite = int(mask.sum())
    if n_finite < 2:
        return arr.copy()
    xp = np.flatnonzero(mask)
    fp = arr[mask]
    x_all = np.arange(arr.size)
    interp = np.interp(x_all, xp, fp)
    # Preserve leading/trailing NaN.
    first, last = int(xp[0]), int(xp[-1])
    interp[:first] = np.nan
    interp[last + 1 :] = np.nan
    return interp


def interpolate(x):
    """Linear interpolation between the nearest finite values."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        return _interpolate_1d(arr)
    if arr.ndim == 2:
        out = arr.copy()
        for c in range(arr.shape[1]):
            out[:, c] = _interpolate_1d(arr[:, c])
        return out
    raise KuantShapeError(
        f"kuant.edgecases.nanpolicies.interpolate: input must be 1D or "
        f"2D, got shape {arr.shape}.  [KE-SHAPE-EXPECTED]\n"
        f"  → Fix: pass a 1D series or 2D panel"
    )


def dropcolumn(x, min_finite_frac: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """For 2D input, drop columns with too many NaN.

    Parameters
    ----------
    x : 2D array
        Panel with columns being different names / features.
    min_finite_frac : float in [0, 1], default 0.5
        Keep a column iff its finite-value fraction is `>= min_finite_frac`.

    Returns
    -------
    values_kept : 2D array
        Panel restricted to surviving columns.
    col_mask : 1D bool array of length `n_cols_in`
        True where the column was kept. Use this to slice a parallel
        column index (e.g. `names[col_mask]`).
    """
    arr = np.asarray(x, dtype=np.float64)
    require_probability(min_finite_frac, "min_finite_frac", kernel="dropcolumn")
    if arr.ndim != 2:
        raise KuantShapeError(
            f"kuant.edgecases.nanpolicies.dropcolumn: input must be 2D "
            f"(panel), got shape {arr.shape}.  [KE-SHAPE-EXPECTED]\n"
            f"  → Fix: pass a 2D panel of shape (n_rows, n_cols); use "
            f"`skipna` for 1D input"
        )
    finite_frac = np.isfinite(arr).mean(axis=0)
    col_mask = finite_frac >= min_finite_frac
    return arr[:, col_mask], col_mask


# ---------- registry ------------------------------------------------------


_POLICIES: dict[str, Callable] = {
    "strict": strict,
    "skipna": skipna,
    "forwardfill": forwardfill,
    "interpolate": interpolate,
    "dropcolumn": dropcolumn,
}


def get(name: str) -> Callable:
    """Look up a policy by name.

    Useful for config-driven code where the policy is chosen at runtime:

        policy = nanpolicies.get(config['nan_policy'])
        y = policy(x)
    """
    if name not in _POLICIES:
        raise KuantValueError(
            f"kuant.edgecases.nanpolicies.get: unknown policy {name!r}; "
            f"known policies are {tuple(_POLICIES.keys())}.  "
            f"[KE-VAL-RANGE]\n"
            f"  → Fix: pick one of {tuple(_POLICIES.keys())}"
        )
    return _POLICIES[name]


def available() -> tuple[str, ...]:
    """Return the sorted tuple of policy names."""
    return tuple(sorted(_POLICIES.keys()))


__all__ = [
    "strict",
    "skipna",
    "forwardfill",
    "interpolate",
    "dropcolumn",
    "get",
    "available",
]
