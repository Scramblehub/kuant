"""kuant._validation — argument validators that raise informative errors.

Every helper here follows one contract:

    kuant.<kernel>: <what went wrong, with actual values>.  [<code>]
      → Fix: <concrete remedy the user can copy-paste>.

The `[<code>]` suffix is a stable identifier (`KE-SHAPE-1D`, `KE-VAL-POSITIVE`,
…). It lets us later point at a docs page (`kuant.dev/errors/KE-SHAPE-1D`) with
a longer explanation, without breaking existing tracebacks or match strings.

Call sites should stay compact — that's the point of centralising here:

    from kuant._validation import require_1d, require_positive

    def rollmean(x, window):
        arr = np.asarray(x)
        require_1d(arr, "x", kernel="rollmean")
        require_positive(window, "window", kernel="rollmean", kind="int")
        ...

Underscore prefix on the module signals internal API. Users catch the classes
in `kuant.errors`; they never import these helpers themselves.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from kuant.errors import (
    KuantConvergenceError,
    KuantDependencyError,
    KuantShapeError,
    KuantValueError,
    KuantWarning,
)


def _msg(kernel: str, code: str, what: str, fix: str) -> str:
    """Build the standard two-line error message.

    Line 1 identifies the kernel, the failure, and the stable error code.
    Line 2 gives a concrete, copy-pasteable remedy.
    """
    return f"kuant.{kernel}: {what}.  [{code}]\n  → Fix: {fix}"


# ---------- shape validators ------------------------------------------------


def require_1d(arr: Any, name: str, *, kernel: str) -> None:
    """Reject non-1D arrays with a suggestion to flatten."""
    shape = getattr(arr, "shape", None)
    if shape is not None and len(shape) == 1:
        return
    got = f"shape {shape}" if shape is not None else f"type {type(arr).__name__}"
    raise KuantShapeError(
        _msg(
            kernel,
            "KE-SHAPE-1D",
            f"'{name}' must be a 1D array, got {got}",
            f"pass a flat array — e.g. `np.asarray({name}).ravel()` if "
            f"flattening is intended, or `{name}[:, 0]` to pick the first column",
        )
    )


def require_2d(arr: Any, name: str, *, kernel: str) -> None:
    """Reject non-2D arrays."""
    shape = getattr(arr, "shape", None)
    if shape is not None and len(shape) == 2:
        return
    raise KuantShapeError(
        _msg(
            kernel,
            "KE-SHAPE-2D",
            f"'{name}' must be 2D, got shape {shape}",
            f"reshape to (n_rows, n_cols) — e.g. `{name}.reshape(-1, k)` "
            f"if you know the column count k",
        )
    )


def require_equal_length(a: Any, name_a: str, b: Any, name_b: str, *, kernel: str) -> None:
    """Reject unequal-length inputs, telling the user which lengths clashed."""
    n_a = len(a)
    n_b = len(b)
    if n_a == n_b:
        return
    raise KuantShapeError(
        _msg(
            kernel,
            "KE-SHAPE-EQUAL-LEN",
            f"'{name_a}' and '{name_b}' must have equal length, got " f"{n_a} and {n_b}",
            "align the series before calling — drop NaNs together, reindex "
            "on a common index, or slice to the shared span",
        )
    )


def require_expected_shape(arr: Any, name: str, expected: tuple, *, kernel: str) -> None:
    """Reject arrays that don't match a specific expected shape.

    `expected` may contain string placeholders like `"N"` to denote a variable
    dimension already fixed by another argument — those are reported as-is in
    the message so the user sees the constraint.
    """
    shape = getattr(arr, "shape", None)
    if shape is None:
        got = "not an array"
    else:
        got = str(tuple(shape))
    # Compare against expected, treating strings as wildcards for display.
    concrete_expected = tuple(d if not isinstance(d, str) else None for d in expected)
    if shape is not None and len(shape) == len(expected):
        ok = all(e is None or e == a for e, a in zip(concrete_expected, shape))
        if ok:
            return
    raise KuantShapeError(
        _msg(
            kernel,
            "KE-SHAPE-EXPECTED",
            f"'{name}' must have shape {expected}, got {got}",
            "check that this argument matches the dimensions implied by the "
            "other inputs (state count, observation count, etc.)",
        )
    )


# ---------- value-range validators ------------------------------------------


def require_positive(value: Any, name: str, *, kernel: str, kind: str = "value") -> None:
    """Reject non-positive numbers. `kind` shapes the fix hint ('int' vs 'value')."""
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise KuantValueError(
            _msg(
                kernel,
                "KE-VAL-POSITIVE",
                f"'{name}' must be a positive number, got {value!r} of type "
                f"{type(value).__name__}",
                "pass a numeric value",
            )
        ) from exc
    if v > 0:
        if kind == "int" and int(v) != v:
            raise KuantValueError(
                _msg(
                    kernel,
                    "KE-VAL-POSITIVE",
                    f"'{name}' must be a positive integer, got {value}",
                    f"cast with `int({name})` if the fractional part is unintended",
                )
            )
        return
    kind_word = "integer" if kind == "int" else "value"
    raise KuantValueError(
        _msg(
            kernel,
            "KE-VAL-POSITIVE",
            f"'{name}' must be positive (> 0), got {value}",
            f"pass a positive {kind_word}",
        )
    )


def require_nonnegative(value: Any, name: str, *, kernel: str, kind: str = "value") -> None:
    """Reject negative numbers."""
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise KuantValueError(
            _msg(
                kernel,
                "KE-VAL-NONNEGATIVE",
                f"'{name}' must be a non-negative number, got {value!r} of "
                f"type {type(value).__name__}",
                "pass a numeric value",
            )
        ) from exc
    if v >= 0:
        if kind == "int" and int(v) != v:
            raise KuantValueError(
                _msg(
                    kernel,
                    "KE-VAL-NONNEGATIVE",
                    f"'{name}' must be a non-negative integer, got {value}",
                    f"cast with `int({name})` if the fractional part is unintended",
                )
            )
        return
    kind_word = "integer" if kind == "int" else "value"
    raise KuantValueError(
        _msg(
            kernel,
            "KE-VAL-NONNEGATIVE",
            f"'{name}' must be non-negative (>= 0), got {value}",
            f"pass a non-negative {kind_word}",
        )
    )


def require_probability(value: Any, name: str, *, kernel: str) -> None:
    """Reject values outside [0, 1]."""
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise KuantValueError(
            _msg(
                kernel,
                "KE-VAL-PROBABILITY",
                f"'{name}' must be in [0, 1], got {value!r} of type " f"{type(value).__name__}",
                "pass a probability",
            )
        ) from exc
    if 0.0 <= v <= 1.0:
        return
    raise KuantValueError(
        _msg(
            kernel,
            "KE-VAL-PROBABILITY",
            f"'{name}' must be in [0, 1], got {value}",
            f"this kernel expects a probability, not a percentage — "
            f"divide by 100 if '{name}' is a percentile like 95",
        )
    )


def require_range(
    value: Any,
    name: str,
    *,
    kernel: str,
    lo: float,
    hi: float,
    lo_inclusive: bool = True,
    hi_inclusive: bool = True,
) -> None:
    """Reject values outside [lo, hi] (bracket-controlled by inclusive flags)."""
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise KuantValueError(
            _msg(
                kernel,
                "KE-VAL-RANGE",
                f"'{name}' must be a number in the required range, got "
                f"{value!r} of type {type(value).__name__}",
                "pass a numeric value",
            )
        ) from exc
    lo_ok = v >= lo if lo_inclusive else v > lo
    hi_ok = v <= hi if hi_inclusive else v < hi
    if lo_ok and hi_ok:
        return
    lb = "[" if lo_inclusive else "("
    rb = "]" if hi_inclusive else ")"
    raise KuantValueError(
        _msg(
            kernel,
            "KE-VAL-RANGE",
            f"'{name}' must be in {lb}{lo}, {hi}{rb}, got {value}",
            f"clip or rescale '{name}' before calling",
        )
    )


def require_window(w: Any, n: int, *, kernel: str, name: str = "window") -> None:
    """Combined window validator: positive integer and fits inside the series."""
    require_positive(w, name, kernel=kernel, kind="int")
    if int(w) > n:
        raise KuantValueError(
            _msg(
                kernel,
                "KE-VAL-WINDOW",
                f"'{name}'={int(w)} exceeds input length {n}",
                f"use a shorter window, or provide at least {int(w)} " f"observations",
            )
        )


# ---------- stochastic constraint validators -------------------------------


def require_stochastic(vec: Any, name: str, *, kernel: str, atol: float = 1e-6) -> None:
    """Reject a vector that isn't a probability distribution.

    Enforces: (a) every entry in [0, 1] (within `atol`), and (b) the
    entries sum to 1 (within `atol`). Used for HMM initial-state π and
    other probability-vector inputs.
    """
    arr = _to_ndarray(vec)
    if arr.dtype.kind not in "fc":
        arr = arr.astype(np.float64)
    if arr.size == 0:
        raise KuantValueError(
            _msg(
                kernel,
                "KE-VAL-STOCHASTIC",
                f"'{name}' is empty; expected a probability distribution",
                "pass a non-empty 1D array of probabilities that sums to 1",
            )
        )
    a_min = float(arr.min())
    a_max = float(arr.max())
    if a_min < -atol or a_max > 1.0 + atol:
        bad_idx = int(np.argmin(arr)) if a_min < -atol else int(np.argmax(arr))
        raise KuantValueError(
            _msg(
                kernel,
                "KE-VAL-STOCHASTIC",
                f"'{name}' must lie in [0, 1] (probability distribution); "
                f"entry {bad_idx} = {float(arr.flat[bad_idx]):.6g}",
                f"clip to [0, 1] and renormalize — "
                f"`np.clip({name}, 0, 1) / np.clip({name}, 0, 1).sum()`",
            )
        )
    s = float(arr.sum())
    if abs(s - 1.0) > atol:
        raise KuantValueError(
            _msg(
                kernel,
                "KE-VAL-STOCHASTIC",
                f"'{name}' must sum to 1 (probability distribution), got sum={s:.6g}",
                f"renormalize before calling — `{name} = {name} / {name}.sum()`",
            )
        )


def require_stochastic_rows(mat: Any, name: str, *, kernel: str, atol: float = 1e-6) -> None:
    """Reject a 2D matrix whose rows aren't probability distributions.

    Used for HMM transition and emission matrices. Delegates the ndim
    check to `require_expected_shape` — callers should validate shape
    first, then hand the matrix to this helper.
    """
    arr = _to_ndarray(mat)
    if arr.dtype.kind not in "fc":
        arr = arr.astype(np.float64)
    if arr.ndim != 2:
        # Shape errors are the shape helper's job; assume caller checked.
        return
    a_min = float(arr.min())
    a_max = float(arr.max())
    if a_min < -atol or a_max > 1.0 + atol:
        raise KuantValueError(
            _msg(
                kernel,
                "KE-VAL-STOCHASTIC-ROWS",
                f"'{name}' contains values outside [0, 1] " f"(min={a_min:.3g}, max={a_max:.3g})",
                f"clip and renormalize each row — "
                f"`np.clip({name}, 0, 1) / np.clip({name}, 0, 1).sum(axis=1, keepdims=True)`",
            )
        )
    row_sums = arr.sum(axis=1)
    bad = np.where(np.abs(row_sums - 1.0) > atol)[0]
    if bad.size:
        r = int(bad[0])
        raise KuantValueError(
            _msg(
                kernel,
                "KE-VAL-STOCHASTIC-ROWS",
                f"'{name}' row {r} must sum to 1, got sum={float(row_sums[r]):.6g}",
                f"renormalize each row before calling — "
                f"`{name} = {name} / {name}.sum(axis=1, keepdims=True)`",
            )
        )


# ---------- mutex-pair validator -------------------------------------------


def require_mutex_pair(
    a: Any,
    name_a: str,
    b: Any,
    name_b: str,
    *,
    kernel: str,
    a_example: str,
    b_example: str,
) -> None:
    """Reject when neither or both of a mutually-exclusive arg pair are set.

    Use for XOR constraints on optional args: (span XOR alpha) in the EMA
    kernels; (n_states XOR full-init) in Baum-Welch. Both cases send
    exactly one of the two to `None`; passing both or neither is a bug.

    `a_example` and `b_example` are appended to the fix line so users
    see the two valid forms.
    """
    a_set = a is not None
    b_set = b is not None
    if a_set ^ b_set:
        return
    got = "both" if a_set and b_set else "neither"
    raise KuantValueError(
        _msg(
            kernel,
            "KE-VAL-MUTEX",
            f"provide exactly one of `{name_a}` or `{name_b}`, got {got}",
            f"`{a_example}` OR `{b_example}`",
        )
    )


# ---------- warnings -------------------------------------------------------


def warn_kuant(
    *,
    kernel: str,
    code: str,
    what: str,
    fix: str,
    category: type = KuantWarning,
    stacklevel: int = 3,
) -> None:
    """Emit a KuantWarning with the standard two-line message shape.

    Errors are for "we can't continue". Warnings are for "we returned
    something, but you should know it may be unreliable". `stacklevel=3`
    aims the warning at the kernel's caller, not the kernel itself.

    Users can promote any warning to an exception with:
        import warnings, kuant.errors
        warnings.filterwarnings("error", category=kuant.errors.KuantWarning)
    """
    import warnings as _warnings

    _warnings.warn(_msg(kernel, code, what, fix), category, stacklevel=stacklevel)


# ---------- NaN / finite validators ----------------------------------------


def _to_ndarray(arr: Any):
    """Best-effort conversion for validation. Handles numpy, cupy, lists.

    We only need this for scanning for NaN/inf; we don't return the array,
    so the caller keeps its original object. Cupy inputs are moved to host
    for the scan — cheap relative to the compute the kernel will do next.
    """
    if hasattr(arr, "get") and hasattr(arr, "shape"):  # cupy ndarray
        try:
            return arr.get()
        except Exception:
            pass
    return np.asarray(arr)


def require_nonnan(arr: Any, name: str, *, kernel: str) -> None:
    """Reject any NaN. Strict — for kernels that cannot handle NaN at all."""
    a = _to_ndarray(arr)
    if a.dtype.kind not in "fc":
        return  # non-float dtype can't hold NaN
    mask = np.isnan(a)
    if not mask.any():
        return
    n_nan = int(mask.sum())
    n_total = a.size
    # Show up to 3 leading NaN positions so users can see where they are.
    idx = np.flatnonzero(mask.ravel())[:3].tolist()
    where = f"first at index {idx[0]}" if len(idx) == 1 else f"first at indices {idx}"
    raise KuantValueError(
        _msg(
            kernel,
            "KE-VAL-NAN",
            f"'{name}' contains {n_nan} NaN of {n_total} values ({where})",
            f"drop NaN before calling — e.g. `{name}[~np.isnan({name})]` — "
            f"or interpolate them if the sequence structure matters",
        )
    )


def require_finite(arr: Any, name: str, *, kernel: str) -> None:
    """Reject any NaN or ±inf. For kernels that would produce garbage or overflow."""
    a = _to_ndarray(arr)
    if a.dtype.kind not in "fc":
        return
    mask = ~np.isfinite(a)
    if not mask.any():
        return
    n_bad = int(mask.sum())
    n_total = a.size
    n_nan = int(np.isnan(a).sum())
    n_inf = n_bad - n_nan
    parts = []
    if n_nan:
        parts.append(f"{n_nan} NaN")
    if n_inf:
        parts.append(f"{n_inf} ±inf")
    detail = " + ".join(parts)
    raise KuantValueError(
        _msg(
            kernel,
            "KE-VAL-FINITE",
            f"'{name}' contains non-finite values: {detail} of {n_total}",
            f"clean the input — drop or interpolate NaN, clip ±inf. "
            f"E.g. `{name} = {name}[np.isfinite({name})]`",
        )
    )


def require_min_clean(
    arr_or_size: Any, name: str, *, kernel: str, min_count: int, purpose: str = "fit"
) -> None:
    """Reject when too few finite values remain after the caller drops NaN.

    Pass either an already-cleaned array (or its size) as `arr_or_size`.
    `purpose` shapes the message ('fit', 'estimate', 'regression', …).
    """
    if hasattr(arr_or_size, "size"):
        n = int(arr_or_size.size)
    elif hasattr(arr_or_size, "__len__"):
        n = len(arr_or_size)
    else:
        n = int(arr_or_size)
    if n >= min_count:
        return
    raise KuantValueError(
        _msg(
            kernel,
            "KE-VAL-MIN-CLEAN",
            f"only {n} clean (finite) row(s) available in '{name}' after "
            f"NaN drop; need at least {min_count} to {purpose}",
            "provide more data, or align inputs so fewer rows are lost to "
            "NaN — check per-column NaN counts with "
            "`np.isnan(x).sum(axis=0)`",
        )
    )


# ---------- dependency + convergence ---------------------------------------


def require_dep(module: str, *, kernel: str, install: str, cause: Exception | None = None) -> None:
    """Raise `KuantDependencyError` for a missing optional dep.

    Call this from an `except ImportError:` inside a lazy import block. Pass
    the original exception via `cause` so `__cause__` gives users the real
    ImportError under the informative wrapper.
    """
    exc = KuantDependencyError(
        _msg(
            kernel,
            "KE-DEP-MISSING",
            f"requires '{module}', which is not installed",
            f"install with: `{install}`",
        )
    )
    if cause is not None:
        raise exc from cause
    raise exc


def did_not_converge(
    *,
    kernel: str,
    iters: int,
    tol: float,
    last_err: float,
    fallback: str | None = None,
    extra: str | None = None,
) -> None:
    """Raise `KuantConvergenceError` with iterate count, tolerance, and a fallback hint."""
    fix_parts = [
        f"raise `max_iter` (currently {iters})",
        f"loosen `tol` (currently {tol:g})",
    ]
    if fallback:
        fix_parts.append(f"or fall back to `{fallback}` which is slower but more robust")
    fix = ", ".join(fix_parts)
    what = (
        f"solver did not converge after {iters} iterations "
        f"(last max residual {last_err:g}, tolerance {tol:g})"
    )
    if extra:
        what = f"{what} — {extra}"
    raise KuantConvergenceError(_msg(kernel, "KE-CONV-MAX-ITER", what, fix))


__all__ = [
    "require_1d",
    "require_2d",
    "require_equal_length",
    "require_expected_shape",
    "require_positive",
    "require_nonnegative",
    "require_probability",
    "require_range",
    "require_window",
    "require_stochastic",
    "require_stochastic_rows",
    "require_mutex_pair",
    "require_nonnan",
    "require_finite",
    "require_min_clean",
    "require_dep",
    "did_not_converge",
    "warn_kuant",
]
