"""Tests for kuant._validation — the argument-validator helpers.

Coverage goals:
1. Each helper raises the *right* exception subclass.
2. Each error message contains: kernel name, argument name, the actual
   received value, the error code, and a `→ Fix:` line.
3. Valid inputs are silently accepted.
"""

from __future__ import annotations

import numpy as np
import pytest

from kuant import _validation as V
from kuant.errors import (
    KuantConvergenceError,
    KuantDependencyError,
    KuantShapeError,
    KuantValueError,
)


# ---------- helpers --------------------------------------------------------


def _assert_message_contract(msg: str, *, kernel: str, code: str, name: str | None = None):
    """Every error message must carry these markers so tracebacks are learnable."""
    assert f"kuant.{kernel}:" in msg, f"missing kernel prefix; got: {msg!r}"
    assert f"[{code}]" in msg, f"missing stable code; got: {msg!r}"
    assert "→ Fix:" in msg, f"missing fix line; got: {msg!r}"
    if name is not None:
        assert f"'{name}'" in msg, f"missing arg name; got: {msg!r}"


# ---------- shape validators -----------------------------------------------


def test_require_1d_accepts_1d():
    V.require_1d(np.arange(5), "x", kernel="rollmean")


def test_require_1d_rejects_2d():
    with pytest.raises(KuantShapeError) as exc:
        V.require_1d(np.zeros((10, 3)), "x", kernel="rollmean")
    _assert_message_contract(str(exc.value), kernel="rollmean", code="KE-SHAPE-1D", name="x")
    assert "(10, 3)" in str(exc.value)  # actual shape shown
    assert "ravel" in str(exc.value)  # concrete remedy


def test_require_1d_rejects_non_array():
    with pytest.raises(KuantShapeError):
        V.require_1d(42, "x", kernel="foo")


def test_require_2d_accepts_2d():
    V.require_2d(np.zeros((10, 3)), "X", kernel="fit")


def test_require_2d_rejects_1d():
    with pytest.raises(KuantShapeError) as exc:
        V.require_2d(np.zeros(10), "X", kernel="fit")
    _assert_message_contract(str(exc.value), kernel="fit", code="KE-SHAPE-2D", name="X")


def test_require_equal_length_accepts_match():
    V.require_equal_length(np.zeros(10), "x", np.zeros(10), "y", kernel="rollcorr")


def test_require_equal_length_rejects_mismatch():
    with pytest.raises(KuantShapeError) as exc:
        V.require_equal_length(np.zeros(500), "x", np.zeros(480), "y", kernel="rollcorr")
    m = str(exc.value)
    _assert_message_contract(m, kernel="rollcorr", code="KE-SHAPE-EQUAL-LEN", name="x")
    assert "'y'" in m
    assert "500" in m and "480" in m


def test_require_expected_shape_accepts_match():
    V.require_expected_shape(np.zeros((3, 3)), "A", (3, 3), kernel="hmm")


def test_require_expected_shape_accepts_wildcard():
    # Pass 'N' as a placeholder for a variable dim.
    V.require_expected_shape(np.zeros((3, 4)), "B", (3, "N"), kernel="hmm")


def test_require_expected_shape_rejects():
    with pytest.raises(KuantShapeError) as exc:
        V.require_expected_shape(np.zeros((3, 2)), "A", (3, 3), kernel="hmm")
    m = str(exc.value)
    _assert_message_contract(m, kernel="hmm", code="KE-SHAPE-EXPECTED", name="A")
    assert "(3, 3)" in m
    assert "(3, 2)" in m


# ---------- value-range validators -----------------------------------------


def test_require_positive_accepts():
    V.require_positive(1, "window", kernel="rollmean", kind="int")
    V.require_positive(0.5, "sigma", kernel="bscall")


def test_require_positive_rejects_zero():
    with pytest.raises(KuantValueError) as exc:
        V.require_positive(0, "window", kernel="rollmean", kind="int")
    _assert_message_contract(
        str(exc.value), kernel="rollmean", code="KE-VAL-POSITIVE", name="window"
    )


def test_require_positive_rejects_negative():
    with pytest.raises(KuantValueError) as exc:
        V.require_positive(-5, "sigma", kernel="bscall")
    assert "-5" in str(exc.value)


def test_require_positive_rejects_non_numeric():
    with pytest.raises(KuantValueError) as exc:
        V.require_positive("hi", "window", kernel="rollmean")
    assert "str" in str(exc.value)


def test_require_positive_int_rejects_fractional():
    with pytest.raises(KuantValueError) as exc:
        V.require_positive(3.5, "window", kernel="rollmean", kind="int")
    m = str(exc.value)
    _assert_message_contract(m, kernel="rollmean", code="KE-VAL-POSITIVE", name="window")
    assert "positive integer" in m


def test_require_nonnegative_accepts_zero():
    V.require_nonnegative(0, "ddof", kernel="rollstd", kind="int")


def test_require_nonnegative_rejects_negative():
    with pytest.raises(KuantValueError) as exc:
        V.require_nonnegative(-1, "ddof", kernel="rollstd", kind="int")
    _assert_message_contract(
        str(exc.value), kernel="rollstd", code="KE-VAL-NONNEGATIVE", name="ddof"
    )


def test_require_probability_accepts_boundary():
    V.require_probability(0.0, "p", kernel="normppf")
    V.require_probability(1.0, "p", kernel="normppf")
    V.require_probability(0.5, "p", kernel="normppf")


def test_require_probability_rejects_out_of_range():
    with pytest.raises(KuantValueError) as exc:
        V.require_probability(1.7, "p", kernel="normppf")
    m = str(exc.value)
    _assert_message_contract(m, kernel="normppf", code="KE-VAL-PROBABILITY", name="p")
    # The hint about percentages is the useful remedy.
    assert "percentage" in m or "percentile" in m


def test_require_range_accepts_inclusive():
    V.require_range(0.5, "alpha", kernel="ema", lo=0.0, hi=1.0)


def test_require_range_rejects_below():
    with pytest.raises(KuantValueError) as exc:
        V.require_range(-0.1, "alpha", kernel="ema", lo=0.0, hi=1.0)
    _assert_message_contract(str(exc.value), kernel="ema", code="KE-VAL-RANGE", name="alpha")


def test_require_range_exclusive_low():
    # alpha in (0, 1] — 0 must be rejected
    with pytest.raises(KuantValueError):
        V.require_range(0.0, "alpha", kernel="ema", lo=0.0, hi=1.0, lo_inclusive=False)
    # 0.001 must pass
    V.require_range(0.001, "alpha", kernel="ema", lo=0.0, hi=1.0, lo_inclusive=False)


def test_require_window_accepts():
    V.require_window(21, n=100, kernel="rollmean")


def test_require_window_rejects_zero():
    with pytest.raises(KuantValueError) as exc:
        V.require_window(0, n=100, kernel="rollmean")
    # Fires KE-VAL-POSITIVE first because that's the tighter check.
    assert "KE-VAL-POSITIVE" in str(exc.value)


def test_require_window_rejects_too_large():
    with pytest.raises(KuantValueError) as exc:
        V.require_window(500, n=100, kernel="rollmean")
    m = str(exc.value)
    _assert_message_contract(m, kernel="rollmean", code="KE-VAL-WINDOW", name="window")
    assert "500" in m and "100" in m


# ---------- NaN / finite validators ----------------------------------------


def test_require_nonnan_accepts_clean():
    V.require_nonnan(np.arange(10.0), "x", kernel="fit")


def test_require_nonnan_accepts_int_dtype():
    # Integer arrays cannot hold NaN — pass through silently.
    V.require_nonnan(np.arange(10), "x", kernel="fit")


def test_require_nonnan_rejects_single_nan():
    a = np.array([1.0, 2.0, np.nan, 4.0])
    with pytest.raises(KuantValueError) as exc:
        V.require_nonnan(a, "x", kernel="fit")
    m = str(exc.value)
    _assert_message_contract(m, kernel="fit", code="KE-VAL-NAN", name="x")
    assert "1 NaN" in m
    assert "index 2" in m  # location shown for debuggability


def test_require_nonnan_shows_multiple_indices():
    a = np.array([1.0, np.nan, 3.0, np.nan, np.nan])
    with pytest.raises(KuantValueError) as exc:
        V.require_nonnan(a, "x", kernel="fit")
    m = str(exc.value)
    assert "3 NaN" in m
    # Should list at least the first 3 indices.
    assert "1" in m and "3" in m and "4" in m


def test_require_finite_accepts_clean():
    V.require_finite(np.arange(10.0), "x", kernel="fit")


def test_require_finite_rejects_inf():
    a = np.array([1.0, 2.0, np.inf, 4.0])
    with pytest.raises(KuantValueError) as exc:
        V.require_finite(a, "x", kernel="fit")
    m = str(exc.value)
    _assert_message_contract(m, kernel="fit", code="KE-VAL-FINITE", name="x")
    assert "inf" in m


def test_require_finite_rejects_mixed():
    a = np.array([1.0, np.nan, np.inf, -np.inf])
    with pytest.raises(KuantValueError) as exc:
        V.require_finite(a, "x", kernel="fit")
    m = str(exc.value)
    # Message should call out both classes of badness.
    assert "1 NaN" in m
    assert "2 ±inf" in m


def test_require_min_clean_accepts_enough():
    V.require_min_clean(np.arange(50), "y", kernel="lasso", min_count=10)


def test_require_min_clean_accepts_int_size():
    V.require_min_clean(50, "y", kernel="lasso", min_count=10)


def test_require_min_clean_rejects_too_few():
    with pytest.raises(KuantValueError) as exc:
        V.require_min_clean(np.arange(5), "y", kernel="lasso", min_count=10)
    m = str(exc.value)
    _assert_message_contract(m, kernel="lasso", code="KE-VAL-MIN-CLEAN", name="y")
    assert "5" in m and "10" in m


def test_require_min_clean_purpose_shows_up():
    with pytest.raises(KuantValueError) as exc:
        V.require_min_clean(np.arange(5), "y", kernel="lasso", min_count=10, purpose="fit LASSO")
    assert "fit LASSO" in str(exc.value)


# ---------- dependency + convergence ---------------------------------------


def test_require_dep_raises():
    with pytest.raises(KuantDependencyError) as exc:
        V.require_dep("statsmodels", kernel="grangerscan", install="pip install statsmodels")
    m = str(exc.value)
    _assert_message_contract(m, kernel="grangerscan", code="KE-DEP-MISSING")
    assert "pip install statsmodels" in m


def test_require_dep_chains_cause():
    original = ImportError("no module named statsmodels")
    with pytest.raises(KuantDependencyError) as exc:
        V.require_dep(
            "statsmodels",
            kernel="grangerscan",
            install="pip install statsmodels",
            cause=original,
        )
    assert exc.value.__cause__ is original


def test_require_dep_is_import_error_too():
    # Users who catch `ImportError` blindly must still catch this.
    with pytest.raises(ImportError):
        V.require_dep("sklearn", kernel="pinnscan", install="pip install scikit-learn")


def test_did_not_converge_shape():
    with pytest.raises(KuantConvergenceError) as exc:
        V.did_not_converge(
            kernel="impvol",
            iters=50,
            tol=1e-8,
            last_err=1.4e-4,
            fallback="impvolbisection",
        )
    m = str(exc.value)
    _assert_message_contract(m, kernel="impvol", code="KE-CONV-MAX-ITER")
    assert "impvolbisection" in m
    assert "50" in m


def test_did_not_converge_no_fallback():
    with pytest.raises(KuantConvergenceError) as exc:
        V.did_not_converge(kernel="fit", iters=100, tol=1e-6, last_err=0.01)
    assert "fall back" not in str(exc.value)


# ---------- catchability contract ------------------------------------------


def test_all_helpers_raise_stdlib_compatible():
    """User code that catches ValueError / ImportError / RuntimeError must still work."""
    with pytest.raises(ValueError):
        V.require_1d(np.zeros((3, 3)), "x", kernel="k")
    with pytest.raises(ValueError):
        V.require_positive(-1, "w", kernel="k")
    with pytest.raises(ValueError):
        V.require_nonnan(np.array([1.0, np.nan]), "x", kernel="k")
    with pytest.raises(ValueError):
        V.require_finite(np.array([1.0, np.inf]), "x", kernel="k")
    with pytest.raises(ValueError):
        V.require_min_clean(3, "y", kernel="k", min_count=10)
    with pytest.raises(ImportError):
        V.require_dep("x", kernel="k", install="pip install x")
    with pytest.raises(RuntimeError):
        V.did_not_converge(kernel="k", iters=1, tol=1e-6, last_err=1.0)
