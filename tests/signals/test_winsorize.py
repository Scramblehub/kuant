"""Tests for kuant.signals.winsorize."""

from __future__ import annotations

import numpy as np
import pytest

from kuant.errors import KuantShapeError, KuantValueError
from kuant.signals.winsorize import winsorize


# ---------- 1D behavior ---------------------------------------------------


def test_1d_clips_top_and_bottom():
    x = np.arange(1.0, 101)  # 1..100
    r = winsorize(x, lo=0.05, hi=0.95)
    assert r.min() >= np.quantile(x, 0.05) - 1e-9
    assert r.max() <= np.quantile(x, 0.95) + 1e-9


def test_1d_extreme_value_clipped():
    x = np.array([1.0, 2, 3, 4, 5, 6, 7, 8, 9, 100])
    r = winsorize(x, lo=0.0, hi=0.9)
    assert r[-1] < 100.0  # 100 clipped down
    assert r[:-1].tolist() == x[:-1].tolist()  # others unchanged


def test_1d_preserves_shape_and_length():
    x = np.arange(50.0)
    r = winsorize(x)
    assert r.shape == x.shape


def test_1d_no_op_at_full_range():
    """(lo=0, hi=1) should be a no-op on any 1D input."""
    x = np.array([1.0, 2, 3, 4, 5])
    r = winsorize(x, lo=0.0, hi=1.0)
    assert r.tolist() == x.tolist()


# ---------- 2D behavior — per_row (cross-sectional) ----------------------


def test_2d_per_row_clips_each_row_independently():
    """Each row's outlier is clipped without affecting other rows."""
    # Row 0: extreme high; row 1: extreme low.
    x = np.array(
        [
            [1.0, 2, 3, 4, 100],
            [-100.0, 1, 2, 3, 4],
        ]
    )
    r = winsorize(x, lo=0.1, hi=0.9, per_row=True)
    assert r[0, -1] < 100.0
    assert r[1, 0] > -100.0
    # Middle entries unchanged.
    assert r[0, 2] == 3.0
    assert r[1, 2] == 2.0


def test_2d_per_row_shape_preserved():
    x = np.arange(20.0).reshape(4, 5)
    r = winsorize(x, per_row=True)
    assert r.shape == x.shape


# ---------- 2D behavior — per_column (time-series) -----------------------


def test_2d_per_column_clips_each_column_independently():
    """Each column's tail is clipped against its own history."""
    rng = np.random.default_rng(0)
    normal = rng.standard_normal((100, 3))
    # Add one huge outlier to column 1.
    normal[50, 1] = 100.0
    r = winsorize(normal, lo=0.01, hi=0.99, per_row=False)
    assert r[50, 1] < 100.0
    # Columns 0 and 2 untouched by the column-1 outlier.
    assert abs(r[:, 0] - normal[:, 0]).max() < 5.0


# ---------- NaN handling -------------------------------------------------


def test_nan_preserved_in_output():
    x = np.array([1.0, np.nan, 3, 4, 100])
    r = winsorize(x, lo=0.0, hi=0.9)
    assert np.isnan(r[1])


def test_nan_excluded_from_quantile_computation():
    """The 99th percentile of [1, 2, 3, nan, 100] should ignore the NaN
    but still 'see' the 100."""
    x = np.array([1.0, 2, 3, np.nan, 100])
    r = winsorize(x, lo=0.0, hi=0.75)
    # The 75th percentile of finite [1, 2, 3, 100] is 27.25 → 100 clipped.
    assert r[-1] < 100.0
    assert np.isnan(r[3])


def test_all_nan_row_untouched():
    x = np.array([np.nan, np.nan, np.nan])
    r = winsorize(x)
    assert np.isnan(r).all()


# ---------- error contract ------------------------------------------------


def test_reject_lo_ge_hi():
    with pytest.raises(KuantValueError) as exc:
        winsorize(np.arange(10.0), lo=0.9, hi=0.1)
    assert "lo" in str(exc.value) and "hi" in str(exc.value)


def test_reject_lo_out_of_range():
    with pytest.raises(KuantValueError):
        winsorize(np.arange(10.0), lo=1.5, hi=0.99)


def test_reject_3d_input():
    with pytest.raises(KuantShapeError):
        winsorize(np.zeros((3, 4, 5)))


def test_reject_lo_and_hi_equal():
    with pytest.raises(KuantValueError):
        winsorize(np.arange(10.0), lo=0.5, hi=0.5)


# ---------- realistic patterns -------------------------------------------


def test_default_1_and_99_percentiles():
    """Default (0.01, 0.99) — extremes clipped to interior quantiles."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(10_000)
    r = winsorize(x)
    # Range should be about the same as np.quantile bounds.
    q1 = np.quantile(x, 0.01)
    q99 = np.quantile(x, 0.99)
    assert r.min() >= q1 - 1e-9
    assert r.max() <= q99 + 1e-9


def test_output_dtype_is_float():
    """Even int input is promoted so NaN can flow through."""
    r = winsorize(np.arange(10))
    assert r.dtype.kind == "f"
