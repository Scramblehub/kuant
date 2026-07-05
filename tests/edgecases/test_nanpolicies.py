"""Tests for kuant.edgecases.nanpolicies."""

from __future__ import annotations

import numpy as np
import pytest

from kuant.edgecases import nanpolicies
from kuant.errors import KuantShapeError, KuantValueError


# ---------- strict -------------------------------------------------------


def test_strict_accepts_clean_1d():
    x = np.array([1.0, 2, 3])
    out = nanpolicies.strict(x)
    assert out.tolist() == [1.0, 2.0, 3.0]


def test_strict_rejects_nan():
    with pytest.raises(KuantValueError) as exc:
        nanpolicies.strict(np.array([1.0, np.nan, 3]))
    m = str(exc.value)
    assert "1 NaN" in m and "3" in m
    assert "→ Fix" in m


def test_strict_passes_through_int_input():
    """Integer arrays cannot hold NaN; pass through unchanged."""
    x = np.arange(5)
    out = nanpolicies.strict(x)
    assert out.dtype.kind in "iu"


# ---------- skipna --------------------------------------------------------


def test_skipna_1d_drops_nan():
    x = np.array([1.0, np.nan, 3, np.nan, 5])
    assert nanpolicies.skipna(x).tolist() == [1.0, 3.0, 5.0]


def test_skipna_2d_drops_rows_with_any_nan():
    """A row with ANY NaN column is dropped."""
    x = np.array([[1.0, 2], [3, np.nan], [5, 6], [np.nan, np.nan]])
    out = nanpolicies.skipna(x)
    assert out.shape == (2, 2)
    assert out.tolist() == [[1.0, 2.0], [5.0, 6.0]]


def test_skipna_empty_result():
    """All-NaN input → empty output."""
    x = np.array([np.nan, np.nan])
    assert nanpolicies.skipna(x).size == 0


def test_skipna_rejects_3d():
    with pytest.raises(KuantShapeError):
        nanpolicies.skipna(np.zeros((3, 2, 4)))


# ---------- forwardfill ---------------------------------------------------


def test_forwardfill_1d_fills_gaps():
    x = np.array([np.nan, 1.0, np.nan, 3, np.nan])
    out = nanpolicies.forwardfill(x)
    assert np.isnan(out[0])  # leading NaN preserved
    assert out[1:].tolist() == [1.0, 1.0, 3.0, 3.0]


def test_forwardfill_all_nan_preserved():
    x = np.array([np.nan, np.nan, np.nan])
    out = nanpolicies.forwardfill(x)
    assert np.isnan(out).all()


def test_forwardfill_2d_per_column():
    x = np.array([[np.nan, 10], [1, np.nan], [np.nan, 20]])
    out = nanpolicies.forwardfill(x)
    # Column 0: [nan, 1, 1]
    assert np.isnan(out[0, 0])
    assert out[1:, 0].tolist() == [1.0, 1.0]
    # Column 1: [10, 10, 20]
    assert out[:, 1].tolist() == [10.0, 10.0, 20.0]


# ---------- interpolate --------------------------------------------------


def test_interpolate_between_finites():
    x = np.array([1.0, np.nan, np.nan, 4.0])
    out = nanpolicies.interpolate(x)
    assert out.tolist() == [1.0, 2.0, 3.0, 4.0]


def test_interpolate_preserves_leading_and_trailing_nan():
    x = np.array([np.nan, 1.0, 2, 3, np.nan])
    out = nanpolicies.interpolate(x)
    assert np.isnan(out[0])
    assert np.isnan(out[-1])
    assert out[1:-1].tolist() == [1.0, 2.0, 3.0]


def test_interpolate_single_finite_returns_unchanged():
    """Only one finite value → cannot interpolate; return unchanged."""
    x = np.array([np.nan, 5.0, np.nan])
    out = nanpolicies.interpolate(x)
    assert np.isnan(out[0]) and np.isnan(out[2])
    assert out[1] == 5.0


def test_interpolate_2d_per_column():
    x = np.array([[1.0, 10], [np.nan, np.nan], [3, 30]])
    out = nanpolicies.interpolate(x)
    assert out[1].tolist() == [2.0, 20.0]


# ---------- dropcolumn ---------------------------------------------------


def test_dropcolumn_removes_sparse_columns():
    """Column with 40% finite dropped at min_finite_frac=0.5."""
    x = np.array(
        [
            [1.0, np.nan, 100],
            [2, np.nan, np.nan],
            [3, np.nan, 300],
            [4, np.nan, 400],
            [5, np.nan, 500],
        ]
    )
    out, mask = nanpolicies.dropcolumn(x, min_finite_frac=0.5)
    # Col 0 is 5/5 finite, col 1 is 0/5, col 2 is 4/5.
    assert mask.tolist() == [True, False, True]
    assert out.shape == (5, 2)


def test_dropcolumn_default_threshold_is_half():
    x = np.array([[np.nan, 1.0], [np.nan, 2], [1.0, 3]])
    _, mask = nanpolicies.dropcolumn(x)  # default 0.5
    # Col 0: 1/3 = 33% finite < 50% → dropped.
    # Col 1: 3/3 = 100% finite → kept.
    assert mask.tolist() == [False, True]


def test_dropcolumn_rejects_1d():
    with pytest.raises(KuantShapeError):
        nanpolicies.dropcolumn(np.array([1.0, 2, 3]))


def test_dropcolumn_rejects_bad_threshold():
    with pytest.raises(KuantValueError):
        nanpolicies.dropcolumn(np.zeros((3, 2)), min_finite_frac=1.5)


# ---------- registry -----------------------------------------------------


def test_get_returns_callable_for_valid_name():
    for name in ("strict", "skipna", "forwardfill", "interpolate", "dropcolumn"):
        policy = nanpolicies.get(name)
        assert callable(policy)


def test_get_rejects_unknown_name():
    with pytest.raises(KuantValueError) as exc:
        nanpolicies.get("magic")
    assert "magic" in str(exc.value)


def test_available_returns_sorted_tuple():
    names = nanpolicies.available()
    assert isinstance(names, tuple)
    assert names == tuple(sorted(names))
    assert set(names) == {"strict", "skipna", "forwardfill", "interpolate", "dropcolumn"}


def test_get_returns_same_callable_as_attribute_access():
    assert nanpolicies.get("forwardfill") is nanpolicies.forwardfill


# ---------- composition --------------------------------------------------


def test_get_by_name_end_to_end():
    """Config-driven use case: policy chosen at runtime."""
    policy_name = "interpolate"
    policy = nanpolicies.get(policy_name)
    x = np.array([1.0, np.nan, 3.0])
    assert policy(x).tolist() == [1.0, 2.0, 3.0]
