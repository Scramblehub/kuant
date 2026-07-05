"""Tests for kuant.portfolio.contribution."""

from __future__ import annotations

import numpy as np
import pytest

from kuant.errors import KuantNumericWarning, KuantShapeError
from kuant.portfolio.contribution import ContributionResult, contribution


# ---------- basic mechanics ---------------------------------------------


def test_per_asset_totals_match_manual():
    positions = np.array([[1.0, 2, 0], [1, 2, 1]])
    returns = np.array([[0.01, 0.02, 0.03], [0.02, -0.01, 0.05]])
    r = contribution(positions, returns, asset_names=["A", "B", "C"])
    # A: 1*0.01 + 1*0.02 = 0.03
    # B: 2*0.02 + 2*(-0.01) = 0.02
    # C: 0*0.03 + 1*0.05 = 0.05
    assert np.allclose(r.total_by_asset, [0.03, 0.02, 0.05])
    assert abs(r.total - 0.10) < 1e-9


def test_per_bar_totals_match():
    positions = np.array([[1.0, 1], [2, 2]])
    returns = np.array([[0.01, -0.01], [0.02, 0.02]])
    r = contribution(positions, returns)
    # Bar 0: 1*0.01 + 1*-0.01 = 0.0
    # Bar 1: 2*0.02 + 2*0.02 = 0.08
    assert np.allclose(r.total_by_bar, [0.0, 0.08])


def test_per_bar_pnl_element_wise():
    positions = np.array([[1.0, 2], [3, 4]])
    returns = np.array([[0.1, 0.2], [0.3, 0.4]])
    r = contribution(positions, returns)
    expected = positions * returns
    assert np.allclose(r.per_bar_pnl, expected)


# ---------- NaN → 0 in totals -------------------------------------------


def test_nan_position_treated_as_zero_pnl():
    """NaN * anything should contribute 0 to the totals (not NaN)."""
    positions = np.array([[1.0, np.nan], [1, 1]])
    returns = np.array([[0.1, 0.2], [0.05, 0.05]])
    r = contribution(positions, returns)
    # Bar-0 asset-1: NaN * 0.2 → treated as 0.
    assert r.total_by_asset[1] == 0.05
    assert abs(r.total - (0.1 + 0.05 + 0.05)) < 1e-9


def test_partial_coverage_warns():
    """<80% finite cells → warning."""
    T, N = 20, 5
    positions = np.zeros((T, N))
    returns = np.zeros((T, N))
    # Fill only 30% of cells with finite values.
    n_finite = int(0.3 * T * N)
    idx = np.arange(T * N)
    finite_positions = idx[:n_finite]
    positions.flat[:] = np.nan
    returns.flat[:] = np.nan
    positions.flat[finite_positions] = 1.0
    returns.flat[finite_positions] = 0.01
    with pytest.warns(KuantNumericWarning) as record:
        contribution(positions, returns)
    assert any("KW-CONTRIB-PARTIAL-COVERAGE" in str(w.message) for w in record)


def test_full_coverage_no_warning():
    positions = np.ones((10, 3))
    returns = np.full((10, 3), 0.01)
    import warnings as _w

    with _w.catch_warnings():
        _w.simplefilter("error", KuantNumericWarning)
        contribution(positions, returns)


# ---------- group aggregation -------------------------------------------


def test_group_aggregation():
    positions = np.array([[1.0, 1, 1, 1]])
    returns = np.array([[0.01, 0.02, 0.03, 0.04]])
    group = np.array(["tech", "tech", "energy", "energy"])
    r = contribution(positions, returns, group=group)
    assert r.per_group is not None
    assert abs(r.per_group["tech"] - 0.03) < 1e-9
    assert abs(r.per_group["energy"] - 0.07) < 1e-9


def test_no_group_leaves_per_group_none():
    positions = np.ones((5, 2))
    returns = np.full((5, 2), 0.01)
    r = contribution(positions, returns)
    assert r.per_group is None


def test_reject_wrong_group_length():
    positions = np.ones((5, 3))
    returns = np.full((5, 3), 0.01)
    with pytest.raises(KuantShapeError):
        contribution(positions, returns, group=np.array(["A", "B"]))


# ---------- asset_names -------------------------------------------------


def test_asset_names_stored_in_result():
    positions = np.ones((3, 2))
    returns = np.full((3, 2), 0.01)
    r = contribution(positions, returns, asset_names=["X", "Y"])
    assert r.asset_names.tolist() == ["X", "Y"]


def test_reject_wrong_asset_names_length():
    positions = np.ones((3, 2))
    returns = np.full((3, 2), 0.01)
    with pytest.raises(KuantShapeError):
        contribution(positions, returns, asset_names=["only_one"])


# ---------- error contract ----------------------------------------------


def test_reject_shape_mismatch():
    positions = np.ones((5, 3))
    returns = np.full((5, 2), 0.01)
    with pytest.raises(KuantShapeError):
        contribution(positions, returns)


def test_reject_1d_positions():
    with pytest.raises(Exception):
        contribution(np.arange(5.0), np.arange(5.0))


# ---------- result contract ---------------------------------------------


def test_returns_dataclass():
    r = contribution(np.ones((3, 2)), np.full((3, 2), 0.01))
    assert isinstance(r, ContributionResult)


def test_summary_contains_top_contributors():
    positions = np.array([[1.0, 1, 1]])
    returns = np.array([[0.05, 0.01, -0.02]])
    r = contribution(positions, returns, asset_names=["A", "B", "C"])
    s = r.summary()
    assert "ContributionResult" in s
    assert "top" in s


def test_to_parquet_roundtrip(tmp_path):
    pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    positions = np.ones((3, 2))
    returns = np.full((3, 2), 0.01)
    r = contribution(positions, returns, asset_names=["A", "B"])
    path = tmp_path / "c.parquet"
    r.to_parquet(path)
    cols = set(pq.read_table(path).column_names)
    assert cols == {"asset", "total_pnl"}
