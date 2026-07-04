"""Tests for kuant.data.align."""

from __future__ import annotations

import numpy as np
import pytest

from kuant.data.align import AlignResult, align
from kuant.errors import KuantShapeError, KuantValueError


# ---------- inner join ----------------------------------------------------


def test_inner_intersects_indices():
    idx_a = np.array([1, 2, 3, 4])
    val_a = np.array([10.0, 20, 30, 40])
    idx_b = np.array([2, 3, 4, 5])
    val_b = np.array([200.0, 300, 400, 500])
    r = align((idx_a, val_a), (idx_b, val_b), method="inner")
    assert r.index.tolist() == [2, 3, 4]
    assert r.arrays[0].tolist() == [20.0, 30.0, 40.0]
    assert r.arrays[1].tolist() == [200.0, 300.0, 400.0]


def test_inner_result_type():
    idx = np.arange(5)
    val = np.arange(5.0)
    r = align((idx, val), (idx, val), method="inner")
    assert isinstance(r, AlignResult)
    assert r.method == "inner"
    assert r.original_lengths == (5, 5)


def test_inner_empty_intersection():
    """Disjoint indices return an AlignResult with an empty shared index."""
    r = align(
        (np.array([1, 2, 3]), np.array([1.0, 2, 3])),
        (np.array([10, 20, 30]), np.array([10.0, 20, 30])),
        method="inner",
    )
    assert r.index.size == 0
    assert all(a.size == 0 for a in r.arrays)


def test_inner_three_way():
    idx_a = np.array([1, 2, 3, 4])
    idx_b = np.array([2, 3, 4, 5])
    idx_c = np.array([3, 4, 5, 6])
    val = np.arange(4.0)
    r = align(
        (idx_a, val),
        (idx_b, val),
        (idx_c, val),
        method="inner",
    )
    assert r.index.tolist() == [3, 4]


def test_unsorted_input_sorted_before_align():
    """Alignment sorts internally — unsorted input is OK."""
    idx_a = np.array([3, 1, 2, 4])
    val_a = np.array([30.0, 10, 20, 40])
    idx_b = np.array([4, 3, 2])
    val_b = np.array([4.0, 3.0, 2.0])
    r = align((idx_a, val_a), (idx_b, val_b), method="inner")
    assert r.index.tolist() == [2, 3, 4]
    assert r.arrays[0].tolist() == [20.0, 30.0, 40.0]
    assert r.arrays[1].tolist() == [2.0, 3.0, 4.0]


# ---------- outer join ----------------------------------------------------


def test_outer_union_with_nan_fill():
    idx_a = np.array([1, 2, 3])
    val_a = np.array([10.0, 20, 30])
    idx_b = np.array([2, 3, 4])
    val_b = np.array([200.0, 300, 400])
    r = align((idx_a, val_a), (idx_b, val_b), method="outer")
    assert r.index.tolist() == [1, 2, 3, 4]
    # First series: NaN at position 4 (not present).
    assert np.isnan(r.arrays[0][-1])
    assert r.arrays[0][:3].tolist() == [10.0, 20.0, 30.0]
    # Second series: NaN at position 1.
    assert np.isnan(r.arrays[1][0])
    assert r.arrays[1][1:].tolist() == [200.0, 300.0, 400.0]


def test_outer_promotes_int_dtype_to_float():
    """Outer needs NaN → int inputs promote to float64."""
    idx_a = np.array([1, 2, 3])
    val_a = np.array([10, 20, 30], dtype=np.int64)
    idx_b = np.array([2, 3, 4])
    val_b = np.array([200, 300, 400], dtype=np.int64)
    r = align((idx_a, val_a), (idx_b, val_b), method="outer")
    assert r.arrays[0].dtype == np.float64
    assert r.arrays[1].dtype == np.float64


# ---------- forward join --------------------------------------------------


def test_forward_fills_gaps_after_first_observation():
    idx_a = np.array([1, 2, 3, 4, 5])
    val_a = np.array([10.0, 20, 30, 40, 50])
    idx_b = np.array([2, 4])
    val_b = np.array([200.0, 400])
    r = align((idx_a, val_a), (idx_b, val_b), method="forward")
    assert r.index.tolist() == [1, 2, 3, 4, 5]
    # Second series: NaN at pos 1 (before first obs), then 200, 200, 400, 400.
    b = r.arrays[1]
    assert np.isnan(b[0])
    assert b[1:].tolist() == [200.0, 200.0, 400.0, 400.0]


def test_forward_leading_nan_preserved():
    """Positions before a series's first observation stay NaN; after the
    first observation, forward-fill propagates the last-seen value."""
    idx_a = np.array([1, 2, 3])
    val_a = np.array([1.0, 2, 3])
    idx_b = np.array([10])
    val_b = np.array([100.0])
    r = align((idx_a, val_a), (idx_b, val_b), method="forward")
    assert r.index.tolist() == [1, 2, 3, 10]
    # First series: 1, 2, 3 observed; position 10 forward-filled with 3.
    assert r.arrays[0].tolist() == [1.0, 2.0, 3.0, 3.0]
    # Second series: positions 1..3 are BEFORE first obs at 10 → NaN.
    assert np.isnan(r.arrays[1][:3]).all()
    assert r.arrays[1][-1] == 100.0


# ---------- 2D value support ----------------------------------------------


def test_2d_values_column_count_preserved():
    """2D value input (a panel) preserves columns after alignment."""
    idx_a = np.array([1, 2, 3, 4])
    val_a = np.arange(4 * 3, dtype=np.float64).reshape(4, 3)
    idx_b = np.array([2, 3])
    val_b = np.array([[100.0, 200, 300], [400, 500, 600]])
    r = align((idx_a, val_a), (idx_b, val_b), method="inner")
    assert r.arrays[0].shape == (2, 3)
    assert r.arrays[1].shape == (2, 3)
    # Row 0 should be idx_a's row where idx_a == 2, which is row index 1.
    assert r.arrays[0][0].tolist() == val_a[1].tolist()


def test_2d_values_outer_fills_nan_rows():
    idx_a = np.array([1, 2])
    val_a = np.array([[1.0, 2, 3], [4, 5, 6]])
    idx_b = np.array([2, 3])
    val_b = np.array([[10.0, 20, 30], [40, 50, 60]])
    r = align((idx_a, val_a), (idx_b, val_b), method="outer")
    assert r.arrays[0].shape == (3, 3)
    # Row at index 3 should be all NaN.
    assert np.isnan(r.arrays[0][-1]).all()


# ---------- to_dict / summary ---------------------------------------------


def test_to_dict_zips_names_with_arrays():
    idx = np.arange(3)
    val = np.arange(3.0)
    r = align((idx, val), (idx, val * 2))
    d = r.to_dict(("returns", "vol"))
    assert set(d.keys()) == {"returns", "vol"}
    assert d["returns"].tolist() == [0.0, 1.0, 2.0]


def test_to_dict_rejects_wrong_name_count():
    idx = np.arange(3)
    val = np.arange(3.0)
    r = align((idx, val), (idx, val))
    with pytest.raises(KuantValueError):
        r.to_dict(("only_one",))


def test_summary_contains_metadata():
    idx = np.arange(3)
    val = np.arange(3.0)
    r = align((idx, val), (idx, val))
    s = r.summary()
    assert "AlignResult" in s
    assert "inner" in s or r.method in s


# ---------- error contract ------------------------------------------------


def test_reject_single_input():
    idx = np.arange(3)
    val = np.arange(3.0)
    with pytest.raises(KuantValueError):
        align((idx, val))


def test_reject_bad_method():
    idx = np.arange(3)
    val = np.arange(3.0)
    with pytest.raises(KuantValueError) as exc:
        align((idx, val), (idx, val), method="jumbo")
    m = str(exc.value)
    assert "method" in m and "inner" in m


def test_reject_non_tuple_pair():
    idx = np.arange(3)
    val = np.arange(3.0)
    with pytest.raises(KuantValueError):
        align([idx, val], (idx, val))  # list, not tuple


def test_reject_index_values_length_mismatch():
    idx = np.arange(3)
    val = np.arange(4.0)
    with pytest.raises(KuantShapeError):
        align((idx, val), (idx, np.arange(3.0)))


def test_reject_duplicate_index():
    idx = np.array([1, 2, 2, 3])
    val = np.array([1.0, 2, 2.5, 3])
    with pytest.raises(KuantValueError) as exc:
        align((idx, val), (np.array([1, 2, 3]), np.array([1.0, 2, 3])))
    m = str(exc.value)
    assert "duplicate" in m


def test_reject_mixed_dtype_kinds():
    """Numeric + datetime index kinds should be rejected."""
    idx_num = np.array([1, 2, 3])
    val = np.array([1.0, 2, 3])
    idx_dt = np.array(["2024-01-01", "2024-01-02", "2024-01-03"], dtype="datetime64[D]")
    with pytest.raises(KuantShapeError):
        align((idx_num, val), (idx_dt, val))


def test_reject_3d_values():
    idx = np.arange(3)
    val3d = np.zeros((3, 2, 4))
    with pytest.raises(KuantShapeError):
        align((idx, val3d), (idx, np.arange(3.0)))


# ---------- datetime64 indices --------------------------------------------


def test_datetime_indices_align_by_value():
    idx_a = np.array(["2024-01-01", "2024-01-02", "2024-01-03"], dtype="datetime64[D]")
    val_a = np.array([1.0, 2, 3])
    idx_b = np.array(["2024-01-02", "2024-01-03", "2024-01-04"], dtype="datetime64[D]")
    val_b = np.array([20.0, 30, 40])
    r = align((idx_a, val_a), (idx_b, val_b), method="inner")
    assert len(r.index) == 2
    assert r.arrays[0].tolist() == [2.0, 3.0]
    assert r.arrays[1].tolist() == [20.0, 30.0]


# ---------- to_parquet ---------------------------------------------------


def test_to_parquet_roundtrip(tmp_path):
    """Parquet output round-trips values via pyarrow."""
    pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    idx_a = np.array([1, 2, 3])
    val_a = np.array([10.0, 20, 30])
    idx_b = np.array([2, 3, 4])
    val_b = np.array([200.0, 300, 400])
    r = align((idx_a, val_a), (idx_b, val_b), method="outer")

    path = tmp_path / "aligned.parquet"
    r.to_parquet(path, names=("returns", "vol"))
    assert path.exists()

    table = pq.read_table(path)
    cols = set(table.column_names)
    assert cols == {"index", "returns", "vol"}
    assert table.num_rows == 4


def test_to_parquet_default_names(tmp_path):
    pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    idx = np.arange(3)
    val = np.arange(3.0)
    r = align((idx, val), (idx, val))
    path = tmp_path / "t.parquet"
    r.to_parquet(path)
    cols = pq.read_table(path).column_names
    assert cols == ["index", "arr0", "arr1"]


def test_to_parquet_2d_values_explode_columns(tmp_path):
    """2D value arrays become one parquet column per feature."""
    pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    idx = np.arange(3)
    panel = np.arange(9.0).reshape(3, 3)
    r = align((idx, panel), (idx, panel * 2), method="inner")
    path = tmp_path / "panel.parquet"
    r.to_parquet(path, names=("A", "B"))
    cols = pq.read_table(path).column_names
    assert set(cols) == {"index", "A[0]", "A[1]", "A[2]", "B[0]", "B[1]", "B[2]"}
