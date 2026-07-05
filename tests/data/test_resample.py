"""Tests for kuant.data.resample."""

from __future__ import annotations

import numpy as np
import pytest

from kuant.data.resample import ResampleResult, resample
from kuant.errors import KuantShapeError, KuantValueError


# ---------- reductions on 1D input ---------------------------------------


def test_last_returns_final_value_per_bucket():
    b = np.array([0, 0, 0, 1, 1])
    v = np.array([10.0, 20, 30, 100, 200])
    r = resample(b, v, method="last")
    assert r.bucket.tolist() == [0, 1]
    assert r.values.tolist() == [30.0, 200.0]


def test_first_returns_leading_value_per_bucket():
    b = np.array([0, 0, 1, 1])
    v = np.array([10.0, 20, 100, 200])
    r = resample(b, v, method="first")
    assert r.values.tolist() == [10.0, 100.0]


def test_mean_averages_per_bucket():
    b = np.array([0, 0, 0, 1, 1])
    v = np.array([10.0, 20, 30, 100, 200])
    r = resample(b, v, method="mean")
    assert r.values.tolist() == [20.0, 150.0]


def test_sum_totals_per_bucket():
    b = np.array([0, 0, 0, 1, 1])
    v = np.array([10.0, 20, 30, 100, 200])
    r = resample(b, v, method="sum")
    assert r.values.tolist() == [60.0, 300.0]


def test_returns_result_type():
    r = resample(np.array([0]), np.array([1.0]))
    assert isinstance(r, ResampleResult)


# ---------- 2D panel input ------------------------------------------------


def test_panel_mean_per_column():
    b = np.array([0, 0, 1, 1])
    v = np.array([[10.0, 100], [20, 200], [30, 300], [40, 400]])
    r = resample(b, v, method="mean")
    assert r.values.shape == (2, 2)
    assert r.values[0].tolist() == [15.0, 150.0]
    assert r.values[1].tolist() == [35.0, 350.0]


def test_panel_last_per_column():
    b = np.array([0, 0, 1, 1])
    v = np.array([[10.0, 100], [20, 200], [30, 300], [40, 400]])
    r = resample(b, v, method="last")
    assert r.values.shape == (2, 2)
    assert r.values[0].tolist() == [20.0, 200.0]
    assert r.values[1].tolist() == [40.0, 400.0]


# ---------- NaN policy ----------------------------------------------------


def test_mean_nans_drop_via_nanmean():
    b = np.array([0, 0, 0])
    v = np.array([10.0, np.nan, 20])
    r = resample(b, v, method="mean")
    assert r.values.tolist() == [15.0]  # nanmean([10, nan, 20]) = 15


def test_sum_all_nan_returns_zero():
    """nansum treats NaN as 0 → an all-NaN bucket sums to 0."""
    b = np.array([0, 0])
    v = np.array([np.nan, np.nan])
    r = resample(b, v, method="sum")
    assert r.values.tolist() == [0.0]


def test_mean_all_nan_returns_nan():
    b = np.array([0, 0])
    v = np.array([np.nan, np.nan])
    r = resample(b, v, method="mean")
    assert np.isnan(r.values).all()


def test_last_preserves_nan():
    """`last` takes the actual last value — NaN passes through."""
    b = np.array([0, 0])
    v = np.array([10.0, np.nan])
    r = resample(b, v, method="last")
    assert np.isnan(r.values[0])


# ---------- unsorted / non-contiguous inputs -----------------------------


def test_unsorted_bucket_ids_sorted_in_output():
    b = np.array([2, 0, 1, 0])
    v = np.array([200.0, 10, 100, 20])
    r = resample(b, v, method="mean")
    assert r.bucket.tolist() == [0, 1, 2]
    assert r.values.tolist() == [15.0, 100.0, 200.0]


def test_non_contiguous_ids_not_filled():
    b = np.array([0, 0, 5, 5])
    v = np.array([1.0, 2, 3, 4])
    r = resample(b, v, method="sum")
    assert r.bucket.tolist() == [0, 5]  # buckets 1..4 are NOT filled


# ---------- error contract -----------------------------------------------


def test_reject_bad_method():
    with pytest.raises(KuantValueError) as exc:
        resample(np.array([0]), np.array([1.0]), method="bogus")
    assert "method" in str(exc.value)


def test_reject_non_integer_bucket():
    with pytest.raises(KuantValueError):
        resample(np.array([0.0, 1.0]), np.array([1.0, 2.0]))


def test_reject_length_mismatch():
    with pytest.raises(Exception):
        resample(np.array([0, 1]), np.array([1.0, 2, 3]))


def test_reject_3d_values():
    with pytest.raises(KuantShapeError):
        resample(np.array([0, 1]), np.zeros((2, 3, 2)))


# ---------- summary + parquet --------------------------------------------


def test_summary_contains_method():
    r = resample(np.array([0, 0, 1]), np.array([1.0, 2, 3]), method="sum")
    s = r.summary()
    assert "ResampleResult" in s
    assert "sum" in s


def test_to_parquet_scalar_roundtrip(tmp_path):
    pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    r = resample(np.array([0, 0, 1]), np.array([10.0, 20, 30]))
    path = tmp_path / "resampled.parquet"
    r.to_parquet(path)
    cols = set(pq.read_table(path).column_names)
    assert cols == {"bucket", "value", "n_input"}


def test_to_parquet_panel_with_column_names(tmp_path):
    pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    b = np.array([0, 0, 1, 1])
    v = np.array([[10.0, 100], [20, 200], [30, 300], [40, 400]])
    r = resample(b, v, method="mean")
    path = tmp_path / "panel.parquet"
    r.to_parquet(path, column_names=("returns", "volume"))
    cols = set(pq.read_table(path).column_names)
    assert cols == {"bucket", "returns", "volume", "n_input"}


def test_to_parquet_wrong_column_count_rejected(tmp_path):
    pytest.importorskip("pyarrow")
    b = np.array([0, 0])
    v = np.array([[1.0, 2], [3, 4]])
    r = resample(b, v, method="mean")
    with pytest.raises(KuantValueError):
        r.to_parquet(tmp_path / "x.parquet", column_names=("only_one",))
