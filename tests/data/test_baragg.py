"""Tests for kuant.data.baragg."""

from __future__ import annotations

import numpy as np
import pytest

from kuant.data.baragg import BarAggResult, baragg
from kuant.errors import KuantShapeError, KuantValueError


# ---------- OHLCV semantics -----------------------------------------------


def test_basic_two_bucket_aggregation():
    bucket = np.array([0, 0, 0, 1, 1, 1])
    close = np.array([100.0, 101, 102, 105, 103, 104])
    volume = np.array([10, 20, 30, 40, 50, 60])
    r = baragg(bucket, close, volume=volume)
    assert r.bucket.tolist() == [0, 1]
    assert r.opens.tolist() == [100.0, 105.0]
    assert r.highs.tolist() == [102.0, 105.0]
    assert r.lows.tolist() == [100.0, 103.0]
    assert r.closes.tolist() == [102.0, 104.0]
    assert r.volumes.tolist() == [60.0, 150.0]
    assert r.n_input.tolist() == [3, 3]


def test_full_ohlc_input_preserves_bar_extremes():
    """When full OHLC is supplied, high should come from `high`, not `close`."""
    bucket = np.array([0, 0])
    open_ = np.array([100.0, 101])
    high = np.array([108.0, 106])  # peak on bar 0
    low = np.array([98.0, 99])
    close = np.array([102.0, 103])
    r = baragg(bucket, close, open=open_, high=high, low=low)
    assert r.opens.tolist() == [100.0]  # first bar's open
    assert r.highs.tolist() == [108.0]  # max across per-bar highs
    assert r.lows.tolist() == [98.0]  # min across per-bar lows
    assert r.closes.tolist() == [103.0]  # last bar's close


def test_tick_series_collapses_to_close():
    """When only `close` is supplied, open=high=low=close per bucket."""
    bucket = np.array([0, 0, 0])
    close = np.array([100.0, 102, 101])
    r = baragg(bucket, close)
    assert r.opens.tolist() == [100.0]
    assert r.highs.tolist() == [102.0]
    assert r.lows.tolist() == [100.0]
    assert r.closes.tolist() == [101.0]
    assert r.volumes is None


def test_result_type():
    r = baragg(np.array([0]), np.array([100.0]))
    assert isinstance(r, BarAggResult)


# ---------- ordering + non-contiguous ids ---------------------------------


def test_unsorted_bucket_ids_get_sorted_output():
    bucket = np.array([2, 0, 1, 0, 2, 1])
    close = np.array([200.0, 100, 150, 105, 205, 155])
    r = baragg(bucket, close)
    assert r.bucket.tolist() == [0, 1, 2]
    # Bucket 0 has close values [100, 105] in input order — first is 100.
    assert r.opens.tolist() == [100.0, 150.0, 200.0]
    assert r.closes.tolist() == [105.0, 155.0, 205.0]


def test_non_contiguous_ids_preserved_not_filled():
    """Gaps in bucket ids (e.g. 0, 3, 5) don't produce empty output buckets."""
    bucket = np.array([0, 0, 3, 3, 5])
    close = np.array([100.0, 101, 200, 201, 300])
    r = baragg(bucket, close)
    assert r.bucket.tolist() == [0, 3, 5]
    # 1 and 2 and 4 are not present at all.


def test_single_row_bucket_has_equal_ohlc():
    bucket = np.array([0, 1])
    close = np.array([100.0, 200.0])
    r = baragg(bucket, close)
    assert r.opens.tolist() == r.closes.tolist() == r.highs.tolist() == r.lows.tolist()
    assert r.n_input.tolist() == [1, 1]


# ---------- NaN policy ----------------------------------------------------


def test_nan_close_flows_through_nanmax_nanmin():
    """NaN in high/low is ignored via nanmax/nanmin; a mixed bucket keeps
    the real max/min from the non-NaN rows."""
    bucket = np.array([0, 0, 0])
    close = np.array([100.0, np.nan, 105])
    high = np.array([102.0, np.nan, 108])
    low = np.array([99.0, np.nan, 103])
    r = baragg(bucket, close, high=high, low=low)
    # nanmax across [102, nan, 108] = 108; nanmin across [99, nan, 103] = 99.
    assert r.highs.tolist() == [108.0]
    assert r.lows.tolist() == [99.0]


def test_all_nan_bucket_produces_nan():
    bucket = np.array([0, 0])
    close = np.array([np.nan, np.nan])
    r = baragg(bucket, close)
    assert np.isnan(r.highs).all()
    assert np.isnan(r.lows).all()


def test_nan_volume_treated_as_zero():
    """nansum drops NaN so volume totals ignore missing volumes."""
    bucket = np.array([0, 0, 0])
    close = np.array([1.0, 2, 3])
    volume = np.array([10.0, np.nan, 30.0])
    r = baragg(bucket, close, volume=volume)
    assert r.volumes.tolist() == [40.0]


# ---------- error contract ------------------------------------------------


def test_reject_non_integer_bucket():
    with pytest.raises(KuantValueError) as exc:
        baragg(np.array([0.0, 1.0]), np.array([1.0, 2.0]))
    assert "bucket" in str(exc.value)


def test_reject_length_mismatch():
    with pytest.raises(KuantShapeError):
        baragg(np.array([0, 1]), np.array([1.0, 2.0, 3.0]))


def test_reject_2d_close():
    with pytest.raises(KuantShapeError):
        baragg(np.array([0, 1]), np.zeros((2, 3)))


def test_reject_length_mismatch_on_volume():
    with pytest.raises(KuantShapeError):
        baragg(
            np.array([0, 0, 0]),
            np.array([1.0, 2, 3]),
            volume=np.array([10.0, 20]),  # wrong length
        )


# ---------- summary + parquet --------------------------------------------


def test_summary_contains_metadata():
    r = baragg(np.array([0, 0, 1]), np.array([1.0, 2, 3]))
    s = r.summary()
    assert "BarAggResult" in s
    assert "n_output_bars" in s


def test_to_parquet_roundtrip(tmp_path):
    pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    r = baragg(
        np.array([0, 0, 1, 1]),
        np.array([100.0, 101, 105, 103]),
        volume=np.array([10, 20, 30, 40]),
    )
    path = tmp_path / "bars.parquet"
    r.to_parquet(path)
    table = pq.read_table(path)
    cols = set(table.column_names)
    assert cols == {"bucket", "open", "high", "low", "close", "volume", "n_input"}
    assert table.num_rows == 2


def test_to_parquet_without_volume_omits_column(tmp_path):
    pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    r = baragg(np.array([0, 1]), np.array([1.0, 2.0]))
    path = tmp_path / "bars.parquet"
    r.to_parquet(path)
    cols = set(pq.read_table(path).column_names)
    assert "volume" not in cols
    assert cols == {"bucket", "open", "high", "low", "close", "n_input"}


# ---------- practical composition ------------------------------------------


def test_daily_buckets_from_datetime64():
    """Compose baragg on top of datetime64 bucket ids."""
    dates = np.array(
        [
            "2024-01-01T09:30",
            "2024-01-01T09:35",
            "2024-01-01T15:59",
            "2024-01-02T09:30",
            "2024-01-02T15:00",
        ],
        dtype="datetime64[m]",
    )
    close = np.array([100.0, 101, 102, 200, 205])
    volume = np.array([10, 20, 30, 40, 50])

    bucket = dates.astype("datetime64[D]").astype(np.int64)
    r = baragg(bucket, close, volume=volume)
    assert len(r.bucket) == 2
    assert r.opens.tolist() == [100.0, 200.0]
    assert r.closes.tolist() == [102.0, 205.0]
    assert r.volumes.tolist() == [60.0, 90.0]
