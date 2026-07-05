"""Bar-frequency aggregation with OHLCV semantics.

Aggregates a stream of higher-frequency bars into coarser ones. The
distinction from generic `resample` (proposed for v1.1) is that
`baragg` knows about **OHLCV semantics**: open uses the FIRST value in
each bucket, high the MAX, low the MIN, close the LAST, volume the SUM.

Buckets are defined by an integer bucket-id per input row. Any function
that maps timestamps or positions to bucket labels can drive this —
we don't ship a calendar; users compose. Common patterns:

    # 5-minute buckets on nanosecond timestamps
    bucket = ts_ns // (5 * 60 * 1_000_000_000)

    # Daily buckets on datetime64[D] dates
    bucket = dates.astype("datetime64[D]").astype(np.int64)

    # 21-day rolling groups by position
    bucket = np.arange(n) // 21

Rows sharing a bucket id are aggregated. Bucket ids need not be
contiguous; the output is sorted by bucket id ascending. Missing
buckets (gaps in the id sequence) are NOT filled — pair with `align`
or `resample` if you need a dense output.

Design: docs/kernels/data/baragg.md.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from kuant._validation import (
    require_1d,
    require_dep,
    require_equal_length,
)
from kuant.errors import KuantValueError


@dataclass
class BarAggResult:
    """Aggregated OHLCV bars.

    Every array has the same length: one entry per unique bucket id.

    Attributes
    ----------
    bucket : 1D np.ndarray
        Unique bucket ids in ascending order.
    opens, highs, lows, closes : 1D np.ndarray
        Per-bucket first, max, min, last of the input `close` (or the
        column passed to each). `close` is the canonical input axis.
    volumes : 1D np.ndarray or None
        Per-bucket sum of `volume`, or None if no volume was supplied.
    n_input : 1D int np.ndarray
        Number of input rows aggregated into each bucket. A bucket with
        `n_input == 1` had no meaningful high-low spread; a bucket with
        many rows had heavy activity.
    """

    bucket: np.ndarray
    opens: np.ndarray
    highs: np.ndarray
    lows: np.ndarray
    closes: np.ndarray
    volumes: np.ndarray | None
    n_input: np.ndarray

    def summary(self) -> str:
        parts = [
            "=== BarAggResult ===",
            f"n_output_bars:  {len(self.bucket)}",
            f"n_input_rows:   {int(self.n_input.sum())}",
            f"rows/bar avg:   {float(self.n_input.mean()):.2f}",
            f"has volume:     {self.volumes is not None}",
        ]
        return "\n".join(parts)

    def to_parquet(self, path) -> None:
        """Write the aggregated bars to a parquet file.

        Columns: bucket, open, high, low, close, [volume], n_input.
        Requires `pyarrow`.
        """
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as e:
            require_dep(
                "pyarrow",
                kernel="baragg.to_parquet",
                install="pip install pyarrow",
                cause=e,
            )

        cols = {
            "bucket": pa.array(self.bucket),
            "open": pa.array(self.opens),
            "high": pa.array(self.highs),
            "low": pa.array(self.lows),
            "close": pa.array(self.closes),
            "n_input": pa.array(self.n_input),
        }
        if self.volumes is not None:
            cols["volume"] = pa.array(self.volumes)
        pq.write_table(pa.table(cols), path)


def baragg(
    bucket,
    close,
    open=None,
    high=None,
    low=None,
    volume=None,
) -> BarAggResult:
    """Aggregate a stream of bars into coarser buckets under OHLCV rules.

    Parameters
    ----------
    bucket : 1D array of int
        Bucket id per input row. Rows sharing an id are aggregated.
        Ids need not be contiguous or sorted — internal sort handles that.
    close : 1D array
        Per-row close values. This is the required series; the other
        four axes default to it if not supplied (i.e. a tick series
        where the "bar" is a single price collapses to open=high=low=close).
    open, high, low : 1D array, optional
        Per-row open/high/low. If any is None, `close` is used for that
        axis. Passing full OHLC is the normal case when aggregating from
        an already-OHLC input (e.g. 1-min bars into 5-min bars).
    volume : 1D array, optional
        Per-row volume. If None, the result's `.volumes` is None.

    Returns
    -------
    BarAggResult
        With `.bucket` (sorted unique ids) and per-bucket
        `.opens`, `.highs`, `.lows`, `.closes`, `.volumes` (None if
        not supplied), `.n_input`.

    Notes
    -----
    - NaN policy: NaN in `close` propagates via `nanmax`/`nanmin`. A
      bucket where EVERY row's close is NaN produces NaN for open/high/
      low/close. Volume uses `nansum`, so NaN volumes are treated as 0.
    - Requires `len(bucket) == len(close)` — same length constraint as
      the other axes when supplied.
    - Missing bucket ids in the input series (gaps) are NOT filled in
      the output. Compose with `align(method='outer')` if you need
      dense output.

    Examples
    --------
    >>> import numpy as np
    >>> # 6 minute-bars, aggregate to 3-minute buckets.
    >>> bucket = np.array([0, 0, 0, 1, 1, 1])
    >>> close = np.array([100.0, 101, 102, 105, 103, 104])
    >>> volume = np.array([10, 20, 30, 40, 50, 60])
    >>> r = baragg(bucket, close, volume=volume)
    >>> r.bucket.tolist()
    [0, 1]
    >>> r.opens.tolist()
    [100.0, 105.0]
    >>> r.highs.tolist()
    [102.0, 105.0]
    >>> r.lows.tolist()
    [100.0, 103.0]
    >>> r.closes.tolist()
    [102.0, 104.0]
    >>> r.volumes.tolist()
    [60, 150]
    """
    bucket_arr = np.asarray(bucket)
    close_arr = np.asarray(close, dtype=np.float64)
    require_1d(bucket_arr, "bucket", kernel="baragg")
    require_1d(close_arr, "close", kernel="baragg")
    require_equal_length(bucket_arr, "bucket", close_arr, "close", kernel="baragg")
    if bucket_arr.dtype.kind not in "iu":
        raise KuantValueError(
            f"kuant.baragg: 'bucket' must be an integer array, got dtype "
            f"{bucket_arr.dtype}.  [KE-VAL-RANGE]\n"
            f"  → Fix: convert bucket ids to int64 first — e.g. "
            f"`bucket.astype(np.int64)` or `(ts // period).astype(np.int64)`"
        )

    # Optional axes: default to close if missing.
    def _axis(name: str, arr):
        if arr is None:
            return close_arr
        a = np.asarray(arr, dtype=np.float64)
        require_1d(a, name, kernel="baragg")
        require_equal_length(bucket_arr, "bucket", a, name, kernel="baragg")
        return a

    open_arr = _axis("open", open)
    high_arr = _axis("high", high)
    low_arr = _axis("low", low)

    vol_arr = None
    if volume is not None:
        vol_arr = np.asarray(volume, dtype=np.float64)
        require_1d(vol_arr, "volume", kernel="baragg")
        require_equal_length(bucket_arr, "bucket", vol_arr, "volume", kernel="baragg")

    # Sort by bucket id, then group.
    order = np.argsort(bucket_arr, kind="stable")
    b_sorted = bucket_arr[order]
    open_s = open_arr[order]
    high_s = high_arr[order]
    low_s = low_arr[order]
    close_s = close_arr[order]
    vol_s = vol_arr[order] if vol_arr is not None else None

    # np.unique gives us the group boundaries via return_index + return_counts.
    unique_ids, first_idx, counts = np.unique(b_sorted, return_index=True, return_counts=True)

    # For each bucket the range is [first_idx[i], first_idx[i] + counts[i]).
    n_out = unique_ids.size
    opens_out = np.empty(n_out, dtype=np.float64)
    highs_out = np.empty(n_out, dtype=np.float64)
    lows_out = np.empty(n_out, dtype=np.float64)
    closes_out = np.empty(n_out, dtype=np.float64)
    n_input_out = counts.astype(np.int64)

    if vol_s is not None:
        vols_out = np.empty(n_out, dtype=np.float64)

    # Group-by loop. For large n_out we could vectorize via np.add.reduceat
    # etc., but per-bucket work is O(count) and dominated by max/min for
    # buckets with many rows — leave the loop simple until profiling says
    # otherwise.
    #
    # nanmax/nanmin emit "All-NaN slice encountered" RuntimeWarnings when a
    # bucket has zero non-NaN entries. The result is NaN, which is exactly
    # what we want — silence the noise so users don't see warnings the
    # kernel handles correctly.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for i in range(n_out):
            lo, hi = first_idx[i], first_idx[i] + counts[i]
            opens_out[i] = open_s[lo]  # FIRST — inputs are sorted stably
            closes_out[i] = close_s[hi - 1]  # LAST
            highs_out[i] = np.nanmax(high_s[lo:hi])
            lows_out[i] = np.nanmin(low_s[lo:hi])
            if vol_s is not None:
                vols_out[i] = np.nansum(vol_s[lo:hi])

    return BarAggResult(
        bucket=unique_ids,
        opens=opens_out,
        highs=highs_out,
        lows=lows_out,
        closes=closes_out,
        volumes=vols_out if vol_s is not None else None,
        n_input=n_input_out,
    )


__all__ = ["baragg", "BarAggResult"]
