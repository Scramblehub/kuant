"""Single-value bucket aggregation for scalar series and panels.

Sibling to `baragg`: takes a per-row bucket-id and aggregates each
bucket down to one value. The difference from `baragg` is scope:

- `baragg` knows about OHLCV — five per-bucket outputs under different
  semantics.
- `resample` produces ONE per-bucket output under a chosen reduction
  (`last`, `mean`, `sum`).

Use `resample` for anything that isn't a price bar: volumes, spreads,
signals, returns, breadth ratios, sentiment counts.

The kernel supports 1D input (a single series) or 2D input (a panel,
shape `(T, N)`). In the panel case, aggregation is per-column
independently.

Design: docs/kernels/data/resample.md.
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
from kuant.errors import KuantShapeError, KuantValueError

_ALLOWED_METHODS = ("last", "first", "mean", "sum")


@dataclass
class ResampleResult:
    """Aggregated series or panel.

    Attributes
    ----------
    bucket : 1D np.ndarray
        Unique bucket ids in ascending order.
    values : np.ndarray
        1D array of length `len(bucket)` for a scalar input, OR 2D of
        shape `(len(bucket), n_cols)` for a panel input.
    method : str
        Reduction used.
    n_input : 1D int np.ndarray
        Rows aggregated into each bucket.
    """

    bucket: np.ndarray
    values: np.ndarray
    method: str
    n_input: np.ndarray

    def summary(self) -> str:
        parts = [
            "=== ResampleResult ===",
            f"method:         {self.method}",
            f"n_output_bars:  {len(self.bucket)}",
            f"n_input_rows:   {int(self.n_input.sum())}",
            f"shape:          {self.values.shape}",
        ]
        return "\n".join(parts)

    def to_parquet(self, path, column_names=None) -> None:
        """Write to parquet. Requires `pyarrow`.

        For 2D values, `column_names` (a sequence of length `n_cols`) can
        be supplied; defaults to `col0, col1, ...`.
        """
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as e:
            require_dep(
                "pyarrow",
                kernel="resample.to_parquet",
                install="pip install pyarrow",
                cause=e,
            )
        cols = {"bucket": pa.array(self.bucket)}
        if self.values.ndim == 1:
            cols["value"] = pa.array(self.values)
        else:
            n_cols = self.values.shape[1]
            names = (
                tuple(column_names)
                if column_names is not None
                else tuple(f"col{i}" for i in range(n_cols))
            )
            if len(names) != n_cols:
                raise KuantValueError(
                    f"kuant.resample.to_parquet: got {len(names)} column "
                    f"names for {n_cols} columns.  [KE-SHAPE-EQUAL-LEN]\n"
                    f"  → Fix: pass one name per column"
                )
            for j, nm in enumerate(names):
                cols[str(nm)] = pa.array(self.values[:, j])
        cols["n_input"] = pa.array(self.n_input)
        pq.write_table(pa.table(cols), path)


def resample(bucket, values, method: str = "last") -> ResampleResult:
    """Aggregate a 1D series or `(T, N)` panel into coarser buckets.

    Parameters
    ----------
    bucket : 1D int array
        Bucket id per input row. Rows sharing an id are aggregated.
    values : 1D or 2D array
        - 1D length T: aggregate a single series.
        - 2D shape (T, N): aggregate a panel per-column.
    method : {'last', 'first', 'mean', 'sum'}
        Reduction. NaN is dropped inside each bucket (`nanmean`,
        `nansum`); buckets that are ALL NaN produce NaN for `mean` and
        0 for `sum`. `first` / `last` take the first/last row inside the
        bucket, INCLUDING if that row is NaN.

    Returns
    -------
    ResampleResult
        `.bucket`, `.values` (1D or 2D matching input), `.method`,
        `.n_input`.

    Notes
    -----
    - Distinct from `baragg`: this is scalar-per-bucket, not OHLCV.
    - Bucket ids need not be sorted or contiguous.
    - Missing buckets in the input sequence (gaps) are NOT filled.

    Examples
    --------
    >>> import numpy as np
    >>> # 6 rows into 2 buckets, mean reduction.
    >>> bucket = np.array([0, 0, 0, 1, 1, 1])
    >>> values = np.array([10.0, 20, 30, 100, 110, 120])
    >>> r = resample(bucket, values, method='mean')
    >>> r.values.tolist()
    [20.0, 110.0]
    """
    if method not in _ALLOWED_METHODS:
        raise KuantValueError(
            f"kuant.resample: 'method' must be one of {_ALLOWED_METHODS}, "
            f"got {method!r}.  [KE-VAL-RANGE]\n"
            f"  → Fix: pick one of {_ALLOWED_METHODS}"
        )

    bucket_arr = np.asarray(bucket)
    values_arr = np.asarray(values, dtype=np.float64)
    require_1d(bucket_arr, "bucket", kernel="resample")

    if values_arr.ndim not in (1, 2):
        raise KuantShapeError(
            f"kuant.resample: 'values' must be 1D or 2D, got shape "
            f"{values_arr.shape}.  [KE-SHAPE-EXPECTED]\n"
            f"  → Fix: pass a 1D series or a 2D panel of shape (T, N)"
        )
    if bucket_arr.dtype.kind not in "iu":
        raise KuantValueError(
            f"kuant.resample: 'bucket' must be an integer array, got "
            f"dtype {bucket_arr.dtype}.  [KE-VAL-RANGE]\n"
            f"  → Fix: cast to int64 first"
        )
    require_equal_length(bucket_arr, "bucket", values_arr, "values", kernel="resample")

    # Sort by bucket id (stable so first/last stays deterministic).
    order = np.argsort(bucket_arr, kind="stable")
    b_sorted = bucket_arr[order]
    v_sorted = values_arr[order] if values_arr.ndim == 1 else values_arr[order, :]

    unique_ids, first_idx, counts = np.unique(b_sorted, return_index=True, return_counts=True)
    n_out = unique_ids.size

    if values_arr.ndim == 1:
        out = np.empty(n_out, dtype=np.float64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for i in range(n_out):
                lo, hi = first_idx[i], first_idx[i] + counts[i]
                seg = v_sorted[lo:hi]
                if method == "last":
                    out[i] = seg[-1]
                elif method == "first":
                    out[i] = seg[0]
                elif method == "mean":
                    out[i] = np.nanmean(seg)
                else:  # sum
                    out[i] = np.nansum(seg)
    else:
        n_cols = v_sorted.shape[1]
        out = np.empty((n_out, n_cols), dtype=np.float64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for i in range(n_out):
                lo, hi = first_idx[i], first_idx[i] + counts[i]
                seg = v_sorted[lo:hi, :]
                if method == "last":
                    out[i, :] = seg[-1, :]
                elif method == "first":
                    out[i, :] = seg[0, :]
                elif method == "mean":
                    out[i, :] = np.nanmean(seg, axis=0)
                else:  # sum
                    out[i, :] = np.nansum(seg, axis=0)

    return ResampleResult(
        bucket=unique_ids,
        values=out,
        method=method,
        n_input=counts.astype(np.int64),
    )


__all__ = ["resample", "ResampleResult"]
