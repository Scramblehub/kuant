"""Multi-series alignment on a shared index.

Every downstream kernel that combines two or more series needs them
aligned on a common index — trading days, timestamps, tick positions.
`align` is the primitive: it takes 2+ `(index, values)` pairs and
returns them aligned on the requested join.

Three join methods:

- **`inner`** — intersection of all indices. The output length equals
  the number of positions present in every input. Zero-copy where
  possible.
- **`outer`** — union of all indices. Missing positions in any input
  become NaN; the value dtype is promoted to float64 to hold NaN.
- **`forward`** — union of all indices, then forward-fill each series.
  NaN is preserved BEFORE a series starts; from the first present
  entry onward, gaps are filled with the previous value. Symmetric
  across inputs.

Values may be 1D (a single series per position) or 2D (a panel:
`(n_positions, k_features)`). All-numeric numpy dtypes are supported.

No pandas dep — inputs are `(index, values)` tuples of numpy arrays,
outputs are the same shape. A `.to_dict(names)` convenience on the
result zips arrays to a name tuple for keyword access.

Design: docs/kernels/data/align.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kuant._validation import (
    require_1d,
    require_dep,
    require_equal_length,
    require_range,
    warn_kuant,
)
from kuant.errors import KuantNumericWarning, KuantShapeError, KuantValueError

_ALLOWED_METHODS = ("inner", "outer", "forward")


@dataclass
class AlignResult:
    """Aligned arrays on a shared index.

    Attributes
    ----------
    index : 1D np.ndarray
        The shared index every array is aligned on. Sorted ascending.
    arrays : tuple[np.ndarray, ...]
        Aligned arrays in the input order. Each has `len(...) == len(index)`
        on axis 0. 2D inputs preserve their column count.
    method : str
        The join method that produced this result.
    original_lengths : tuple[int, ...]
        Length of each input's index BEFORE alignment. Useful for
        reporting how much of the raw data survived the join.
    """

    index: np.ndarray
    arrays: tuple[np.ndarray, ...]
    method: str
    original_lengths: tuple[int, ...]

    def to_dict(self, names) -> dict:
        """Zip the aligned arrays with a name tuple for keyword access."""
        names = tuple(names)
        if len(names) != len(self.arrays):
            raise KuantValueError(
                f"kuant.align.to_dict: got {len(names)} names for "
                f"{len(self.arrays)} arrays.  [KE-SHAPE-EQUAL-LEN]\n"
                f"  → Fix: pass one name per aligned array"
            )
        return dict(zip(names, self.arrays))

    def summary(self) -> str:
        parts = [
            "=== AlignResult ===",
            f"method:            {self.method}",
            f"shared length:     {len(self.index)}",
            f"per-input lengths: {self.original_lengths}",
            f"n_arrays:          {len(self.arrays)}",
        ]
        return "\n".join(parts)

    def to_parquet(self, path, names=None) -> None:
        """Write the aligned data to a parquet file.

        Parameters
        ----------
        path : str or path-like
            Output path. Existing files are overwritten.
        names : sequence of str, optional
            Column names for each aligned array. Defaults to
            `('arr0', 'arr1', ...)`. If any array is 2D, its columns
            are exploded to `{name}[0]`, `{name}[1]`, ...
            The shared index is always written as column `'index'`.

        Notes
        -----
        Requires `pyarrow`. Parquet is the preferred kuant output format:
        columnar, typed, compressed, cheap to append via row-group writes,
        and 5-20x faster to read than CSV on typical panels.

        Examples
        --------
        >>> r = align((idx_a, x), (idx_b, y), method="inner")             # doctest: +SKIP
        >>> r.to_parquet("aligned.parquet", names=("returns", "vol"))     # doctest: +SKIP
        """
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as e:
            require_dep(
                "pyarrow",
                kernel="align.to_parquet",
                install="pip install pyarrow",
                cause=e,
            )

        if names is None:
            names = tuple(f"arr{i}" for i in range(len(self.arrays)))
        else:
            names = tuple(names)
            if len(names) != len(self.arrays):
                raise KuantValueError(
                    f"kuant.align.to_parquet: got {len(names)} names for "
                    f"{len(self.arrays)} arrays.  [KE-SHAPE-EQUAL-LEN]\n"
                    f"  → Fix: pass one name per aligned array"
                )

        columns = {"index": pa.array(self.index)}
        for name, arr in zip(names, self.arrays):
            if arr.ndim == 1:
                columns[name] = pa.array(arr)
            else:
                for c in range(arr.shape[1]):
                    columns[f"{name}[{c}]"] = pa.array(arr[:, c])

        table = pa.table(columns)
        pq.write_table(table, path)


def _sort_pair(idx: np.ndarray, val: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (sorted_idx, sorted_val) sorted by idx ascending."""
    order = np.argsort(idx, kind="stable")
    return idx[order], val[order]


def _check_index_dtypes_compatible(indices: list[np.ndarray]) -> None:
    """Reject silently-incompatible index dtypes (e.g. int vs datetime64)."""
    kinds = {i.dtype.kind for i in indices}
    # Numeric kinds i/u/f are interconvertible enough; datetime is 'M'; str is 'U'/'S'.
    # Reject mixing numeric with datetime or numeric with string.
    numeric = {"i", "u", "f"}
    if kinds & numeric and (kinds - numeric):
        raise KuantShapeError(
            f"kuant.align: mixed index dtype kinds {sorted(kinds)}; refusing "
            f"to silently coerce numeric and non-numeric indices.  "
            f"[KE-SHAPE-EXPECTED]\n"
            f"  → Fix: cast every index to the same dtype before calling "
            f"(e.g. `idx.astype('datetime64[D]')` on both sides)"
        )


def align(*series_pairs, method: str = "inner") -> AlignResult:
    """Align 2+ (index, values) series pairs on a shared index.

    Parameters
    ----------
    *series_pairs : tuple[array, array]
        Each is a `(index, values)` pair. `index` is 1D. `values` is 1D
        with `len(values) == len(index)`, OR 2D with `values.shape[0]
        == len(index)`.
    method : {'inner', 'outer', 'forward'}
        - `'inner'`: intersect all indices; each output has that length.
        - `'outer'`: union all indices; missing positions become NaN.
          Value dtypes are promoted to float64 to hold NaN.
        - `'forward'`: union all indices, then forward-fill each series.
          Positions BEFORE a series's first observation stay NaN.

    Returns
    -------
    AlignResult
        `.index` — shared 1D array (sorted ascending).
        `.arrays` — tuple of aligned arrays in input order.
        `.method` — the join used.
        `.original_lengths` — per-input pre-align lengths.

    Notes
    -----
    - Duplicate entries in any input index raise `KuantValueError`.
      Aggregate duplicates upstream (e.g. group-by-sum) before calling.
    - Indices need not be pre-sorted; alignment sorts internally.
    - Mixing numeric indices with datetime64 or string indices is
      rejected — silent dtype coercion is a known source of bugs.
    - Empty intersection (`inner` with no shared positions) returns
      an `AlignResult` with `len(index) == 0` — no exception.

    Examples
    --------
    >>> import numpy as np
    >>> dates_a = np.array([1, 2, 3, 4])
    >>> vals_a = np.array([10.0, 20, 30, 40])
    >>> dates_b = np.array([2, 3, 4, 5])
    >>> vals_b = np.array([200.0, 300, 400, 500])
    >>> r = align((dates_a, vals_a), (dates_b, vals_b), method="inner")
    >>> r.index.tolist()
    [2, 3, 4]
    >>> r.arrays[0].tolist()
    [20.0, 30.0, 40.0]
    >>> r.arrays[1].tolist()
    [200.0, 300.0, 400.0]
    """
    if method not in _ALLOWED_METHODS:
        raise KuantValueError(
            f"kuant.align: 'method' must be one of {_ALLOWED_METHODS}, "
            f"got {method!r}.  [KE-VAL-RANGE]\n"
            f"  → Fix: pick one of {_ALLOWED_METHODS}"
        )
    require_range(
        len(series_pairs),
        "number of series_pairs",
        kernel="align",
        lo=2,
        hi=float("inf"),
    )

    # Validate each pair; extract sorted (idx, val) copies.
    indices: list[np.ndarray] = []
    values: list[np.ndarray] = []
    original_lengths: list[int] = []
    for i, pair in enumerate(series_pairs):
        if not (isinstance(pair, tuple) and len(pair) == 2):
            raise KuantValueError(
                f"kuant.align: series_pairs[{i}] must be a (index, values) "
                f"tuple, got {type(pair).__name__}.  [KE-VAL-RANGE]\n"
                f"  → Fix: pass tuples like `align((dates, x), (dates, y))`"
            )
        idx_raw, val_raw = pair
        idx = np.asarray(idx_raw)
        val = np.asarray(val_raw)
        require_1d(idx, f"series_pairs[{i}][0] (index)", kernel="align")

        # Values must be 1D or 2D and their leading axis must match idx.
        if val.ndim not in (1, 2):
            raise KuantShapeError(
                f"kuant.align: series_pairs[{i}][1] (values) must be 1D or "
                f"2D, got shape {val.shape}.  [KE-SHAPE-EXPECTED]\n"
                f"  → Fix: pass a 1D series or 2D panel of shape "
                f"(n_positions, k_features)"
            )
        require_equal_length(idx, f"index[{i}]", val, f"values[{i}]", kernel="align")

        original_lengths.append(int(idx.size))
        # Duplicate check + sort together.
        _check_duplicates(idx, f"series_pairs[{i}][0]")
        idx_sorted, val_sorted = _sort_pair(idx, val)
        indices.append(idx_sorted)
        values.append(val_sorted)

    _check_index_dtypes_compatible(indices)

    # Build the shared index.
    if method == "inner":
        shared = indices[0]
        for idx in indices[1:]:
            shared = np.intersect1d(shared, idx, assume_unique=True)
        if shared.size == 0:
            per_len = [len(idx) for idx in indices]
            warn_kuant(
                kernel="align",
                code="KW-ALIGN-EMPTY-INTERSECT",
                what=(
                    f"inner join produced an empty shared index; per-input "
                    f"lengths were {per_len} and none of the indices overlap"
                ),
                fix=(
                    "verify the inputs cover a common period, or switch to "
                    "method='outer' if disjoint coverage is intentional"
                ),
                category=KuantNumericWarning,
            )
    else:
        # union1d is used for both outer and forward.
        shared = indices[0]
        for idx in indices[1:]:
            shared = np.union1d(shared, idx)

    # Reindex each series onto `shared`.
    aligned: list[np.ndarray] = []
    for idx, val in zip(indices, values):
        aligned.append(_reindex(idx, val, shared, method))

    return AlignResult(
        index=shared,
        arrays=tuple(aligned),
        method=method,
        original_lengths=tuple(original_lengths),
    )


def _check_duplicates(idx: np.ndarray, name: str) -> None:
    if idx.size <= 1:
        return
    order = np.argsort(idx, kind="stable")
    sorted_idx = idx[order]
    dup_mask = sorted_idx[1:] == sorted_idx[:-1]
    if bool(dup_mask.any()):
        dup_pos = int(np.where(dup_mask)[0][0])
        raise KuantValueError(
            f"kuant.align: '{name}' contains duplicate entries; first "
            f"duplicate is {sorted_idx[dup_pos]!r}.  [KE-VAL-DUPLICATE]\n"
            f"  → Fix: dedupe or aggregate duplicate positions before "
            f"calling — one row per index position"
        )


def _reindex(
    idx: np.ndarray,
    val: np.ndarray,
    shared: np.ndarray,
    method: str,
) -> np.ndarray:
    """Place `val` (indexed by `idx`) onto `shared` under the given join."""
    if method == "inner":
        # Every entry in `shared` is present in `idx` by construction.
        positions = np.searchsorted(idx, shared)
        return val[positions]

    # For outer/forward, we need a NaN-fillable dtype.
    if val.dtype.kind not in "fc":
        out_dtype = np.float64
    else:
        out_dtype = val.dtype

    out_shape = (shared.size,) if val.ndim == 1 else (shared.size, val.shape[1])
    out = np.full(out_shape, np.nan, dtype=out_dtype)

    # Positions in `shared` that correspond to entries in `idx`.
    positions = np.searchsorted(shared, idx)
    if val.ndim == 1:
        out[positions] = val
    else:
        out[positions, :] = val

    if method == "outer":
        return out

    # method == "forward": forward-fill NaN positions.
    return _forward_fill(out)


def _forward_fill(arr: np.ndarray) -> np.ndarray:
    """Vectorized forward-fill along axis 0.

    Leading NaNs (before the first non-NaN) are preserved as NaN.
    """
    if arr.ndim == 1:
        mask = ~np.isnan(arr)
        if not bool(mask.any()):
            return arr
        # For each position, the index of the most recent True in `mask`.
        idx = np.where(mask, np.arange(arr.size), 0)
        idx = np.maximum.accumulate(idx)
        out = arr[idx]
        # Any leading positions before the first True must stay NaN.
        first_true = int(np.argmax(mask))
        out[:first_true] = np.nan
        return out

    # 2D: apply per column.
    out = arr.copy()
    for c in range(arr.shape[1]):
        out[:, c] = _forward_fill(arr[:, c])
    return out


__all__ = ["align", "AlignResult"]
