"""Corporate-action-adjusted price series.

Every historical price series needs adjustment for splits (always) and
dividends (mode-dependent). This kernel produces the two conventions
users actually want:

- **`'split_only'`** — split-adjusted prices. Level series that
  compares like-for-like across a split boundary. Trading strategies
  usually want this for entry/exit logic.
- **`'total_return'`** — split-adjusted AND dividend-reinvested.
  Suitable for return computation and long-horizon performance
  comparison. Dividends are reinvested at the ex-dividend close.

Two adjustment styles for `'total_return'`:

- **backward** (default): historical prices are scaled DOWN so the
  most recent price is unchanged. Preferred for live tools where you
  want today's price to match today's traded price.
- **forward**: current prices are scaled UP so the oldest price is
  unchanged. Preferred for research where you want the earliest
  historical print to be canonical.

Design: docs/kernels/data/corpaction.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kuant._validation import (
    require_1d,
    require_dep,
    require_equal_length,
    warn_kuant,
)
from kuant.errors import KuantNumericWarning, KuantValueError

_ALLOWED_MODES = ("split_only", "total_return")
_ALLOWED_DIRECTIONS = ("backward", "forward")


@dataclass
class CorpActionResult:
    """Adjusted price series + adjustment metadata.

    Attributes
    ----------
    prices : 1D np.ndarray
        Adjusted price series. Same length as the input `prices`.
    cumulative_factor : 1D np.ndarray
        Per-row multiplicative adjustment applied. `prices_adj = prices *
        cumulative_factor` (backward) or `prices_adj = prices *
        cumulative_factor` (forward, but starting at 1.0 on the first row).
    mode : str
        `'split_only'` or `'total_return'`.
    direction : str
        `'backward'` or `'forward'`. Ignored for `'split_only'`.
    n_splits_applied : int
    n_dividends_applied : int
    """

    prices: np.ndarray
    cumulative_factor: np.ndarray
    mode: str
    direction: str
    n_splits_applied: int
    n_dividends_applied: int

    def summary(self) -> str:
        parts = [
            "=== CorpActionResult ===",
            f"mode:                 {self.mode}",
            f"direction:            {self.direction}",
            f"n_splits_applied:     {self.n_splits_applied}",
            f"n_dividends_applied:  {self.n_dividends_applied}",
            f"length:               {len(self.prices)}",
        ]
        return "\n".join(parts)

    def to_parquet(self, path) -> None:
        """Write the adjusted series to parquet.

        Columns: prices, cumulative_factor. Requires `pyarrow`.
        """
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as e:
            require_dep(
                "pyarrow",
                kernel="corpaction.to_parquet",
                install="pip install pyarrow",
                cause=e,
            )
        cols = {
            "prices": pa.array(self.prices),
            "cumulative_factor": pa.array(self.cumulative_factor),
        }
        pq.write_table(pa.table(cols), path)


def _resolve_events(
    positions: np.ndarray,
    values: np.ndarray,
    name: str,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Coerce (positions, values) into aligned sorted arrays."""
    if positions is None:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    pos = np.asarray(positions)
    val = np.asarray(values, dtype=np.float64)
    require_1d(pos, f"{name}_positions", kernel="corpaction")
    require_1d(val, f"{name}_values", kernel="corpaction")
    require_equal_length(pos, f"{name}_positions", val, f"{name}_values", kernel="corpaction")
    if pos.dtype.kind not in "iu":
        raise KuantValueError(
            f"kuant.corpaction: '{name}_positions' must be integer indices "
            f"into the price array, got dtype {pos.dtype}.  [KE-VAL-RANGE]\n"
            f"  → Fix: pass row indices (0-based) as int64; e.g. "
            f"`np.searchsorted(dates, event_dates)`"
        )
    if pos.size == 0:
        return pos.astype(np.int64), val
    if pos.min() < 0 or pos.max() >= n:
        raise KuantValueError(
            f"kuant.corpaction: '{name}_positions' contains out-of-bounds "
            f"indices for prices of length {n}; got range "
            f"[{int(pos.min())}, {int(pos.max())}].  [KE-VAL-RANGE]\n"
            f"  → Fix: filter events to those within the price series's index"
        )
    # Sort by position ascending.
    order = np.argsort(pos, kind="stable")
    return pos[order].astype(np.int64), val[order]


def corpaction(
    prices,
    split_positions=None,
    split_ratios=None,
    dividend_positions=None,
    dividend_amounts=None,
    mode: str = "total_return",
    direction: str = "backward",
) -> CorpActionResult:
    """Adjust a price series for splits and (optionally) dividends.

    Parameters
    ----------
    prices : 1D array
        Raw close prices, one row per period. Length T.
    split_positions : 1D int array, optional
        Row indices at which splits take effect. Each index is the
        FIRST row on which the new-adjusted price should apply
        (typically the ex-split date).
    split_ratios : 1D float array, optional
        Split ratios aligned with `split_positions`. A 2-for-1 split
        is `ratio=2.0` (shares double, price halves). A 1-for-10
        reverse split is `ratio=0.1`.
    dividend_positions : 1D int array, optional
        Row indices at which dividends are ex-date. Only used if
        `mode='total_return'`.
    dividend_amounts : 1D float array, optional
        Cash dividend per share, aligned with `dividend_positions`.
    mode : {'total_return', 'split_only'}, default 'total_return'
        - `'split_only'`: split-adjusted price (dividend inputs ignored).
        - `'total_return'`: split-adjusted AND dividend-reinvested price.
    direction : {'backward', 'forward'}, default 'backward'
        - `'backward'`: scale historical prices so the most recent
          price is unchanged. Recommended for live-price contexts.
        - `'forward'`: scale future prices so the oldest price is
          unchanged. Recommended for research contexts.

    Returns
    -------
    CorpActionResult
        `.prices` — adjusted series, same length as input.
        `.cumulative_factor` — per-row multiplicative adjustment applied.
        `.mode`, `.direction`, `.n_splits_applied`, `.n_dividends_applied`.

    Warnings
    --------
    - `KuantNumericWarning` if any split ratio is > 100 or < 0.001;
      that's usually a typo signature in vendor split files.

    Notes
    -----
    Split ratio convention: `ratio = new_shares / old_shares`. A 2-for-1
    split (shares double) is `ratio=2.0`; the price BEFORE the ex-date
    is divided by 2 in backward mode.

    Examples
    --------
    >>> import numpy as np
    >>> # 5 daily closes; a 2-for-1 split at position 2.
    >>> raw = np.array([100.0, 100, 50, 51, 52])
    >>> r = corpaction(
    ...     raw,
    ...     split_positions=[2],
    ...     split_ratios=[2.0],
    ...     mode='split_only',
    ...     direction='backward',
    ... )
    >>> r.prices.tolist()   # historical prices halved to match post-split scale
    [50.0, 50.0, 50.0, 51.0, 52.0]
    """
    if mode not in _ALLOWED_MODES:
        raise KuantValueError(
            f"kuant.corpaction: 'mode' must be one of {_ALLOWED_MODES}, "
            f"got {mode!r}.  [KE-VAL-RANGE]\n"
            f"  → Fix: pick one of {_ALLOWED_MODES}"
        )
    if direction not in _ALLOWED_DIRECTIONS:
        raise KuantValueError(
            f"kuant.corpaction: 'direction' must be one of "
            f"{_ALLOWED_DIRECTIONS}, got {direction!r}.  [KE-VAL-RANGE]\n"
            f"  → Fix: pick one of {_ALLOWED_DIRECTIONS}"
        )

    prices_arr = np.asarray(prices, dtype=np.float64)
    require_1d(prices_arr, "prices", kernel="corpaction")
    n = prices_arr.size
    if n == 0:
        return CorpActionResult(
            prices=prices_arr.copy(),
            cumulative_factor=np.empty(0, dtype=np.float64),
            mode=mode,
            direction=direction,
            n_splits_applied=0,
            n_dividends_applied=0,
        )

    # Splits mut-ex on inputs.
    if (split_positions is None) != (split_ratios is None):
        raise KuantValueError(
            "kuant.corpaction: 'split_positions' and 'split_ratios' must "
            "be supplied together (both or neither).  [KE-VAL-MUTEX]\n"
            "  → Fix: pass both arrays, or leave both as None"
        )
    if (dividend_positions is None) != (dividend_amounts is None):
        raise KuantValueError(
            "kuant.corpaction: 'dividend_positions' and 'dividend_amounts' "
            "must be supplied together (both or neither).  [KE-VAL-MUTEX]\n"
            "  → Fix: pass both arrays, or leave both as None"
        )

    split_pos, split_ratio = _resolve_events(split_positions, split_ratios, "split", n)
    div_pos, div_amt = _resolve_events(dividend_positions, dividend_amounts, "dividend", n)

    if split_ratio.size and float(split_ratio.min()) <= 0.0:
        bad = int(np.argmin(split_ratio))
        raise KuantValueError(
            f"kuant.corpaction: split ratio at event {bad} = "
            f"{float(split_ratio[bad])} is non-positive; a legitimate "
            f"split ratio is strictly positive (2-for-1 forward is 2.0, "
            f"1-for-10 reverse is 0.1).  [KE-CORP-SPLIT-NONPOSITIVE]\n"
            f"  → Fix: drop the malformed event or repair its sign; a "
            f"zero ratio would produce division by zero in backward mode "
            f"and a negative ratio flips price sign"
        )

    # Sanity warning on outlandish split ratios.
    if split_ratio.size and (float(split_ratio.max()) > 100.0 or float(split_ratio.min()) < 0.001):
        warn_kuant(
            kernel="corpaction",
            code="KW-SPLIT-EXTREME",
            what=(
                f"split ratios span [{float(split_ratio.min()):.4g}, "
                f"{float(split_ratio.max()):.4g}]; values > 100 or < 0.001 "
                f"are the typical typo signature in vendor split files"
            ),
            fix=(
                "verify the ratio convention: 2-for-1 forward split is "
                "ratio=2.0 (not 0.5); 1-for-10 reverse split is ratio=0.1"
            ),
            category=KuantNumericWarning,
        )

    # Split-only mode: dividends are dropped even if supplied.
    n_div_applied = 0 if mode == "split_only" else int(div_pos.size)

    # Build the per-row cumulative adjustment factor.
    #
    # Backward direction: work from the end. Each event's factor is
    # applied to every row PRIOR to that event.
    #   - split at position i with ratio r: rows [0..i-1] get factor 1/r.
    #   - dividend at position i with amount d, close c_i (ex-div close):
    #       total-return factor for rows [0..i-1] is (c_i - d) / c_i.
    #
    # Forward direction: work from the start. Each event's factor is
    # applied to every row AT OR AFTER that event.
    #   - split at i, ratio r: rows [i..] get factor r.
    #   - dividend at i, amount d, close c_i: rows [i..] get factor
    #       c_i / (c_i - d).

    factor = np.ones(n, dtype=np.float64)

    if direction == "backward":
        # Apply from newest to oldest so cumulative factor multiplies.
        # Combine splits + (optional) dividends into event lists sorted
        # by position descending.
        events: list[tuple[int, float]] = []
        for i, r in zip(split_pos, split_ratio):
            events.append((int(i), 1.0 / float(r)))
        if mode == "total_return":
            for i, d in zip(div_pos, div_amt):
                c_i = float(prices_arr[i])
                if c_i - d <= 0:
                    # Degenerate: dividend >= price → skip with warning.
                    warn_kuant(
                        kernel="corpaction",
                        code="KW-DIV-DEGENERATE",
                        what=(
                            f"dividend {d:.4g} at row {int(i)} >= close "
                            f"{c_i:.4g}; total-return factor would be "
                            f"non-positive, skipping this dividend"
                        ),
                        fix=(
                            "verify dividend/close units and currency; "
                            "check for a data-vendor decimal-shift bug"
                        ),
                        category=KuantNumericWarning,
                    )
                    continue
                events.append((int(i), (c_i - d) / c_i))
        events.sort(key=lambda e: e[0], reverse=True)
        for pos, f in events:
            factor[:pos] *= f
    else:
        # Forward direction.
        events: list[tuple[int, float]] = []
        for i, r in zip(split_pos, split_ratio):
            events.append((int(i), float(r)))
        if mode == "total_return":
            for i, d in zip(div_pos, div_amt):
                c_i = float(prices_arr[i])
                if c_i - d <= 0:
                    warn_kuant(
                        kernel="corpaction",
                        code="KW-DIV-DEGENERATE",
                        what=(
                            f"dividend {d:.4g} at row {int(i)} >= close "
                            f"{c_i:.4g}; total-return factor would be "
                            f"non-positive, skipping this dividend"
                        ),
                        fix=("verify dividend/close units and currency"),
                        category=KuantNumericWarning,
                    )
                    continue
                events.append((int(i), c_i / (c_i - d)))
        events.sort(key=lambda e: e[0])
        for pos, f in events:
            factor[pos:] *= f

    adjusted = prices_arr * factor
    return CorpActionResult(
        prices=adjusted,
        cumulative_factor=factor,
        mode=mode,
        direction=direction,
        n_splits_applied=int(split_pos.size),
        n_dividends_applied=n_div_applied,
    )


__all__ = ["corpaction", "CorpActionResult"]
