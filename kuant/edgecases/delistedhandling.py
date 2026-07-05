"""Delisted-name handling for historical backtests.

Historical backtests routinely overstate returns because delisted
names either drop from the universe entirely (survivor bias) or hold
their last-known price forever ("phantom equity"). This module ships
the three primitives that fix each failure mode.

- **`zero_after_delist`** — after the delist row, prices become 0.
  Correct for bankruptcy-style delists where the equity was truly
  worthless. Aggressive; use for known bankruptcies only.
- **`hold_last_price`** — after the delist row, prices hold the last
  known value. Correct for merger/spin-off delists where the equity
  was acquired at the last close. Warns if the held span exceeds
  `max_hold_days` (default 20) — that's the phantom-equity pattern.
- **`full_recovery_check`** — universe-level sanity check. Given the
  full historical universe and a list of names known to have delisted,
  verify coverage. Low coverage → survivor bias flag.

Design: docs/kernels/edgecases/delistedhandling.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kuant._validation import (
    require_1d,
    require_positive,
    require_probability,
    warn_kuant,
)
from kuant.errors import KuantNumericWarning, KuantValueError


def zero_after_delist(prices, delist_position: int) -> np.ndarray:
    """Set every price at or after `delist_position` to 0.

    Parameters
    ----------
    prices : 1D array
        Raw price series. Length T.
    delist_position : int
        Row index of the delist event. `prices[delist_position:]` becomes 0.

    Returns
    -------
    1D np.ndarray of length T
        Prices with zeros after the delist boundary.

    Notes
    -----
    Aggressive — treats the equity as worthless from the delist row
    onward. Use `hold_last_price` for merger/spin-off delists where
    shareholders were paid out at the last close.

    Examples
    --------
    >>> import numpy as np
    >>> prices = np.array([100.0, 95, 90, 80, 70])
    >>> zero_after_delist(prices, 3).tolist()
    [100.0, 95.0, 90.0, 0.0, 0.0]
    """
    arr = np.asarray(prices, dtype=np.float64)
    require_1d(arr, "prices", kernel="zero_after_delist")
    n = arr.size
    _validate_delist_position(delist_position, n, kernel="zero_after_delist")
    out = arr.copy()
    out[delist_position:] = 0.0
    return out


def hold_last_price(
    prices,
    delist_position: int,
    max_hold_days: int = 20,
) -> np.ndarray:
    """Hold the last known price after the delist row.

    Parameters
    ----------
    prices : 1D array
        Raw price series. Length T.
    delist_position : int
        Row index of the delist event.
    max_hold_days : int, default 20
        Warn if the held span (T - delist_position) exceeds this. That
        pattern is "phantom equity" — an acquired stock's price never
        moves, artificially inflating any strategy that holds it.

    Returns
    -------
    1D np.ndarray of length T
        Prices, with all rows at or after `delist_position` set to
        `prices[delist_position - 1]` (or the first row if delist is
        at position 0).

    Warnings
    --------
    - `KuantNumericWarning` (`KW-PHANTOM-EQUITY`) if the held span
      exceeds `max_hold_days`.

    Examples
    --------
    >>> import numpy as np
    >>> prices = np.array([100.0, 95, 90, 80, 70])
    >>> hold_last_price(prices, 3, max_hold_days=100).tolist()
    [100.0, 95.0, 90.0, 90.0, 90.0]
    """
    arr = np.asarray(prices, dtype=np.float64)
    require_1d(arr, "prices", kernel="hold_last_price")
    n = arr.size
    _validate_delist_position(delist_position, n, kernel="hold_last_price")
    require_positive(max_hold_days, "max_hold_days", kernel="hold_last_price", kind="int")

    out = arr.copy()
    if delist_position == 0:
        # No prior known price — hold the first value (best we can do).
        held = float(arr[0])
    else:
        held = float(arr[delist_position - 1])
    out[delist_position:] = held

    hold_span = n - int(delist_position)
    if hold_span > int(max_hold_days):
        warn_kuant(
            kernel="hold_last_price",
            code="KW-PHANTOM-EQUITY",
            what=(
                f"held span {hold_span} exceeds max_hold_days "
                f"({max_hold_days}); phantom-equity risk — this stock's "
                f"price will not move for {hold_span} rows"
            ),
            fix=(
                "for M&A/spin-off delists, hold only through the payout "
                "settlement (~5-10 rows) then drop the name; for a "
                "true zero, use zero_after_delist instead"
            ),
            category=KuantNumericWarning,
        )
    return out


@dataclass
class RecoveryCheckResult:
    """Report from `full_recovery_check`.

    Attributes
    ----------
    coverage : float
        Fraction of `known_delisted` names present in `universe`.
    n_expected : int
        Length of `known_delisted`.
    n_found : int
        How many `known_delisted` names actually appeared in `universe`.
    missing : list
        Names in `known_delisted` NOT found in `universe`. Sorted.
    status : str
        `'clean'` if coverage >= tolerance; else `'survivor_bias'`.
    tolerance : float
        The threshold used to classify status.
    """

    coverage: float
    n_expected: int
    n_found: int
    missing: list
    status: str
    tolerance: float

    def summary(self) -> str:
        parts = [
            "=== RecoveryCheckResult ===",
            f"status:      {self.status}",
            f"coverage:    {self.coverage:.2%}  ({self.n_found}/{self.n_expected})",
            f"tolerance:   {self.tolerance:.2%}",
            f"missing:     {len(self.missing)} name(s)",
        ]
        if self.missing:
            preview = self.missing[:10]
            more = f" (+{len(self.missing) - 10} more)" if len(self.missing) > 10 else ""
            parts.append(f"  first: {preview}{more}")
        return "\n".join(parts)


def full_recovery_check(
    universe,
    known_delisted,
    tolerance: float = 0.9,
) -> RecoveryCheckResult:
    """Verify a historical universe includes the expected delisted names.

    Parameters
    ----------
    universe : 1D array
        Every name that ever appeared in your historical universe.
        Order doesn't matter; duplicates are handled.
    known_delisted : 1D array
        Names known to have delisted during the universe's span.
        The universe should contain most or all of these; low coverage
        implies the reconstruction dropped delisted names, i.e.
        survivor bias.
    tolerance : float in [0, 1], default 0.9
        Minimum acceptable coverage. Below this threshold the result's
        status is `'survivor_bias'`.

    Returns
    -------
    RecoveryCheckResult

    Warnings
    --------
    - `KuantNumericWarning` (`KW-SURVIVOR-BIAS`) if coverage < tolerance.
      Downstream backtests should be considered biased-upward until the
      universe reconstruction is fixed.

    Examples
    --------
    >>> import numpy as np
    >>> universe = np.array(["AAPL", "MSFT", "ENRN", "LEH"])
    >>> known_delisted = np.array(["ENRN", "LEH", "WCOM"])   # WCOM missing
    >>> r = full_recovery_check(universe, known_delisted)
    >>> r.coverage
    0.6666666666666666
    >>> r.missing
    ['WCOM']
    """
    universe_arr = np.asarray(universe)
    known_arr = np.asarray(known_delisted)
    require_1d(universe_arr, "universe", kernel="full_recovery_check")
    require_1d(known_arr, "known_delisted", kernel="full_recovery_check")
    require_probability(tolerance, "tolerance", kernel="full_recovery_check")

    if known_arr.size == 0:
        raise KuantValueError(
            "kuant.full_recovery_check: 'known_delisted' is empty; nothing "
            "to check.  [KE-VAL-RANGE]\n"
            "  → Fix: pass at least one known-delisted name (from a "
            "bankruptcy database, historical SP500 changes, etc.)"
        )

    universe_set = set(universe_arr.tolist())
    known_list = known_arr.tolist()
    found = [n for n in known_list if n in universe_set]
    missing = sorted([n for n in known_list if n not in universe_set])
    coverage = len(found) / len(known_list)
    status = "clean" if coverage >= tolerance else "survivor_bias"

    if status == "survivor_bias":
        warn_kuant(
            kernel="full_recovery_check",
            code="KW-SURVIVOR-BIAS",
            what=(
                f"coverage {coverage:.1%} below tolerance {tolerance:.1%}; "
                f"{len(missing)} of {len(known_list)} known-delisted names "
                f"missing from universe"
            ),
            fix=(
                "verify universe reconstruction is point-in-time; a "
                "missing bankruptcy set is the classic survivor-bias "
                "pattern that inflates backtested returns"
            ),
            category=KuantNumericWarning,
        )

    return RecoveryCheckResult(
        coverage=coverage,
        n_expected=len(known_list),
        n_found=len(found),
        missing=missing,
        status=status,
        tolerance=tolerance,
    )


# ---------- helpers -------------------------------------------------------


def _validate_delist_position(pos: int, n: int, *, kernel: str) -> None:
    """Reject non-integer, fractional, or out-of-bounds positions."""
    if isinstance(pos, bool) or not isinstance(pos, (int, np.integer)):
        # Accept float only if it's actually integral (e.g. np.float64(3.0)).
        if not isinstance(pos, (float, np.floating)) or pos != int(pos):
            raise KuantValueError(
                f"kuant.{kernel}: 'delist_position' must be an integer, "
                f"got {pos!r} of type {type(pos).__name__}.  "
                f"[KE-VAL-RANGE]\n"
                f"  → Fix: pass a 0-based row index"
            )
    p = int(pos)
    if p < 0 or p > n:
        raise KuantValueError(
            f"kuant.{kernel}: 'delist_position' ({p}) out of bounds for "
            f"prices of length {n}; valid range is [0, {n}].  "
            f"[KE-VAL-RANGE]\n"
            f"  → Fix: pass a row index in [0, len(prices)]"
        )


__all__ = [
    "zero_after_delist",
    "hold_last_price",
    "full_recovery_check",
    "RecoveryCheckResult",
]
