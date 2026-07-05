"""FillModel implementations: constant-bps, linear impact, square-root impact.

A `FillModel` answers the question "what slippage does this order pay,
given its size relative to today's volume?" Slippage is returned as a
signed fraction of the reference price. Positive slippage widens the
fill in the direction of the order (buys pay more, sells receive less);
negative slippage is unusual but permitted for models that reward
liquidity-providing orders.

Three concrete models ship in v1:

- **`FlatSlippage(bps)`**: constant per-order slippage, no size
  dependence. Matches the behaviour of a typical backtest engine's
  `slippage` parameter. Cheap and famous; wrong for anything not tiny.
- **`LinearImpact(k)`**: slippage scales linearly with participation
  rate `size / adv`. Reasonable first-cut for mid-size orders. Coefficient
  `k` calibrates against historical fill data (typical range 5-25 bps
  at 10% ADV).
- **`SquareRootImpact(k)`**: slippage scales as `sqrt(size / adv)`.
  The Almgren-Chriss form; matches the empirical literature on large
  meta-orders. Coefficient `k` is again calibrated; a common starting
  point is 10-30 bps at unity participation.

Users writing their own model should expose a `.compute_slippage(size,
adv, side)` method with the same signature. `size` and `adv` share
units (typically shares); `side` is `+1` for buy, `-1` for sell.

Design: docs/kernels/backtest/liquidity/models.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from kuant._validation import require_nonnegative
from kuant.errors import KuantValueError


@dataclass(frozen=True)
class FlatSlippage:
    """Constant per-order slippage in basis points, no size dependence.

    Parameters
    ----------
    bps : float
        Slippage in basis points (10000 bps = 100%). Must be >= 0.

    Notes
    -----
    Behaviour matches the `slippage` parameter of a typical backtest
    engine: a $1000 order and a $10M order pay the same fill haircut,
    which is fine for tiny orders and wildly wrong for anything at
    scale. Include this model as a baseline, not as a realistic default.
    """

    bps: float

    def __post_init__(self) -> None:
        require_nonnegative(self.bps, "bps", kernel="FlatSlippage")

    def compute_slippage(self, size: float, adv: float, side: int) -> float:
        """Slippage as a fraction of price.

        Parameters
        ----------
        size : float
            Order size in the same units as ADV. Ignored here.
        adv : float
            Average daily volume. Ignored here.
        side : int
            +1 for buy, -1 for sell. Ignored here; slippage is symmetric.

        Returns
        -------
        float
        """
        _ = size, adv, side  # unused
        return self.bps / 10_000.0


@dataclass(frozen=True)
class LinearImpact:
    """Slippage scales linearly with participation rate.

    Parameters
    ----------
    k : float
        Slippage coefficient in basis points at 100% participation.
        A `k` of 20 means an order equal to today's ADV pays 20 bps.
        A `k` of 20 at 10% participation pays 2 bps.

    Notes
    -----
    Model form:
        slippage_frac = (k / 10_000) * (size / adv)

    Reasonable first-cut for mid-size orders that don't blow through
    the depth of book. For very large meta-orders, the empirical
    literature prefers `SquareRootImpact`.
    """

    k: float

    def __post_init__(self) -> None:
        require_nonnegative(self.k, "k", kernel="LinearImpact")

    def compute_slippage(self, size: float, adv: float, side: int) -> float:
        _ = side  # symmetric in this model
        if adv <= 0:
            raise KuantValueError(
                f"kuant.LinearImpact.compute_slippage: 'adv' must be "
                f"positive to compute participation rate, got {adv}.  "
                f"[KE-VAL-POSITIVE]\n"
                f"  → Fix: guard the caller against zero-volume bars "
                f"(use `liquidity_mask` first) or set adv to a floor"
            )
        return (self.k / 10_000.0) * (abs(size) / adv)


@dataclass(frozen=True)
class SquareRootImpact:
    """Almgren-Chriss square-root price-impact model.

    Parameters
    ----------
    k : float
        Slippage coefficient in basis points at 100% participation.
        Empirically calibrated to historical meta-order fill data;
        published ranges span roughly 10-30 bps at unity participation
        for liquid US equities.

    Notes
    -----
    Model form:
        slippage_frac = (k / 10_000) * sqrt(size / adv)

    The dominant empirical model for temporary market impact of large
    orders. Concave in size means each additional share pays less
    incremental slippage than the last, which reflects real order-book
    behaviour: aggressive fills sweep the top of book cheaply and pay
    exponentially more for the tail.
    """

    k: float

    def __post_init__(self) -> None:
        require_nonnegative(self.k, "k", kernel="SquareRootImpact")

    def compute_slippage(self, size: float, adv: float, side: int) -> float:
        _ = side
        if adv <= 0:
            raise KuantValueError(
                f"kuant.SquareRootImpact.compute_slippage: 'adv' must "
                f"be positive to compute participation rate, got "
                f"{adv}.  [KE-VAL-POSITIVE]\n"
                f"  → Fix: guard the caller against zero-volume bars "
                f"(use `liquidity_mask` first) or set adv to a floor"
            )
        return (self.k / 10_000.0) * math.sqrt(abs(size) / adv)


__all__ = ["FlatSlippage", "LinearImpact", "SquareRootImpact"]
