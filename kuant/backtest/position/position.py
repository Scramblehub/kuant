"""Position: per-symbol accounting.

One `Position` per symbol tracks:

- `size`: signed quantity currently held (+ long, - short)
- `avg_cost`: volume-weighted average entry price for the OPEN portion
  of the position
- `realized_pnl`: cumulative P&L from closes / reversals

Semantics use netting (one signed size per symbol) rather than long-
short segregation. When a fill adds to the position in the same
direction, `avg_cost` is re-weighted. When a fill reduces or flips the
position, the closed portion produces `realized_pnl` at the difference
between `fill_price` and `avg_cost`, and `avg_cost` for the remaining
size is preserved (or reset if the position flipped side).
"""

from __future__ import annotations

from dataclasses import dataclass

from kuant.errors import KuantValueError


@dataclass
class Position:
    """A per-symbol netting position.

    Attributes
    ----------
    symbol : str
    size : float
        Signed quantity. Positive = long, negative = short, zero =
        flat.
    avg_cost : float
        Volume-weighted average entry price for the currently-open
        portion. Undefined (0.0) when `size == 0`.
    realized_pnl : float
        Cumulative P&L from all closes and reversals.

    Examples
    --------
    >>> p = Position(symbol='XYZ')
    >>> p.apply_fill(size_filled=100.0, price=50.0)
    >>> p.size, round(p.avg_cost, 2)
    (100.0, 50.0)
    >>> p.apply_fill(size_filled=100.0, price=60.0)  # add
    >>> p.size, round(p.avg_cost, 2)
    (200.0, 55.0)
    >>> p.apply_fill(size_filled=-100.0, price=70.0)  # partial close
    >>> p.size, round(p.avg_cost, 2), round(p.realized_pnl, 2)
    (100.0, 55.0, 1500.0)
    """

    symbol: str
    size: float = 0.0
    avg_cost: float = 0.0
    realized_pnl: float = 0.0

    def apply_fill(self, size_filled: float, price: float) -> None:
        """Update the position with an executed fill.

        Parameters
        ----------
        size_filled : float
            Signed quantity filled. Positive = bought, negative = sold.
            Zero is a no-op. Must be finite.
        price : float
            The fill price. Must be finite and strictly positive.

        Notes
        -----
        Three mutually-exclusive cases:

        1. **Zero fill** or **starting from flat**: set size to
           `size_filled`, avg_cost to `price`.
        2. **Same direction as current position** (both long, both
           short): weighted-average avg_cost against the new size.
        3. **Reducing or flipping**: realize P&L on the closed
           quantity, keep avg_cost on the remainder, or reset avg_cost
           to `price` on the flipped remainder.
        """
        import math

        if not math.isfinite(size_filled):
            raise KuantValueError(
                f"kuant.Position.apply_fill: 'size_filled' must be "
                f"finite, got {size_filled}.  [KE-POS-SIZE-INVALID]\n"
                f"  → Fix: reject NaN sizes upstream in execute_fill or "
                f"clean the signal via kuant.edgecases.nanpolicies"
            )
        if not math.isfinite(price) or price <= 0.0:
            raise KuantValueError(
                f"kuant.Position.apply_fill: 'price' must be finite and "
                f"strictly positive, got {price}.  [KE-POS-PRICE-INVALID]\n"
                f"  → Fix: gate NaN prices at the liquidity layer "
                f"(execute_fill already returns reason='MISSING_DATE' or "
                f"'NO_LIQUIDITY' on unpriced fills)"
            )
        if size_filled == 0.0:
            return
        if self.size == 0.0:
            self.size = float(size_filled)
            self.avg_cost = float(price)
            return
        same_sign = (self.size > 0 and size_filled > 0) or (self.size < 0 and size_filled < 0)
        if same_sign:
            new_size = self.size + size_filled
            # Weighted average of costs.
            self.avg_cost = (self.avg_cost * self.size + price * size_filled) / new_size
            self.size = new_size
            return
        # Opposite sign: reducing, closing, or flipping.
        # The closed quantity is min(|new|, |current|) in absolute terms,
        # with sign carried by the CURRENT position side (we're closing
        # our position, so the realized P&L direction depends on
        # long-vs-short, not on the side of the incoming fill).
        abs_current = abs(self.size)
        abs_incoming = abs(size_filled)
        closed_qty = min(abs_current, abs_incoming)
        # Long position: profit if price > avg_cost. Short: profit if
        # avg_cost > price.
        side_sign = 1.0 if self.size > 0 else -1.0
        self.realized_pnl += side_sign * closed_qty * (price - self.avg_cost)
        remaining = abs_current - abs_incoming
        if remaining > 0:
            # Partial close; sign unchanged, avg_cost preserved.
            self.size = side_sign * remaining
            # avg_cost unchanged
            return
        if remaining == 0:
            # Fully closed to flat.
            self.size = 0.0
            self.avg_cost = 0.0
            return
        # Flipped past zero. Remaining absolute size on the OPPOSITE
        # side, entered at the fill price.
        flip_size = abs_incoming - abs_current
        self.size = -side_sign * flip_size
        self.avg_cost = float(price)

    def market_value(self, price: float) -> float:
        """Signed mark-to-market value of the open position."""
        return float(self.size) * float(price)

    def unrealized_pnl(self, price: float) -> float:
        """P&L if the position were closed at `price` right now."""
        if self.size == 0.0:
            return 0.0
        side_sign = 1.0 if self.size > 0 else -1.0
        return side_sign * abs(self.size) * (float(price) - self.avg_cost)

    def total_pnl(self, price: float) -> float:
        """Realized plus unrealized P&L."""
        return self.realized_pnl + self.unrealized_pnl(price)

    def summary(self) -> str:
        return (
            "=== Position ===\n"
            f"symbol:         {self.symbol}\n"
            f"size:           {self.size:+g}\n"
            f"avg_cost:       {self.avg_cost:.6f}\n"
            f"realized_pnl:   {self.realized_pnl:+.4f}"
        )


__all__ = ["Position"]
