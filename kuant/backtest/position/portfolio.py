"""PortfolioState: cash + positions across all symbols.

Top-level accounting for a backtest run. Holds:

- `cash`: available cash balance
- `positions`: dict mapping symbol to `Position`

Consumes `FillReport` objects from `kuant.backtest.fill.submit_order`,
updating both cash and the per-symbol position atomically. Provides
`total_value(prices)` and `mark_to_market(prices)` for producing an
equity curve.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from kuant.backtest.fill.submit import FillReport
from kuant.backtest.position.position import Position
from kuant.errors import KuantValueError


@dataclass
class EquitySnapshot:
    """Marked-to-market portfolio state at one point in time.

    Attributes
    ----------
    cash : float
    positions_value : float
        Sum of `position.market_value(price)` across all held symbols.
    total_value : float
        `cash + positions_value`.
    n_positions : int
        Number of symbols with nonzero size.
    unrealized_pnl : float
        Sum of unrealized P&L across all held positions.
    realized_pnl : float
        Sum of realized P&L across all positions (including flat ones).
    """

    cash: float
    positions_value: float
    total_value: float
    n_positions: int
    unrealized_pnl: float
    realized_pnl: float

    def summary(self) -> str:
        return (
            "=== EquitySnapshot ===\n"
            f"cash:              {self.cash:.4f}\n"
            f"positions_value:   {self.positions_value:.4f}\n"
            f"total_value:       {self.total_value:.4f}\n"
            f"n_positions:       {self.n_positions}\n"
            f"unrealized_pnl:    {self.unrealized_pnl:+.4f}\n"
            f"realized_pnl:      {self.realized_pnl:+.4f}"
        )


@dataclass
class PortfolioState:
    """Cash + all positions.

    Attributes
    ----------
    cash : float
        Available cash balance. Can go negative if the engine allows
        leverage; kuant does not enforce a non-negative constraint at
        this layer (that's an engine-level policy).
    positions : dict[str, Position]
        Symbol to Position. Positions with `size == 0` are retained
        (so realized_pnl history survives).

    Examples
    --------
    >>> from kuant.backtest.fill import Order, OrderSide, OrderType, FillReport, OrderStatus
    >>> from kuant.backtest.liquidity import FillResult
    >>> ps = PortfolioState(cash=100_000.0)
    >>> fake_fill = FillResult(
    ...     price=50.0, size_filled=100.0, size_rejected=0.0,
    ...     slippage_bps=0.0, reason='OK', cost=5000.0,
    ... )
    >>> report = FillReport(
    ...     order_id=1, symbol='XYZ',
    ...     status=OrderStatus.FILLED, fill=fake_fill,
    ... )
    >>> ps.apply_fill(report)
    >>> ps.cash
    95000.0
    >>> ps.positions['XYZ'].size
    100.0
    """

    cash: float = 0.0
    positions: dict = field(default_factory=dict)

    def _get_or_create(self, symbol: str) -> Position:
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        return self.positions[symbol]

    def apply_fill(self, report: FillReport) -> None:
        """Update cash and the per-symbol position from a FillReport.

        Rejected fills (status REJECTED, size_filled == 0) are no-ops.
        Cash is debited on buys and credited on sells; the fill's
        `size_filled` carries the sign.
        """
        import math

        fill = report.fill
        if fill.size_filled == 0.0:
            return
        if not math.isfinite(fill.price):
            raise KuantValueError(
                f"kuant.PortfolioState.apply_fill: fill has non-finite "
                f"price ({fill.price}) but size_filled={fill.size_filled} "
                f"is nonzero; cash would become NaN.  "
                f"[KE-PORTFOLIO-FILL-PRICE-INVALID]\n"
                f"  → Fix: liquidity.execute_fill sets price=NaN only on "
                f"rejected fills (size_filled=0). A hand-built FillReport "
                f"or a custom model violated this invariant"
            )
        # Cash: on a buy (size_filled > 0), cash decreases by fill_price *
        # size_filled. On a sell, cash increases. Sign handles both.
        self.cash -= fill.size_filled * fill.price
        pos = self._get_or_create(report.symbol)
        pos.apply_fill(size_filled=fill.size_filled, price=fill.price)

    def total_value(self, prices: dict) -> float:
        """Cash plus mark-to-market value of held positions.

        Parameters
        ----------
        prices : dict[str, float]
            Symbol to reference price. Symbols with a nonzero position
            but no entry in `prices` raise a hard error; use NaN to
            explicitly mark a symbol as unpriced without crashing.

        Notes
        -----
        NaN in prices leaves the position's contribution as NaN, which
        propagates into the total. This mirrors the lifecycle
        semantics: NaN means "unpriced today," not "worth zero."
        """
        total = float(self.cash)
        for sym, pos in self.positions.items():
            if pos.size == 0.0:
                continue
            if sym not in prices:
                raise KuantValueError(
                    f"kuant.PortfolioState.total_value: symbol "
                    f"{sym!r} has an open position but no price "
                    f"in `prices`.  [KE-VAL-SCHEMA]\n"
                    f"  → Fix: include the symbol in prices (pass NaN "
                    f"to explicitly mark unpriced, otherwise your "
                    f"equity curve is understating exposure)"
                )
            total += pos.market_value(prices[sym])
        return total

    def mark_to_market(self, prices: dict) -> EquitySnapshot:
        """Snapshot cash + positions valued at `prices`."""
        positions_value = 0.0
        unrealized = 0.0
        realized = 0.0
        n_open = 0
        for sym, pos in self.positions.items():
            realized += pos.realized_pnl
            if pos.size == 0.0:
                continue
            if sym not in prices:
                raise KuantValueError(
                    f"kuant.PortfolioState.mark_to_market: symbol "
                    f"{sym!r} has an open position but no price in "
                    f"`prices`.  [KE-VAL-SCHEMA]\n"
                    f"  → Fix: include the symbol in prices (pass NaN "
                    f"if you want to accept the position as unpriced)"
                )
            price = float(prices[sym])
            positions_value += pos.market_value(price)
            unrealized += pos.unrealized_pnl(price)
            n_open += 1
        return EquitySnapshot(
            cash=float(self.cash),
            positions_value=positions_value,
            total_value=float(self.cash) + positions_value,
            n_positions=n_open,
            unrealized_pnl=unrealized,
            realized_pnl=realized,
        )

    def summary(self) -> str:
        n_open = sum(1 for p in self.positions.values() if p.size != 0.0)
        realized = sum(p.realized_pnl for p in self.positions.values())
        return (
            "=== PortfolioState ===\n"
            f"cash:              {self.cash:.4f}\n"
            f"n symbols tracked: {len(self.positions)}\n"
            f"n open positions:  {n_open}\n"
            f"realized_pnl sum:  {realized:+.4f}"
        )


__all__ = ["EquitySnapshot", "PortfolioState"]
