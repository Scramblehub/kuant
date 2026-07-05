"""Reference orchestrator over the correctness-first backtest primitives.

Wires `warmup`, `lifecycle`, `liquidity`, `fill`, and `position` into
one bar-driven simulation loop. Intentionally small (~200 lines).
Users who want fancier features (multi-strategy composition, event
callbacks, cross-asset margining, real-time hooks) build on top of the
primitives directly; this engine is a reference orchestrator, not a
framework.

Bar semantics
-------------
Orders emitted on bar `t` fill at bar `t`'s reference price (default:
the value at `cache.prices.loc[t, symbol]`). This is the "close-to-
close" convention. Strategies that need one-bar-ahead semantics
should lag their signal internally by one bar; a `mode="next_bar"`
opt-in is deferred to a later version.

Gating
------
Each order is checked against `cache.tradeable(t, sym)` before
submission. Symbols outside their lifecycle window are silently
skipped (recorded as `gated`), not passed to `execute_fill`. Symbols
missing a liquidity profile are skipped and recorded as `no_profile`.
Symbols whose price is NaN on the current bar are skipped and
recorded as `no_price`.

Terminal actions
----------------
The engine does NOT auto-close positions on delisting_date. A
strategy that holds through delisting will retain a NaN mark-to-
market from that point forward. A `terminal_actions=True` opt-in
that auto-liquidates per `SecurityLifecycle.terminal_action` is
deferred to v1.1.

Design: docs/kernels/backtest/engine/README.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from kuant._validation import require_dep, require_positive
from kuant.backtest.fill.order import Order
from kuant.backtest.fill.submit import submit_order
from kuant.backtest.position.portfolio import PortfolioState
from kuant.backtest.warmup.cache import WarmupCache
from kuant.errors import KuantShapeError, KuantValueError

StrategyFn = Callable[[WarmupCache, PortfolioState, pd.Timestamp], list[Order]]


# ---------- BacktestResult --------------------------------------------


@dataclass
class BacktestResult:
    """Snapshot of an engine run.

    Attributes
    ----------
    equity : pandas.DataFrame
        Per-bar snapshot with columns `cash`, `positions_value`,
        `total_value`, `unrealized_pnl`, `realized_pnl`. Indexed by the
        timestamps of the price panel.
    trades : pandas.DataFrame
        Per-order record with columns `timestamp`, `order_id`, `symbol`,
        `side`, `requested_size`, `fill_price`, `size_filled`,
        `slippage_bps`, `cost`, `status`, `reason`, `tag`. Includes
        gated / no-profile / no-price entries so a caller can reconcile
        every intent.
    portfolio_final : PortfolioState
        Final portfolio state (cash + positions dict). Realized P&L
        history lives on the per-symbol positions.
    initial_cash : float
    n_bars : int
    n_orders_seen : int
        Total orders returned by the strategy across all bars.
    n_orders_filled : int
        Orders that produced a fill with nonzero size_filled.
    n_orders_rejected : int
        Orders rejected by the liquidity / fill layer
        (BELOW_MIN_SIZE, NO_LIQUIDITY, MISSING_DATE).
    n_orders_gated : int
        Orders skipped before reaching the liquidity layer
        (tradeable_mask False, no profile registered, or NaN price).
    """

    equity: object  # pandas.DataFrame
    trades: object  # pandas.DataFrame
    portfolio_final: PortfolioState
    initial_cash: float
    n_bars: int
    n_orders_seen: int
    n_orders_filled: int
    n_orders_rejected: int
    n_orders_gated: int

    def summary(self) -> str:
        eq = self.equity
        if len(eq) == 0:
            return "=== BacktestResult ===\n(no bars)"
        first = float(eq["total_value"].iloc[0])
        last = float(eq["total_value"].iloc[-1])
        total_return = (last / first - 1.0) if first != 0 else float("nan")
        return (
            "=== BacktestResult ===\n"
            f"bars:            {self.n_bars}\n"
            f"orders seen:     {self.n_orders_seen}\n"
            f"filled:          {self.n_orders_filled}\n"
            f"rejected:        {self.n_orders_rejected}\n"
            f"gated:           {self.n_orders_gated}\n"
            f"initial cash:    {self.initial_cash:.4f}\n"
            f"final total:     {last:.4f}\n"
            f"total return:    {total_return:+.4%}"
        )

    def to_parquet(self, path) -> None:
        """Write equity + trades to two parquet files under `path`.

        `path` is treated as a directory prefix: `{path}_equity.parquet`
        and `{path}_trades.parquet` are written. Requires pyarrow.
        """
        try:
            import pyarrow as pa  # noqa: F401
            import pyarrow.parquet as pq
        except ImportError as e:
            require_dep(
                "pyarrow",
                kernel="BacktestResult.to_parquet",
                install="pip install pyarrow",
                cause=e,
            )
        import pyarrow as pa

        eq_str = str(path) + "_equity.parquet"
        tr_str = str(path) + "_trades.parquet"
        eq_out = self.equity.copy()
        eq_out["timestamp"] = [str(x) for x in eq_out.index]
        pq.write_table(pa.Table.from_pandas(eq_out, preserve_index=False), eq_str)
        tr_out = self.trades.copy()
        if "timestamp" in tr_out.columns:
            tr_out["timestamp"] = [str(x) for x in tr_out["timestamp"]]
        pq.write_table(pa.Table.from_pandas(tr_out, preserve_index=False), tr_str)


# ---------- run --------------------------------------------------------


_TRADE_COLUMNS = (
    "timestamp",
    "order_id",
    "symbol",
    "side",
    "requested_size",
    "fill_price",
    "size_filled",
    "slippage_bps",
    "cost",
    "status",
    "reason",
    "tag",
)


def _empty_trades_frame() -> pd.DataFrame:
    return pd.DataFrame({c: [] for c in _TRADE_COLUMNS})


def run(
    cache: WarmupCache,
    strategy: StrategyFn,
    *,
    liquidity_profiles: dict,
    fill_model,
    initial_cash: float,
    lifecycles: dict | None = None,
) -> BacktestResult:
    """Run a bar-driven backtest.

    Parameters
    ----------
    cache : WarmupCache
        Materialized from a `Warmup`. Its `prices` panel drives the bar
        iteration; its `.tradeable` / `.liquid` / `.universe` gates are
        consulted to skip orders on non-tradeable dates.
    strategy : callable
        `strategy(cache, portfolio_state, timestamp) -> list[Order]`.
        The state is passed for read access to current positions and
        cash; the strategy MUST NOT mutate it. Return an empty list to
        do nothing on a bar.
    liquidity_profiles : dict[str, LiquidityProfile]
        Symbol -> profile. Symbols with no profile registered are
        skipped at order time and marked `gated` with reason
        `NO_PROFILE`.
    fill_model : FillModel-like
        Any object exposing `compute_slippage(size, adv, side)`.
    initial_cash : float
        Starting cash balance. Passed straight to `PortfolioState`.
    lifecycles : dict[str, SecurityLifecycle], optional
        If provided, the engine calls `cache.tradeable(t, sym)` to gate
        orders. If `cache` was already built with lifecycle registration
        via `Warmup.add_lifecycles`, pass None here; the cache already
        knows the answer.

    Returns
    -------
    BacktestResult

    Notes
    -----
    Bar sequencing: orders emitted on bar `t` are filled at bar `t`'s
    reference price from `cache.prices.loc[t, symbol]`. Strategies
    needing one-bar-ahead semantics should lag their signal internally.
    """
    require_positive(initial_cash, "initial_cash", kernel="engine.run")
    if not hasattr(fill_model, "compute_slippage"):
        raise KuantValueError(
            f"kuant.engine.run: 'fill_model' must expose a "
            f"compute_slippage(size, adv, side) method, got "
            f"{type(fill_model).__name__}.  [KE-VAL-CONTRACT]\n"
            f"  → Fix: pass one of FlatSlippage / LinearImpact / "
            f"SquareRootImpact, or a custom class with a matching method"
        )
    if not isinstance(liquidity_profiles, dict):
        raise KuantShapeError(
            f"kuant.engine.run: 'liquidity_profiles' must be a dict, "
            f"got {type(liquidity_profiles).__name__}.  "
            f"[KE-SHAPE-EXPECTED]\n"
            f"  → Fix: pass a symbol-to-LiquidityProfile mapping"
        )
    if lifecycles is not None and not isinstance(lifecycles, dict):
        raise KuantShapeError(
            f"kuant.engine.run: 'lifecycles' must be a dict or None, "
            f"got {type(lifecycles).__name__}.  [KE-SHAPE-EXPECTED]\n"
            f"  → Fix: pass a symbol-to-SecurityLifecycle mapping"
        )

    prices = cache.prices
    if not isinstance(prices, pd.DataFrame):
        raise KuantShapeError(
            "kuant.engine.run: WarmupCache.prices must be a "
            "pandas.DataFrame.  [KE-SHAPE-EXPECTED]\n"
            "  → Fix: build the cache via Warmup on a DataFrame panel"
        )

    state = PortfolioState(cash=float(initial_cash))
    trade_rows: list[dict] = []
    equity_rows: list[dict] = []
    n_orders_seen = 0
    n_orders_filled = 0
    n_orders_rejected = 0
    n_orders_gated = 0

    for timestamp in prices.index:
        orders = strategy(cache, state, timestamp) or []
        for order in orders:
            n_orders_seen += 1
            sym = order.symbol
            # Gate 1: lifecycle tradeable window (cache-provided if
            # cache was built with lifecycles; else default True).
            if not cache.tradeable(timestamp, sym):
                n_orders_gated += 1
                trade_rows.append(_gated_row(timestamp, order, "GATED_LIFECYCLE"))
                continue
            # Gate 2: liquidity profile must be registered.
            profile = liquidity_profiles.get(sym)
            if profile is None:
                n_orders_gated += 1
                trade_rows.append(_gated_row(timestamp, order, "NO_PROFILE"))
                continue
            # Gate 3: reference price must exist and be finite.
            if sym not in prices.columns:
                n_orders_gated += 1
                trade_rows.append(_gated_row(timestamp, order, "SYMBOL_NOT_IN_PANEL"))
                continue
            ref_price = float(prices.loc[timestamp, sym])
            if not np.isfinite(ref_price) or ref_price <= 0:
                n_orders_gated += 1
                trade_rows.append(_gated_row(timestamp, order, "NO_PRICE"))
                continue
            # Submit through the standard fill path.
            report = submit_order(order, profile, price=ref_price, model=fill_model)
            state.apply_fill(report)
            trade_rows.append(_trade_row_from_report(timestamp, order, report))
            if report.fill.size_filled != 0.0:
                n_orders_filled += 1
            else:
                n_orders_rejected += 1

        # Mark to market at bar close using the SAME price row.
        # NaN prices propagate; symbols with an open position but no
        # column in `prices` raise. Users are responsible for panel
        # completeness for symbols they trade.
        prices_row = prices.loc[timestamp].to_dict()
        snap = state.mark_to_market(prices_row)
        equity_rows.append(
            {
                "cash": snap.cash,
                "positions_value": snap.positions_value,
                "total_value": snap.total_value,
                "unrealized_pnl": snap.unrealized_pnl,
                "realized_pnl": snap.realized_pnl,
            }
        )

    equity_df = pd.DataFrame(equity_rows, index=prices.index)
    if trade_rows:
        trades_df = pd.DataFrame(trade_rows)
    else:
        trades_df = _empty_trades_frame()

    return BacktestResult(
        equity=equity_df,
        trades=trades_df,
        portfolio_final=state,
        initial_cash=float(initial_cash),
        n_bars=int(len(prices)),
        n_orders_seen=int(n_orders_seen),
        n_orders_filled=int(n_orders_filled),
        n_orders_rejected=int(n_orders_rejected),
        n_orders_gated=int(n_orders_gated),
    )


def _gated_row(timestamp, order: Order, reason: str) -> dict:
    return {
        "timestamp": timestamp,
        "order_id": order.order_id,
        "symbol": order.symbol,
        "side": int(order.side.value),
        "requested_size": float(order.size),
        "fill_price": float("nan"),
        "size_filled": 0.0,
        "slippage_bps": 0.0,
        "cost": 0.0,
        "status": "gated",
        "reason": reason,
        "tag": order.tag,
    }


def _trade_row_from_report(timestamp, order: Order, report) -> dict:
    return {
        "timestamp": timestamp,
        "order_id": order.order_id,
        "symbol": order.symbol,
        "side": int(order.side.value),
        "requested_size": float(order.size),
        "fill_price": float(report.fill.price),
        "size_filled": float(report.fill.size_filled),
        "slippage_bps": float(report.fill.slippage_bps),
        "cost": float(report.fill.cost),
        "status": report.status.value,
        "reason": report.fill.reason,
        "tag": order.tag,
    }


__all__ = ["BacktestResult", "StrategyFn", "run"]
