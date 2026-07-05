"""Tests for kuant.backtest.engine.run."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from kuant.backtest.engine import BacktestResult, run
from kuant.backtest.fill import Order, OrderSide
from kuant.backtest.lifecycle import SecurityLifecycle, TerminalAction
from kuant.backtest.liquidity import (
    FlatSlippage,
    LiquidityProfile,
)
from kuant.backtest.warmup import Warmup
from kuant.errors import KuantShapeError, KuantValueError


# ---------- fixtures ---------------------------------------------------


def _panel(n: int = 50, symbols=("A", "B"), start_price: float = 100.0) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    data = {}
    rng = np.random.default_rng(42)
    for i, sym in enumerate(symbols):
        drift = 0.0005 + 0.0005 * i
        noise = rng.normal(0, 0.005, n)
        rets = drift + noise
        data[sym] = start_price * np.cumprod(1 + rets)
    return pd.DataFrame(data, index=idx)


def _profile(sym: str, idx, adv: float = 1_000_000.0) -> LiquidityProfile:
    return LiquidityProfile(
        symbol=sym,
        adv_series=pd.Series([adv] * len(idx), index=idx, dtype=float),
        min_size=1.0,
        max_participation=0.10,
    )


def _buy_A_once(cache, state, timestamp):
    """Buy 100 shares of A on the first bar, then hold forever."""
    if state.positions.get("A") is not None and state.positions["A"].size != 0.0:
        return []
    return [
        Order(
            symbol="A",
            side=OrderSide.BUY,
            size=100.0,
            timestamp=(timestamp.date() if hasattr(timestamp, "date") else timestamp),
            tag="test",
        )
    ]


# ---------- basic run + shape ----------------------------------------


def test_run_returns_backtest_result():
    prices = _panel()
    idx = prices.index
    profiles = {"A": _profile("A", idx), "B": _profile("B", idx)}
    w = Warmup(prices, mode="eager")
    cache = w.materialize()
    r = run(
        cache,
        strategy=lambda cache, state, t: [],
        liquidity_profiles=profiles,
        fill_model=FlatSlippage(bps=0),
        initial_cash=100_000.0,
    )
    assert isinstance(r, BacktestResult)
    assert r.n_bars == len(prices)
    assert r.n_orders_seen == 0
    assert r.equity.shape == (len(prices), 5)


def test_run_hold_cash_flat_equity():
    prices = _panel()
    idx = prices.index
    profiles = {"A": _profile("A", idx), "B": _profile("B", idx)}
    w = Warmup(prices, mode="eager")
    cache = w.materialize()
    r = run(
        cache,
        strategy=lambda cache, state, t: [],
        liquidity_profiles=profiles,
        fill_model=FlatSlippage(bps=0),
        initial_cash=100_000.0,
    )
    # All bars: cash unchanged, no positions.
    assert (r.equity["cash"] == 100_000.0).all()
    assert (r.equity["positions_value"] == 0.0).all()
    assert (r.equity["total_value"] == 100_000.0).all()


# ---------- strategy interaction --------------------------------------


def test_strategy_buy_and_hold_A_grows_equity():
    prices = _panel()
    idx = prices.index
    profiles = {"A": _profile("A", idx), "B": _profile("B", idx)}
    w = Warmup(prices, mode="eager")
    cache = w.materialize()
    r = run(
        cache,
        strategy=_buy_A_once,
        liquidity_profiles=profiles,
        fill_model=FlatSlippage(bps=0),
        initial_cash=100_000.0,
    )
    # Exactly one buy filled.
    assert r.n_orders_filled == 1
    # Position stays at 100 shares.
    assert r.portfolio_final.positions["A"].size == 100.0
    # Total value moves with A's price. Since A drifts up, equity > initial.
    assert r.equity["total_value"].iloc[-1] > r.equity["total_value"].iloc[0]


def test_trades_dataframe_records_intent_and_fill():
    prices = _panel()
    idx = prices.index
    profiles = {"A": _profile("A", idx), "B": _profile("B", idx)}
    cache = Warmup(prices, mode="eager").materialize()
    r = run(
        cache,
        strategy=_buy_A_once,
        liquidity_profiles=profiles,
        fill_model=FlatSlippage(bps=10),
        initial_cash=100_000.0,
    )
    assert len(r.trades) == 1
    row = r.trades.iloc[0]
    assert row["symbol"] == "A"
    assert row["side"] == 1
    assert row["size_filled"] == 100.0
    assert row["reason"] == "OK"
    assert row["status"] == "filled"
    assert row["tag"] == "test"


# ---------- gating ----------------------------------------------------


def test_no_liquidity_profile_gates_orders():
    prices = _panel()
    idx = prices.index
    profiles = {"A": _profile("A", idx)}  # B intentionally omitted.
    cache = Warmup(prices, mode="eager").materialize()

    submitted = [False]

    def buy_B_on_first_bar(cache, state, t):
        if submitted[0]:
            return []
        submitted[0] = True
        return [
            Order(
                symbol="B",
                side=OrderSide.BUY,
                size=50.0,
                timestamp=t.date() if hasattr(t, "date") else t,
            )
        ]

    r = run(
        cache,
        strategy=buy_B_on_first_bar,
        liquidity_profiles=profiles,
        fill_model=FlatSlippage(bps=0),
        initial_cash=100_000.0,
    )
    assert r.n_orders_filled == 0
    assert r.n_orders_gated == 1
    assert r.trades.iloc[0]["reason"] == "NO_PROFILE"


def test_lifecycle_gate_blocks_orders_after_delisting():
    prices = _panel(n=20)
    idx = prices.index
    profiles = {"A": _profile("A", idx), "B": _profile("B", idx)}
    lc = SecurityLifecycle(
        symbol="A",
        delisting_date=date(2020, 1, 5),
        terminal_action=TerminalAction.MARK_TO_ZERO,
    )
    w = Warmup(prices, mode="eager")
    w.add_lifecycles({"A": lc})
    cache = w.materialize()

    def always_try_to_buy_A(cache, state, t):
        return [
            Order(
                symbol="A",
                side=OrderSide.BUY,
                size=10.0,
                timestamp=t.date() if hasattr(t, "date") else t,
            )
        ]

    r = run(
        cache,
        strategy=always_try_to_buy_A,
        liquidity_profiles=profiles,
        fill_model=FlatSlippage(bps=0),
        initial_cash=1_000_000.0,
    )
    # 20 bars total; first 5 within lifecycle window, remaining 15 gated.
    assert r.n_orders_filled == 5
    assert r.n_orders_gated == 15
    gated_rows = r.trades[r.trades["reason"] == "GATED_LIFECYCLE"]
    assert len(gated_rows) == 15


def test_nan_price_gates_order():
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    prices = pd.DataFrame({"A": [100.0, 101.0, np.nan, 103.0, 104.0], "B": [50.0] * 5}, index=idx)
    profiles = {"A": _profile("A", idx), "B": _profile("B", idx)}
    cache = Warmup(prices, mode="eager").materialize()

    def buy_A_every_bar(cache, state, t):
        return [
            Order(
                symbol="A",
                side=OrderSide.BUY,
                size=1.0,
                timestamp=t.date() if hasattr(t, "date") else t,
            )
        ]

    r = run(
        cache,
        strategy=buy_A_every_bar,
        liquidity_profiles=profiles,
        fill_model=FlatSlippage(bps=0),
        initial_cash=100_000.0,
    )
    assert r.n_orders_seen == 5
    assert r.n_orders_gated == 1  # the NaN bar
    nan_rows = r.trades[r.trades["reason"] == "NO_PRICE"]
    assert len(nan_rows) == 1


def test_symbol_not_in_panel_gates_order():
    prices = _panel(symbols=("A", "B"))
    idx = prices.index
    profiles = {"A": _profile("A", idx), "B": _profile("B", idx)}
    profiles["C"] = _profile("C", idx)  # profile exists but no column
    cache = Warmup(prices, mode="eager").materialize()

    def buy_C(cache, state, t):
        # Symbol not in panel; every submission gets gated. Emit each
        # bar to verify the gate fires uniformly.
        return [
            Order(
                symbol="C",
                side=OrderSide.BUY,
                size=10.0,
                timestamp=t.date() if hasattr(t, "date") else t,
            )
        ]

    r = run(
        cache,
        strategy=buy_C,
        liquidity_profiles=profiles,
        fill_model=FlatSlippage(bps=0),
        initial_cash=100_000.0,
    )
    # Every bar submits an order; every one gets gated because C isn't
    # in the panel columns.
    assert r.n_orders_gated == r.n_orders_seen
    assert r.n_orders_seen > 0
    assert (r.trades["reason"] == "SYMBOL_NOT_IN_PANEL").all()


# ---------- rejected fills surface -----------------------------------


def test_below_min_size_rejected_recorded():
    prices = _panel()
    idx = prices.index
    profiles = {
        "A": LiquidityProfile(
            symbol="A",
            adv_series=pd.Series([1e6] * len(idx), index=idx),
            min_size=100.0,  # forces micro orders to reject.
            max_participation=0.10,
        )
    }
    cache = Warmup(prices, mode="eager").materialize()

    submitted = [False]

    def micro_buy_first_bar(cache, state, t):
        if submitted[0]:
            return []
        submitted[0] = True
        return [
            Order(
                symbol="A",
                side=OrderSide.BUY,
                size=1.0,  # below min_size=100
                timestamp=t.date() if hasattr(t, "date") else t,
            )
        ]

    r = run(
        cache,
        strategy=micro_buy_first_bar,
        liquidity_profiles=profiles,
        fill_model=FlatSlippage(bps=0),
        initial_cash=100_000.0,
    )
    assert r.n_orders_filled == 0
    assert r.n_orders_rejected == 1
    assert r.trades.iloc[0]["status"] == "rejected"
    assert r.trades.iloc[0]["reason"] == "BELOW_MIN_SIZE"


# ---------- summary + parquet ----------------------------------------


def test_summary_string_contains_final_return():
    prices = _panel()
    idx = prices.index
    profiles = {"A": _profile("A", idx), "B": _profile("B", idx)}
    cache = Warmup(prices, mode="eager").materialize()
    r = run(
        cache,
        strategy=_buy_A_once,
        liquidity_profiles=profiles,
        fill_model=FlatSlippage(bps=0),
        initial_cash=100_000.0,
    )
    s = r.summary()
    assert "BacktestResult" in s
    assert "total return" in s


def test_to_parquet_writes_equity_and_trades(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    prices = _panel()
    idx = prices.index
    profiles = {"A": _profile("A", idx), "B": _profile("B", idx)}
    cache = Warmup(prices, mode="eager").materialize()
    r = run(
        cache,
        strategy=_buy_A_once,
        liquidity_profiles=profiles,
        fill_model=FlatSlippage(bps=0),
        initial_cash=100_000.0,
    )
    path_prefix = tmp_path / "run1"
    r.to_parquet(path_prefix)
    eq_file = str(path_prefix) + "_equity.parquet"
    tr_file = str(path_prefix) + "_trades.parquet"
    eq_table = pq.read_table(eq_file)
    tr_table = pq.read_table(tr_file)
    assert "total_value" in eq_table.column_names
    assert "symbol" in tr_table.column_names


# ---------- input validation -----------------------------------------


def test_reject_bad_fill_model():
    prices = _panel()
    idx = prices.index
    profiles = {"A": _profile("A", idx)}
    cache = Warmup(prices, mode="eager").materialize()
    with pytest.raises(KuantValueError):
        run(
            cache,
            strategy=lambda c, s, t: [],
            liquidity_profiles=profiles,
            fill_model=object(),
            initial_cash=100_000.0,
        )


def test_reject_nonpositive_cash():
    prices = _panel()
    idx = prices.index
    profiles = {"A": _profile("A", idx)}
    cache = Warmup(prices, mode="eager").materialize()
    with pytest.raises(KuantValueError):
        run(
            cache,
            strategy=lambda c, s, t: [],
            liquidity_profiles=profiles,
            fill_model=FlatSlippage(bps=0),
            initial_cash=0.0,
        )


def test_reject_non_dict_profiles():
    prices = _panel()
    cache = Warmup(prices, mode="eager").materialize()
    with pytest.raises(KuantShapeError):
        run(
            cache,
            strategy=lambda c, s, t: [],
            liquidity_profiles=["not", "a", "dict"],
            fill_model=FlatSlippage(bps=0),
            initial_cash=100_000.0,
        )


# ---------- state visibility -----------------------------------------


def test_strategy_sees_updated_state_across_bars():
    """After a buy fills, the strategy should observe positions on the
    next call."""
    prices = _panel(n=5)
    idx = prices.index
    profiles = {"A": _profile("A", idx)}
    cache = Warmup(prices, mode="eager").materialize()

    seen_sizes = []

    def spy(cache, state, t):
        seen_sizes.append(state.positions.get("A").size if "A" in state.positions else 0.0)
        if seen_sizes[-1] == 0.0:
            return [
                Order(
                    symbol="A",
                    side=OrderSide.BUY,
                    size=10.0,
                    timestamp=t.date() if hasattr(t, "date") else t,
                )
            ]
        return []

    _ = run(
        cache,
        strategy=spy,
        liquidity_profiles=profiles,
        fill_model=FlatSlippage(bps=0),
        initial_cash=100_000.0,
    )
    # First bar: no position. Second bar onward: 10 shares.
    assert seen_sizes[0] == 0.0
    assert all(s == 10.0 for s in seen_sizes[1:])


# ---------- end-to-end sanity ----------------------------------------


def test_slippage_reduces_equity_vs_zero_slip():
    prices = _panel()
    idx = prices.index
    profiles = {"A": _profile("A", idx), "B": _profile("B", idx)}
    cache = Warmup(prices, mode="eager").materialize()

    def churn(cache, state, t):
        pos = state.positions.get("A")
        size = pos.size if pos else 0.0
        # Alternate: flat -> long -> flat -> long ...
        if size == 0.0:
            return [
                Order(
                    symbol="A",
                    side=OrderSide.BUY,
                    size=50.0,
                    timestamp=t.date() if hasattr(t, "date") else t,
                )
            ]
        return [
            Order(
                symbol="A",
                side=OrderSide.SELL,
                size=50.0,
                timestamp=t.date() if hasattr(t, "date") else t,
            )
        ]

    r0 = run(
        cache,
        strategy=churn,
        liquidity_profiles=profiles,
        fill_model=FlatSlippage(bps=0),
        initial_cash=100_000.0,
    )
    # Rebuild fresh state.
    r10 = run(
        cache,
        strategy=churn,
        liquidity_profiles=profiles,
        fill_model=FlatSlippage(bps=10),
        initial_cash=100_000.0,
    )
    # Same fills, more slippage → lower final equity.
    assert r10.equity["total_value"].iloc[-1] < r0.equity["total_value"].iloc[-1]
