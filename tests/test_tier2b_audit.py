"""Tier-2B audit tests: warnings added in the v0.4.7 close-out pass."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from kuant.errors import KuantDeprecationWarning, KuantNumericWarning


# ---------- options impvol / impvolbisection -------------------------


def test_impvol_max_iter_non_convergence_warns():
    from kuant.options import impvol

    # A price at the very edge of arb bounds with a tight iteration cap
    # so at least one cell can't converge.
    with pytest.warns(KuantNumericWarning, match="KW-"):
        impvol(
            price=np.array([1e-10, 5.0]),
            S=100.0,
            K=np.array([200.0, 100.0]),
            T=0.01,
            r=0.05,
            is_call=True,
            max_iter=3,
        )


def test_impvolbisection_max_iter_warns():
    from kuant.options import impvolbisection

    with pytest.warns(KuantNumericWarning, match="KW-CONV-MAX-ITER"):
        impvolbisection(
            price=1.0,
            S=100.0,
            K=100.0,
            T=1.0,
            r=0.05,
            max_iter=3,
        )


def test_impvolbisection_out_of_bracket_warns():
    from kuant.options import impvolbisection

    # Price above the no-arbitrage upper bound (~S for a call) is
    # un-invertible; even sigma_hi=5.0 cannot reach it.
    with pytest.warns(KuantNumericWarning, match="KW-VAL-RANGE"):
        impvolbisection(
            price=np.array([500.0]),  # far above S; unreachable
            S=100.0,
            K=100.0,
            T=1.0,
            r=0.05,
        )


# ---------- options deltabucket --------------------------------------


def test_deltabucket_no_match_warns():
    from kuant.options import deltabucket

    with pytest.warns(KuantNumericWarning, match="KW-NUM-NO-MATCH"):
        deltabucket(np.array([0.90, 0.95]), 0.10)


# ---------- sindy grangerscan sample-size ----------------------------


def test_grangerscan_sample_size_warns():
    pytest.importorskip("statsmodels")
    from kuant.sindy.grangerscan import grangerscan

    rng = np.random.default_rng(0)
    n = 60
    target = rng.normal(size=n)
    cand = rng.normal(size=n)
    with pytest.warns(KuantNumericWarning, match="KW-NUM-SAMPLE-SIZE"):
        grangerscan(target, {"x": cand}, horizons=[1])


# ---------- qm decoherencescan bucket-small --------------------------


def test_decoherencescan_bucket_small_warns():
    from kuant.qm.decoherencescan import decoherencescan

    rng = np.random.default_rng(0)
    n = 40
    X = rng.normal(size=(n, 2))
    y = rng.normal(size=n)

    def fit(Xt, yt):
        return None

    def predict(_model, Xt):
        return np.zeros(len(Xt))

    with pytest.warns(KuantNumericWarning, match="KW-NUM-BUCKET-SMALL"):
        decoherencescan(
            fit_fn=fit,
            predict_fn=predict,
            X=X,
            y=y,
            predict_window=10,
            train_window=20,
        )


# ---------- backtest execute_fill_panel + PortfolioState ------------


def _profile(sym: str, idx, adv: float = 1e6):
    from kuant.backtest.liquidity import LiquidityProfile

    return LiquidityProfile(
        symbol=sym,
        adv_series=pd.Series([adv] * len(idx), index=idx, dtype=float),
        min_size=1.0,
        max_participation=0.10,
    )


def test_execute_fill_panel_empty_warns():
    from kuant.backtest.liquidity import FlatSlippage, execute_fill_panel

    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    p = _profile("XYZ", idx)
    empty = pd.DataFrame({"timestamp": [], "size": [], "price": []})
    with pytest.warns(KuantNumericWarning, match="KW-FILL-PANEL-EMPTY"):
        execute_fill_panel(empty, p, FlatSlippage(bps=0))


def test_execute_fill_panel_extra_cols_warns():
    from kuant.backtest.liquidity import FlatSlippage, execute_fill_panel

    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    p = _profile("XYZ", idx)
    orders = pd.DataFrame(
        {
            "timestamp": [date(2020, 1, 2)],
            "size": [100.0],
            "price": [50.0],
            "tag": ["custom_metadata"],
        }
    )
    with pytest.warns(KuantNumericWarning, match="KW-FILL-PANEL-EXTRA-COLS"):
        execute_fill_panel(orders, p, FlatSlippage(bps=0))


def test_portfolio_mark_to_market_nan_price_warns():
    from kuant.backtest.fill import FillReport, OrderStatus
    from kuant.backtest.liquidity import FillResult
    from kuant.backtest.position import PortfolioState

    ps = PortfolioState(cash=100_000.0)
    fill = FillResult(
        price=50.0,
        size_filled=100.0,
        size_rejected=0.0,
        slippage_bps=0.0,
        reason="OK",
        cost=5000.0,
    )
    report = FillReport(order_id=1, symbol="XYZ", status=OrderStatus.FILLED, fill=fill)
    ps.apply_fill(report)
    with pytest.warns(KuantNumericWarning, match="KW-PORTFOLIO-NAN-MARK"):
        ps.mark_to_market({"XYZ": float("nan")})


# ---------- warmup cache ---------------------------------------------


def test_warmupcache_ts_not_in_panel_warns():
    from kuant.backtest.lifecycle import SecurityLifecycle
    from kuant.backtest.warmup import Warmup

    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    prices = pd.DataFrame({"A": [1.0, 2, 3, 4, 5]}, index=idx)
    lc = SecurityLifecycle(symbol="A", delisting_date=date(2020, 1, 5))
    w = Warmup(prices)
    w.add_lifecycles({"A": lc})
    cache = w.materialize()
    with pytest.warns(KuantNumericWarning, match="KW-CACHE-TS-NOT-IN-PANEL"):
        # 2099 is not in the panel.
        cache.tradeable(date(2099, 1, 1), "A")


def test_warmupcache_universe_unknown_symbol_warns():
    from kuant.backtest.warmup import Warmup

    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    prices = pd.DataFrame({"A": [1.0] * 5, "B": [2.0] * 5}, index=idx)
    membership = pd.DataFrame({"A": [True] * 5, "B": [True] * 5}, index=idx)
    w = Warmup(prices)
    w.add_universe_membership(membership)
    cache = w.materialize()
    with pytest.warns(KuantNumericWarning, match="KW-CACHE-UNIVERSE-UNKNOWN-SYMBOL"):
        cache.universe(date(2020, 1, 1), "GHOST")


# ---------- outlierpolicy extreme rate -------------------------------


def test_outlierpolicy_extreme_rate_warns():
    from kuant.edgecases import outlierpolicy

    # z-score threshold 0.01 flags nearly everything as an outlier.
    rng = np.random.default_rng(0)
    x = rng.normal(size=1000)
    with pytest.warns(KuantNumericWarning, match="KW-OUTLIER-EXTREME-RATE"):
        outlierpolicy(x, method="zscore", threshold=0.01)


# ---------- delisting deprecation shim -------------------------------


def test_zero_after_delist_deprecation_warns():
    from kuant.edgecases.delistedhandling import zero_after_delist

    with pytest.warns(KuantDeprecationWarning, match="KW-DEPRECATED-USE-LIFECYCLE"):
        zero_after_delist(np.array([100.0, 101, 102, 0, 0]), delist_position=3)


def test_hold_last_price_deprecation_warns():
    from kuant.edgecases.delistedhandling import hold_last_price

    with pytest.warns(KuantDeprecationWarning, match="KW-DEPRECATED-USE-LIFECYCLE"):
        hold_last_price(np.array([100.0, 101, 102, 0, 0]), delist_position=3)
