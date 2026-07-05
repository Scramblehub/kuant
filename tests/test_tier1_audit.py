"""Tier-1 audit tests: 39 errors added in the v0.4.4 warnings/errors sweep.

One test per finding. Each verifies the specific error code fires on
the described trigger, so if a future refactor loosens a guard we
catch it immediately.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from kuant.errors import KuantShapeError, KuantValueError


# ---------- Group A: backtest correctness (4) --------------------------


def test_execute_fill_size_nan_raises():
    from kuant.backtest.liquidity import FlatSlippage, LiquidityProfile, execute_fill

    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    adv = pd.Series([1e6] * 3, index=idx)
    p = LiquidityProfile(symbol="X", adv_series=adv, min_size=1.0)
    with pytest.raises(KuantValueError, match="KE-FILL-SIZE-NAN"):
        execute_fill(float("nan"), 50.0, p, date(2020, 1, 2), FlatSlippage(bps=5))


def test_position_apply_fill_nan_price_raises():
    from kuant.backtest.position import Position

    p = Position(symbol="X")
    with pytest.raises(KuantValueError, match="KE-POS-PRICE-INVALID"):
        p.apply_fill(size_filled=100.0, price=float("nan"))
    with pytest.raises(KuantValueError, match="KE-POS-PRICE-INVALID"):
        p.apply_fill(size_filled=100.0, price=0.0)


def test_position_apply_fill_nan_size_raises():
    from kuant.backtest.position import Position

    p = Position(symbol="X")
    with pytest.raises(KuantValueError, match="KE-POS-SIZE-INVALID"):
        p.apply_fill(size_filled=float("nan"), price=50.0)


def test_portfolio_apply_fill_nan_price_with_nonzero_size_raises():
    from kuant.backtest.fill import FillReport, OrderStatus
    from kuant.backtest.liquidity import FillResult
    from kuant.backtest.position import PortfolioState

    ps = PortfolioState(cash=100_000.0)
    fake_fill = FillResult(
        price=float("nan"),
        size_filled=100.0,  # NONZERO with NaN price
        size_rejected=0.0,
        slippage_bps=0.0,
        reason="OK",
        cost=0.0,
    )
    report = FillReport(order_id=1, symbol="X", status=OrderStatus.FILLED, fill=fake_fill)
    with pytest.raises(KuantValueError, match="KE-PORTFOLIO-FILL-PRICE-INVALID"):
        ps.apply_fill(report)


# ---------- Group B: NaN pollution (2) --------------------------------


def test_mht_correction_nan_p_raises():
    from kuant.nulltest.mht_correction import mht_correction

    with pytest.raises(KuantValueError, match="KE-VAL-NAN-PVALUES"):
        mht_correction(np.array([0.01, np.nan, 0.2]))


def test_permtest_nan_real_metric_raises():
    from kuant.sindy.permtest import permtest

    def metric(x, y):
        return 0.5

    with pytest.raises(KuantValueError, match="KE-VAL-FINITE"):
        permtest(
            real_metric=float("nan"),
            metric_fn=metric,
            x=np.arange(50.0),
            y=np.arange(50.0),
            n_perms=100,
        )


# ---------- Group C: empty inputs (5) ---------------------------------


def test_drawdown_empty_raises():
    from kuant.portfolio import drawdown

    with pytest.raises(KuantValueError, match="KE-VAL-EMPTY"):
        drawdown(np.array([]))


def test_belltest_empty_features_raises():
    pytest.importorskip("sklearn")
    from kuant.qm.belltest import belltest

    with pytest.raises(KuantValueError, match="KE-VAL-EMPTY"):
        belltest({}, target=np.arange(100.0))


def test_sindylasso_empty_library_raises():
    pytest.importorskip("sklearn")
    from kuant.sindy.sindylasso import sindylasso

    with pytest.raises(KuantValueError, match="KE-VAL-EMPTY"):
        sindylasso(np.arange(200.0), library={})


def test_deltabucket_empty_deltas_raises():
    from kuant.options import deltabucket

    with pytest.raises(KuantValueError, match="KE-VAL-EMPTY"):
        deltabucket(np.array([]), 0.25)


def test_warmup_empty_panel_raises():
    from kuant.backtest.warmup import Warmup

    idx = pd.date_range("2020-01-01", periods=0, freq="D")
    empty_zero_rows = pd.DataFrame(columns=["A"], index=idx)
    with pytest.raises(KuantValueError, match="KE-WARMUP-EMPTY-PANEL"):
        Warmup(empty_zero_rows)
    idx2 = pd.date_range("2020-01-01", periods=5, freq="D")
    empty_zero_cols = pd.DataFrame(index=idx2)
    with pytest.raises(KuantValueError, match="KE-WARMUP-EMPTY-PANEL"):
        Warmup(empty_zero_cols)


# ---------- Group D: fn contract (2) ----------------------------------


def test_belltest_bad_joint_model_fn_raises():
    pytest.importorskip("sklearn")
    from kuant.qm.belltest import belltest

    def bad_fn(X_tr, y_tr):
        # Returns wrong-length prediction; neither len(y) nor len(test_fold).
        return np.zeros(3)

    rng = np.random.default_rng(0)
    n = 100
    features = {"a": rng.normal(size=n), "b": rng.normal(size=n)}
    y = rng.normal(size=n)
    with pytest.raises(KuantValueError, match="KE-VAL-CONTRACT"):
        belltest(features, y, joint_model_fn=bad_fn, n_splits=5)


def test_decoherencescan_bad_predict_fn_raises():
    from kuant.qm.decoherencescan import decoherencescan

    rng = np.random.default_rng(0)
    n = 200
    X = rng.normal(size=(n, 2))
    y = rng.normal(size=n)

    def fit(Xt, yt):
        return None

    def bad_predict(_model, X_bar):
        return np.zeros(1)  # wrong length

    with pytest.raises(KuantValueError, match="KE-VAL-CONTRACT"):
        decoherencescan(
            fit_fn=fit,
            predict_fn=bad_predict,
            X=X,
            y=y,
            predict_window=10,
            train_window=50,
        )


# ---------- Group E: parameter guards (10) ----------------------------


def test_gpdcdf_nonpositive_scale_raises():
    from kuant.core import gpdcdf

    with pytest.raises(KuantValueError, match="KE-VAL-POSITIVE"):
        gpdcdf(0.5, 0.2, 0.0)


def test_gpdpdf_nonpositive_scale_raises():
    from kuant.core import gpdpdf

    with pytest.raises(KuantValueError, match="KE-VAL-POSITIVE"):
        gpdpdf(0.5, 0.2, -1.0)


def test_gpdppf_nonpositive_scale_raises():
    from kuant.core import gpdppf

    with pytest.raises(KuantValueError, match="KE-VAL-POSITIVE"):
        gpdppf(0.5, 0.2, 0.0)


def test_tcdf_nonpositive_df_raises():
    from kuant.core import tcdf

    with pytest.raises(KuantValueError, match="KE-VAL-POSITIVE"):
        tcdf(0.5, 0.0)


def test_tpdf_nonpositive_df_raises():
    from kuant.core import tpdf

    with pytest.raises(KuantValueError, match="KE-VAL-POSITIVE"):
        tpdf(0.5, -2.0)


def test_logtcdf_nonpositive_df_raises():
    from kuant.core import logtcdf

    with pytest.raises(KuantValueError, match="KE-VAL-POSITIVE"):
        logtcdf(1.0, -1.0)


def test_moneynessbucket_nonpositive_S_or_K_raises():
    from kuant.options import moneynessbucket

    with pytest.raises(KuantValueError, match="KE-VAL-POSITIVE"):
        moneynessbucket(np.array([100.0, 0.0]), np.array([100.0, 100.0]), 1.0, 0.05)
    with pytest.raises(KuantValueError, match="KE-VAL-POSITIVE"):
        moneynessbucket(np.array([100.0]), np.array([-1.0]), 1.0, 0.05)


def test_zenoscan_nonpositive_retrain_freq_raises():
    from kuant.qm.zenoscan import zenoscan

    rng = np.random.default_rng(0)
    n = 200
    X = rng.normal(size=(n, 2))
    y = rng.normal(size=n)

    def fit(Xt, yt):
        return None

    def predict(_model, Xt):
        return np.zeros(len(Xt))

    def metric(a, b):
        return 0.0

    with pytest.raises(KuantValueError, match="KE-VAL-POSITIVE"):
        zenoscan(
            fit_fn=fit,
            predict_fn=predict,
            metric_fn=metric,
            X=X,
            y=y,
            retrain_freqs=[0, 21],
            train_window=100,
        )


def test_grangerscan_nonpositive_horizon_raises():
    pytest.importorskip("statsmodels")
    from kuant.sindy.grangerscan import grangerscan

    rng = np.random.default_rng(0)
    n = 200
    with pytest.raises(KuantValueError, match="KE-VAL-POSITIVE"):
        grangerscan(rng.normal(size=n), {"x": rng.normal(size=n)}, horizons=[0, 1])


def test_corpaction_nonpositive_split_raises():
    from kuant.data import corpaction

    with pytest.raises(KuantValueError, match="KE-CORP-SPLIT-NONPOSITIVE"):
        corpaction(
            np.ones(5, dtype=np.float64),
            split_positions=[2],
            split_ratios=[0.0],
            mode="split_only",
        )
    with pytest.raises(KuantValueError, match="KE-CORP-SPLIT-NONPOSITIVE"):
        corpaction(
            np.ones(5, dtype=np.float64),
            split_positions=[2],
            split_ratios=[-1.0],
            mode="split_only",
        )


# ---------- Group F: range/domain (12) -------------------------------


def test_liquidity_mask_negative_min_adv_raises():
    from kuant.backtest.liquidity import LiquidityProfile, liquidity_mask

    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    adv = pd.Series([1e6] * 3, index=idx)
    p = LiquidityProfile(symbol="X", adv_series=adv, min_size=1.0)
    with pytest.raises(KuantValueError, match="KE-LIQ-MASK-MIN-ADV-NEGATIVE"):
        liquidity_mask(idx, p, min_adv=-1.0)


def test_rollskew_window_lt_3_raises():
    from kuant.stats import rollskew

    with pytest.raises(KuantValueError, match="KE-VAL-RANGE"):
        rollskew(np.arange(20.0), window=2)


def test_rollkurt_window_lt_4_raises():
    from kuant.stats import rollkurt

    with pytest.raises(KuantValueError, match="KE-VAL-RANGE"):
        rollkurt(np.arange(20.0), window=3)


def test_rollcorr_window_lt_2_raises():
    from kuant.stats import rollcorr

    with pytest.raises(KuantValueError, match="KE-VAL-RANGE"):
        rollcorr(np.arange(10.0), np.arange(10.0), window=1)


def test_rollbeta_window_lt_2_raises():
    from kuant.stats import rollbeta

    with pytest.raises(KuantValueError, match="KE-VAL-RANGE"):
        rollbeta(np.arange(10.0), np.arange(10.0), window=1)


def test_varianceratiotest_lags_lt_2_raises():
    pytest.importorskip("arch")
    from kuant.stats import varianceratiotest

    with pytest.raises(KuantValueError, match="KE-VAL-RANGE"):
        varianceratiotest(np.random.default_rng(0).standard_normal(500), lags=1)


def test_accelerationscan_len_lt_3_raises():
    from kuant.sindy.accelerationscan import accelerationscan

    with pytest.raises(KuantValueError, match="KE-VAL-RANGE"):
        accelerationscan(np.array([1.0, 2.0]), np.array([0.0, 0.0]))


def test_bettiseries_window_gt_len_raises():
    from kuant.topology import bettiseries

    with pytest.raises(KuantValueError, match="KE-VAL-RANGE"):
        bettiseries(np.arange(50.0), window=100)


def test_dispersioncollapse_window_gt_n_bars_raises():
    from kuant.topology import dispersioncollapse

    with pytest.raises(KuantValueError, match="KE-VAL-RANGE"):
        dispersioncollapse(np.zeros((30, 10)), window=63)


def test_moneynessbucket_non_monotone_edges_raises():
    from kuant.options import moneynessbucket

    with pytest.raises(KuantValueError, match="KE-VAL-RANGE"):
        moneynessbucket(100.0, 100.0, 1.0, 0.05, edges=np.array([0.03, -0.10, 0.10]))


def test_parkinson_H_lt_L_raises():
    from kuant.stats import parkinson

    with pytest.raises(KuantValueError, match="KE-VAL-RANGE"):
        parkinson(high=np.array([1.0, 1.0]), low=np.array([2.0, 2.0]))


def test_garmanklass_ohlc_ordering_violation_raises():
    from kuant.stats import garmanklass

    with pytest.raises(KuantValueError, match="KE-VAL-RANGE"):
        garmanklass(
            open_=np.array([1.0]),
            high=np.array([0.9]),  # H < L
            low=np.array([1.1]),
            close=np.array([1.0]),
        )


def test_zenoscan_train_window_ge_T_raises():
    from kuant.qm.zenoscan import zenoscan

    rng = np.random.default_rng(0)
    n = 50
    X = rng.normal(size=(n, 2))
    y = rng.normal(size=n)

    def fit(Xt, yt):
        return None

    def predict(_model, Xt):
        return np.zeros(len(Xt))

    def metric(a, b):
        return 0.0

    with pytest.raises(KuantValueError, match="KE-VAL-RANGE"):
        zenoscan(
            fit_fn=fit,
            predict_fn=predict,
            metric_fn=metric,
            X=X,
            y=y,
            retrain_freqs=[5],
            train_window=100,
        )


def test_zenoscan_len_X_ne_len_y_raises():
    from kuant.qm.zenoscan import zenoscan

    X = np.zeros((100, 3))
    y = np.zeros(50)

    def fit(Xt, yt):
        return None

    def predict(_model, Xt):
        return np.zeros(len(Xt))

    def metric(a, b):
        return 0.0

    with pytest.raises(KuantShapeError, match="KE-SHAPE-EQUAL-LEN"):
        zenoscan(
            fit_fn=fit,
            predict_fn=predict,
            metric_fn=metric,
            X=X,
            y=y,
            retrain_freqs=[21],
            train_window=50,
        )


# ---------- Group G: categorical / enum (3) --------------------------


def test_order_limit_invalid_limit_price_raises():
    from kuant.backtest.fill import Order, OrderSide, OrderType

    with pytest.raises(KuantValueError, match="KE-ORDER-LIMIT-INVALID"):
        Order(
            symbol="X",
            side=OrderSide.BUY,
            size=100.0,
            timestamp=date(2020, 1, 1),
            order_type=OrderType.LIMIT,
            limit_price=0.0,
        )
    with pytest.raises(KuantValueError, match="KE-ORDER-LIMIT-INVALID"):
        Order(
            symbol="X",
            side=OrderSide.BUY,
            size=100.0,
            timestamp=date(2020, 1, 1),
            order_type=OrderType.STOP,
            limit_price=float("nan"),
        )


def test_submit_unknown_reason_raises():
    """A FillResult with a stale reason string should surface, not silently
    become REJECTED."""
    from kuant.backtest.fill.submit import _status_from_reason

    with pytest.raises(KuantValueError, match="KE-SUBMIT-UNKNOWN-REASON"):
        _status_from_reason("SETTLEMENT_HALT")


def test_warmup_indicator_kernel_failure_names_indicator():
    from kuant.backtest.warmup import Warmup

    idx = pd.date_range("2020-01-01", periods=10, freq="D")
    prices = pd.DataFrame({"A": np.arange(10.0)}, index=idx)

    def broken(_x):
        raise RuntimeError("kernel-internal boom")

    w = Warmup(prices, mode="eager")
    w.add_indicator("broken_ind", broken, per_symbol=True)
    with pytest.raises(KuantValueError, match="KE-WARMUP-INDICATOR-FAILED"):
        w.materialize()


# ---------- New helpers in kuant._validation --------------------------


def test_require_non_empty_helper():
    from kuant._validation import require_non_empty

    with pytest.raises(KuantValueError, match="KE-VAL-EMPTY"):
        require_non_empty(np.array([]), "x", kernel="testk")
    require_non_empty(np.array([1.0]), "x", kernel="testk")


def test_require_monotone_increasing_helper():
    from kuant._validation import require_monotone_increasing

    with pytest.raises(KuantValueError, match="KE-VAL-RANGE"):
        require_monotone_increasing(np.array([1.0, 0.5, 2.0]), "e", kernel="testk")
    with pytest.raises(KuantValueError, match="KE-VAL-RANGE"):
        require_monotone_increasing(np.array([1.0, 1.0, 2.0]), "e", kernel="testk", strict=True)
    require_monotone_increasing(np.array([1.0, 2.0, 3.0]), "e", kernel="testk")
    require_monotone_increasing(np.array([1.0, 1.0, 2.0]), "e", kernel="testk", strict=False)


def test_require_ohlc_ordering_helper():
    from kuant._validation import require_ohlc_ordering

    # Valid case: pass through.
    require_ohlc_ordering(
        np.array([100.0]),
        np.array([105.0]),
        np.array([95.0]),
        np.array([103.0]),
        kernel="testk",
    )
    # H < L.
    with pytest.raises(KuantValueError, match="KE-VAL-RANGE"):
        require_ohlc_ordering(
            np.array([100.0]),
            np.array([90.0]),
            np.array([95.0]),
            np.array([100.0]),
            kernel="testk",
        )
    # max(O,C) > H.
    with pytest.raises(KuantValueError, match="KE-VAL-RANGE"):
        require_ohlc_ordering(
            np.array([120.0]),
            np.array([105.0]),
            np.array([95.0]),
            np.array([103.0]),
            kernel="testk",
        )
    # min(O,C) < L.
    with pytest.raises(KuantValueError, match="KE-VAL-RANGE"):
        require_ohlc_ordering(
            np.array([90.0]),
            np.array([105.0]),
            np.array([95.0]),
            np.array([103.0]),
            kernel="testk",
        )
