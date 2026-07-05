"""Tier-2 audit tests: warnings added in the v0.4.5 silent-hides-bug sweep.

Each test verifies that a specific `KW-*` warning fires on its
described trigger, so silent behavior can't sneak back in via a
refactor.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from kuant.errors import KuantNumericWarning


# ---------- Portfolio silent-zero warnings ----------------------------


def test_sharperatio_constant_returns_warns():
    from kuant.portfolio import sharperatio

    with pytest.warns(KuantNumericWarning, match="KW-SHARPE-CONSTANT-RETURNS"):
        sharperatio(np.full(500, 0.001))


def test_sortinoratio_tiny_downside_warns():
    from kuant.portfolio import sortinoratio

    r = np.full(500, 0.001)
    # A handful of trivially-below-target returns to trigger the
    # "downside exists but is FP-noise-tiny" branch.
    r[::100] = 0.001 - 1e-16
    with pytest.warns(KuantNumericWarning, match="KW-SORTINO-TINY-DOWNSIDE"):
        sortinoratio(r, target=0.001)


def test_drawdown_all_nan_warns():
    from kuant.portfolio import drawdown

    with pytest.warns(KuantNumericWarning, match="KW-DRAWDOWN-ALL-NAN"):
        r = drawdown(np.array([np.nan, np.nan, np.nan, np.nan]))
    assert np.isnan(r.max_dd)


def test_kelly_zero_variance_warns():
    from kuant.portfolio.riskmetrics import kelly

    with pytest.warns(KuantNumericWarning, match="KW-KELLY-ZERO-VARIANCE"):
        assert kelly(np.full(100, 0.01)) == 0.0


def test_kelly_negative_edge_warns():
    from kuant.portfolio.riskmetrics import kelly

    rng = np.random.default_rng(0)
    r = rng.normal(-0.01, 0.005, size=500)
    with pytest.warns(KuantNumericWarning, match="KW-KELLY-NEGATIVE-EDGE"):
        assert kelly(r) == 0.0


def test_up_capture_no_up_periods_warns():
    from kuant.portfolio.riskmetrics import up_capture

    with pytest.warns(KuantNumericWarning, match="KW-CAPTURE-NO-UP-PERIODS"):
        up_capture(
            returns=np.array([-0.01, -0.02, -0.005]),
            benchmark=np.array([-0.01, -0.02, -0.005]),
        )


def test_down_capture_no_down_periods_warns():
    from kuant.portfolio.riskmetrics import down_capture

    with pytest.warns(KuantNumericWarning, match="KW-CAPTURE-NO-DOWN-PERIODS"):
        down_capture(
            returns=np.array([0.01, 0.02, 0.005]),
            benchmark=np.array([0.01, 0.02, 0.005]),
        )


def test_probabilistic_sharpe_invalid_moments_warns():
    pytest.importorskip("scipy")
    from kuant.portfolio.riskmetrics import probabilistic_sharpe

    with pytest.warns(KuantNumericWarning, match="KW-PSR-INVALID-MOMENTS"):
        probabilistic_sharpe(2.0, 200, skew=5.0, kurt=1.5)


def test_deflated_sharpe_no_trials_warns():
    pytest.importorskip("scipy")
    from kuant.portfolio.riskmetrics import deflated_sharpe

    with pytest.warns(KuantNumericWarning, match="KW-DSR-NO-TRIALS"):
        deflated_sharpe(1.0, 252, n_trials=1, variance_of_sharpes=0.1)


# ---------- Signals silent-degenerate warnings ------------------------


def test_winsorize_aggressive_limits_warns():
    from kuant.signals import winsorize

    with pytest.warns(KuantNumericWarning, match="KW-WINSORIZE-AGGRESSIVE-LIMITS"):
        winsorize(np.arange(100.0), lo=0.4, hi=0.6)


def test_factor_ic_skipped_periods_warns():
    pytest.importorskip("scipy")
    from kuant.signals import factor_ic

    rng = np.random.default_rng(0)
    T, N = 200, 20
    F = rng.normal(size=(T, N))
    R = rng.normal(size=(T, N))
    R[:100] = np.nan  # first half unusable
    with pytest.warns(KuantNumericWarning, match="KW-FIC-SKIPPED-PERIODS"):
        factor_ic(F, R)


def test_factor_rank_autocorr_constant_warns():
    pytest.importorskip("scipy")
    from kuant.signals import factor_rank_autocorr

    with pytest.warns(KuantNumericWarning, match="KW-RANK-CONSTANT-FACTOR"):
        factor_rank_autocorr(np.ones((100, 50)))


def test_mean_return_by_quantile_thin_buckets_warns():
    from kuant.signals import mean_return_by_quantile

    rng = np.random.default_rng(0)
    F = rng.normal(size=(200, 10))  # 2 names/bucket at n_quantiles=5
    R = rng.normal(size=(200, 10))
    with pytest.warns(KuantNumericWarning, match="KW-QUANTILE-THIN-BUCKETS"):
        mean_return_by_quantile(F, R, n_quantiles=5)


def test_quantile_turnover_degenerate_factor_warns():
    from kuant.signals import quantile_turnover

    F = np.tile(np.arange(50), (100, 1)).astype(np.float64)
    with pytest.warns(KuantNumericWarning, match="KW-TURNOVER-DEGENERATE-FACTOR"):
        quantile_turnover(F)


def test_neutralize_constant_signal_warns():
    from kuant.signals import neutralize

    rng = np.random.default_rng(0)
    factors = {"f": rng.normal(size=500)}
    with pytest.warns(KuantNumericWarning, match="KW-NEUTRALIZE-CONSTANT-SIGNAL"):
        neutralize(np.ones(500), factors)


def test_icdecay_no_clean_warns():
    pytest.importorskip("scipy")
    from kuant.signals import icdecay

    sig = np.full(200, np.nan)
    sig[-3:] = np.arange(3.0)
    ret = np.full(200, np.nan)
    ret[:3] = np.arange(3.0)
    with pytest.warns(KuantNumericWarning, match="KW-ICDECAY-NO-CLEAN"):
        icdecay(sig, ret, horizons=(21, 63))


# ---------- Nulltest resolution warnings ------------------------------


def test_stationary_bootstrap_block_too_long_warns():
    from kuant.nulltest import stationary_bootstrap

    with pytest.warns(KuantNumericWarning, match="KW-BOOT-BLOCK-TOO-LONG"):
        stationary_bootstrap(np.arange(50.0), mean_block_length=100)


def test_bootstrap_ic_low_n_boot_warns():
    from kuant.nulltest import bootstrap_ic

    rng = np.random.default_rng(0)
    sig = rng.normal(size=200)
    ret = 0.1 * sig + rng.normal(size=200)
    with pytest.warns(KuantNumericWarning, match="KW-BOOT-LOW-N-BOOT"):
        bootstrap_ic(sig, ret, n_boot=50)


# ---------- Stats window / ddof / zero-denom warnings ----------------


def test_rollstd_window_exceeds_warns():
    from kuant.stats import rollstd

    with pytest.warns(KuantNumericWarning, match="KW-VAL-WINDOW-EXCEEDS-DATA"):
        rollstd(np.arange(10.0), window=50)


def test_rollstd_ddof_exceeds_window_warns():
    from kuant.stats import rollstd

    with pytest.warns(KuantNumericWarning, match="KW-VAL-DDOF-EXCEEDS-WINDOW"):
        rollstd(np.arange(50.0), window=5, ddof=5)


def test_rollmean_window_exceeds_warns():
    from kuant.stats import rollmean

    with pytest.warns(KuantNumericWarning, match="KW-VAL-WINDOW-EXCEEDS-DATA"):
        rollmean(np.arange(10.0), window=100)


def test_rollcov_ddof_exceeds_window_warns():
    from kuant.stats import rollcov

    with pytest.warns(KuantNumericWarning, match="KW-VAL-DDOF-EXCEEDS-WINDOW"):
        rollcov(np.arange(50.0), np.arange(50.0), window=3, ddof=3)


def test_rollcorr_window_exceeds_warns():
    from kuant.stats import rollcorr

    with pytest.warns(KuantNumericWarning, match="KW-VAL-WINDOW-EXCEEDS-DATA"):
        rollcorr(np.arange(10.0), np.arange(10.0), window=100)


def test_rollmdd_window_exceeds_warns():
    from kuant.stats import rollmdd

    with pytest.warns(KuantNumericWarning, match="KW-VAL-WINDOW-EXCEEDS-DATA"):
        rollmdd(np.array([0.01, -0.02]), window=10)


def test_atr_window_exceeds_warns():
    from kuant.stats import atr

    with pytest.warns(KuantNumericWarning, match="KW-VAL-WINDOW-EXCEEDS-DATA"):
        atr(
            high=np.arange(10.0) + 1,
            low=np.arange(10.0),
            close=np.arange(10.0) + 0.5,
            window=50,
        )


def test_rollsharpe_zero_std_warns():
    from kuant.stats import rollsharpe

    with pytest.warns(KuantNumericWarning, match="KW-NUMERIC-ZERO-STD"):
        rollsharpe(np.zeros(300), window=60, ann_factor=252)


def test_rollsortino_zero_downside_warns():
    from kuant.stats import rollsortino

    with pytest.warns(KuantNumericWarning, match="KW-NUMERIC-ZERO-DOWNSIDE"):
        rollsortino(np.full(300, 0.001), window=60, target=0.0, ann_factor=252)


def test_rollcalmar_zero_drawdown_warns():
    from kuant.stats import rollcalmar

    with pytest.warns(KuantNumericWarning, match="KW-NUMERIC-ZERO-DRAWDOWN"):
        rollcalmar(np.linspace(0.001, 0.002, 300), window=60, ann_factor=252)


# ---------- QM HMM state-order warnings ------------------------------


def test_hmm_baumwelch_state_order_warning():
    from kuant.qm.hmm import baumwelch

    rng = np.random.default_rng(0)
    obs = rng.integers(0, 3, size=200)
    with pytest.warns(KuantNumericWarning, match="KW-HMM-STATE-ORDER"):
        baumwelch(obs, n_states=2, n_symbols=3, max_iter=5, seed=0)


def test_ghmm_baumwelch_state_order_warning():
    from kuant.qm.ghmm import baumwelch as ghmm_baumwelch

    rng = np.random.default_rng(0)
    obs = rng.normal(size=200)
    with pytest.warns(KuantNumericWarning, match="KW-HMM-STATE-ORDER"):
        ghmm_baumwelch(obs, n_states=2, max_iter=5, seed=0)


# ---------- Backtest/data warnings ------------------------------------


def test_align_empty_inner_intersect_warns():
    from kuant.data import align

    idx_a = np.array([1, 2, 3])
    idx_b = np.array([10, 11, 12])
    with pytest.warns(KuantNumericWarning, match="KW-ALIGN-EMPTY-INTERSECT"):
        align((idx_a, np.arange(3.0)), (idx_b, np.arange(3.0)), method="inner")


def test_apply_lifecycle_panel_unknown_symbol_warns():
    from kuant.backtest.lifecycle import (
        SecurityLifecycle,
        apply_lifecycle_panel,
    )

    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    df = pd.DataFrame({"AAA": [1.0, 2, 3, 4, 5]}, index=idx)
    lcs = {"AAA": SecurityLifecycle(symbol="AAA")}
    lcs["ZZZ"] = SecurityLifecycle(symbol="ZZZ", delisting_date=date(2019, 1, 1))
    with pytest.warns(KuantNumericWarning, match="KW-LIFECYCLE-UNKNOWN-SYMBOL"):
        apply_lifecycle_panel(df, lcs)


def test_liquidity_mask_all_false_warns():
    from kuant.backtest.liquidity import LiquidityProfile, liquidity_mask

    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    p = LiquidityProfile(
        symbol="X",
        adv_series=pd.Series([1e6] * 5, index=idx),
        min_size=1.0,
    )
    with pytest.warns(KuantNumericWarning, match="KW-LIQ-MASK-ALL-FALSE"):
        liquidity_mask(idx, p, min_adv=1e12)
