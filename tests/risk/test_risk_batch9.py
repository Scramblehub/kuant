"""Tests for kuant.risk (v0.6.0 batch 9)."""

from __future__ import annotations

import numpy as np
import pytest

from kuant.errors import KuantNumericWarning, KuantValueError
from kuant.risk import (
    cornishfishervar,
    covar,
    esbootstrap,
    evtvar,
    mes,
)


# ---------- cornishfishervar ----------


def test_cornishfishervar_gaussian_matches_gaussian_var():
    rng = np.random.default_rng(0)
    r = rng.standard_normal(50_000) * 0.01  # zero skew, zero excess kurt
    res = cornishfishervar(r, alpha=0.95)
    # CF should collapse to the Gaussian z (~ -1.6449); CF quantile close to Gaussian z
    assert abs(res.z_cf - res.z_gaussian) < 0.05
    # And VaR close to 1.645 * sigma
    expected = 1.6449 * res.std
    assert abs(res.var - expected) < 0.001


def test_cornishfishervar_fat_tail_var_larger_than_gaussian():
    rng = np.random.default_rng(1)
    r = rng.standard_t(df=3, size=20_000) * 0.005  # heavy tails
    res = cornishfishervar(r, alpha=0.99)
    gauss_var = 2.326 * res.std - res.mean
    # Fat tail should push CF VaR above the Gaussian
    assert res.var > gauss_var


def test_cornishfishervar_rejects_short_series():
    with pytest.raises(KuantValueError):
        cornishfishervar(np.zeros(10))


def test_cornishfishervar_warns_on_extreme_kurtosis():
    rng = np.random.default_rng(11)
    r = rng.standard_normal(2_000) * 0.01
    r[0] = 0.5  # a single monster-day spike blows up kurtosis
    with pytest.warns(KuantNumericWarning, match="KW-CF-EXPANSION-INVALID"):
        cornishfishervar(r, alpha=0.95)


# ---------- evtvar ----------


def test_evtvar_recovers_var_and_es_ordering():
    rng = np.random.default_rng(2)
    r = rng.standard_t(df=4, size=5_000) * 0.01
    res = evtvar(r, alpha=0.99, threshold_pct=0.90)
    # ES must be at least as large as VaR (same-sign loss magnitudes)
    assert res.es >= res.var
    assert res.var > res.threshold
    assert res.n_exceedances > 100


def test_evtvar_rejects_short_series():
    with pytest.raises(KuantValueError):
        evtvar(np.zeros(100))


def test_evtvar_rejects_too_high_threshold():
    rng = np.random.default_rng(3)
    r = rng.standard_normal(300) * 0.01
    with pytest.raises(KuantValueError):
        evtvar(r, threshold_pct=0.99)  # too few exceedances


# ---------- esbootstrap ----------


def test_esbootstrap_ci_contains_point():
    rng = np.random.default_rng(4)
    r = rng.standard_normal(2_000) * 0.01
    res = esbootstrap(r, conf_alpha=0.95, n_boot=200, block_size=21)
    assert res.es_ci_low <= res.es_point <= res.es_ci_high
    assert res.es_point >= res.var_point


def test_esbootstrap_shrinks_ci_with_more_data():
    rng = np.random.default_rng(5)
    r_small = rng.standard_normal(500) * 0.01
    r_big = rng.standard_normal(5_000) * 0.01
    ci_small = esbootstrap(r_small, n_boot=200, block_size=21)
    ci_big = esbootstrap(r_big, n_boot=200, block_size=21)
    small_w = ci_small.es_ci_high - ci_small.es_ci_low
    big_w = ci_big.es_ci_high - ci_big.es_ci_low
    assert big_w < small_w


def test_esbootstrap_block_size_range_error():
    rng = np.random.default_rng(6)
    r = rng.standard_normal(200) * 0.01
    with pytest.raises(KuantValueError):
        esbootstrap(r, n_boot=50, block_size=200)


# ---------- covar ----------


def test_covar_positive_dependence_gives_positive_slope():
    rng = np.random.default_rng(7)
    n = 2_000
    sys = rng.standard_t(df=5, size=n) * 0.01
    asset = 0.7 * sys + 0.3 * rng.standard_t(df=5, size=n) * 0.01
    res = covar(asset, sys, alpha=0.95)
    assert res.q_regression_slope > 0.3
    assert res.covar > res.var_x_uncond * 0.8  # CoVaR at least in same ballpark
    assert res.delta_covar > 0


def test_covar_independent_gives_low_delta():
    rng = np.random.default_rng(8)
    n = 2_000
    sys = rng.standard_normal(n) * 0.01
    asset = rng.standard_normal(n) * 0.01
    res = covar(asset, sys, alpha=0.95)
    assert abs(res.q_regression_slope) < 0.25
    assert abs(res.delta_covar) < 0.005


def test_covar_length_mismatch():
    with pytest.raises(KuantValueError):
        covar(np.zeros(200), np.zeros(300))


# ---------- mes ----------


def test_mes_positive_when_asset_moves_with_system():
    rng = np.random.default_rng(9)
    n = 2_000
    sys = rng.standard_normal(n) * 0.01
    asset = 0.8 * sys + 0.2 * rng.standard_normal(n) * 0.01
    res = mes(asset, sys, tau=0.05)
    assert res.mes > 0
    assert res.n_tail_days > 50


def test_mes_near_zero_when_independent():
    rng = np.random.default_rng(10)
    n = 5_000
    sys = rng.standard_normal(n) * 0.01
    asset = rng.standard_normal(n) * 0.01
    res = mes(asset, sys, tau=0.05)
    assert abs(res.mes) < 0.003


def test_mes_length_mismatch():
    with pytest.raises(KuantValueError):
        mes(np.zeros(200), np.zeros(300))


# ---------- shared: KE-VAL-MIN-CLEAN ----------


@pytest.mark.parametrize(
    "fn",
    [
        cornishfishervar,
        evtvar,
        esbootstrap,
    ],
)
def test_min_clean_gate(fn):
    r = np.array([np.nan, np.inf, -np.inf, 0.01, 0.02])
    with pytest.raises(KuantValueError) as exc:
        fn(r)
    assert "KE-VAL-MIN-CLEAN" in str(exc.value)


def test_covar_min_clean_gate():
    with pytest.raises(KuantValueError) as exc:
        covar(np.array([np.nan, 0.01, 0.02]), np.array([np.nan, 0.01, 0.02]))
    assert "KE-VAL-MIN-CLEAN" in str(exc.value)


def test_mes_min_clean_gate():
    with pytest.raises(KuantValueError) as exc:
        mes(np.array([np.nan, 0.01, 0.02]), np.array([np.nan, 0.01, 0.02]))
    assert "KE-VAL-MIN-CLEAN" in str(exc.value)
