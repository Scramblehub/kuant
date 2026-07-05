"""Tests for kuant.signals.factorscoring."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scipy")

from kuant.errors import KuantShapeError, KuantValueError  # noqa: E402
from kuant.signals.factorscoring import (  # noqa: E402
    FactorICResult,
    QuantileReturnsResult,
    QuantileSpreadResult,
    QuantileTurnoverResult,
    RankAutocorrResult,
    factor_ic,
    factor_rank_autocorr,
    mean_return_by_quantile,
    quantile_spread,
    quantile_turnover,
)


# ---------- factor_ic ---------------------------------------------------


def test_factor_ic_positive_when_factor_drives_returns():
    rng = np.random.default_rng(0)
    T, N = 500, 50
    F = rng.standard_normal((T, N))
    R = 0.10 * F + rng.standard_normal((T, N))
    res = factor_ic(F, R)
    assert res.mean > 0.05
    assert res.t_stat > 3.0
    assert res.n_periods == T


def test_factor_ic_zero_on_independent_returns():
    rng = np.random.default_rng(0)
    F = rng.standard_normal((500, 50))
    R = rng.standard_normal((500, 50))
    res = factor_ic(F, R)
    # Under the null the IC should be near zero. Give plenty of slack.
    assert abs(res.mean) < 0.03


def test_factor_ic_return_type():
    F = np.random.default_rng(0).standard_normal((100, 10))
    R = np.random.default_rng(1).standard_normal((100, 10))
    assert isinstance(factor_ic(F, R), FactorICResult)


def test_factor_ic_pearson_method_works():
    rng = np.random.default_rng(0)
    F = rng.standard_normal((100, 20))
    R = 0.5 * F + rng.standard_normal((100, 20))
    res = factor_ic(F, R, method="pearson")
    assert res.method == "pearson"
    assert res.mean > 0.2


def test_factor_ic_rejects_bad_method():
    F = np.zeros((5, 5))
    with pytest.raises(KuantValueError):
        factor_ic(F, F, method="bogus")


def test_factor_ic_rejects_shape_mismatch():
    with pytest.raises(KuantShapeError):
        factor_ic(np.zeros((5, 5)), np.zeros((5, 6)))


def test_factor_ic_nan_row_yields_nan():
    """A row with fewer than 3 finite pairs cannot produce a correlation."""
    F = np.array([[1.0, 2, 3], [np.nan, np.nan, np.nan]])
    R = np.array([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]])
    res = factor_ic(F, R)
    assert np.isfinite(res.ic[0])
    assert np.isnan(res.ic[1])
    assert res.n_periods == 1


def test_factor_ic_summary():
    F = np.random.default_rng(0).standard_normal((100, 10))
    R = np.random.default_rng(1).standard_normal((100, 10))
    s = factor_ic(F, R).summary()
    assert "FactorICResult" in s
    assert "IR" in s


# ---------- factor_rank_autocorr ----------------------------------------


def test_rank_autocorr_high_on_persistent_factor():
    """AR(1) factor with ρ=0.9 → rank autocorr should be ~0.9."""
    rng = np.random.default_rng(0)
    T, N = 200, 50
    F = np.zeros((T, N))
    F[0] = rng.standard_normal(N)
    for t in range(1, T):
        F[t] = 0.95 * F[t - 1] + 0.05 * rng.standard_normal(N)
    res = factor_rank_autocorr(F)
    assert res.mean > 0.85


def test_rank_autocorr_low_on_random_factor():
    rng = np.random.default_rng(0)
    F = rng.standard_normal((200, 50))
    res = factor_rank_autocorr(F)
    assert abs(res.mean) < 0.1


def test_rank_autocorr_lag_beyond_T_rejected():
    with pytest.raises(KuantValueError):
        factor_rank_autocorr(np.zeros((5, 3)), lag=10)


def test_rank_autocorr_returns_dataclass():
    F = np.random.default_rng(0).standard_normal((100, 20))
    assert isinstance(factor_rank_autocorr(F), RankAutocorrResult)


def test_rank_autocorr_lag_gt_1():
    """Verify the lag parameter changes the offset."""
    rng = np.random.default_rng(0)
    F = rng.standard_normal((100, 20))
    res_1 = factor_rank_autocorr(F, lag=1)
    res_5 = factor_rank_autocorr(F, lag=5)
    # NaN mask shifts.
    assert np.isnan(res_5.autocorr[:5]).all()
    assert not np.isnan(res_5.autocorr[5:]).all()
    assert res_5.n_periods == res_1.n_periods - 4


# ---------- mean_return_by_quantile ------------------------------------


def test_quantile_returns_monotone_when_factor_drives_returns():
    """Top quantile should out-earn bottom when factor predicts returns."""
    rng = np.random.default_rng(0)
    T, N = 500, 50
    F = rng.standard_normal((T, N))
    R = 0.1 * F + rng.standard_normal((T, N))
    res = mean_return_by_quantile(F, R, n_quantiles=5)
    assert res.total_by_quantile[-1] > res.total_by_quantile[0]


def test_quantile_returns_shape():
    F = np.random.default_rng(0).standard_normal((100, 25))
    R = np.random.default_rng(1).standard_normal((100, 25))
    res = mean_return_by_quantile(F, R, n_quantiles=5)
    assert res.mean_by_quantile.shape == (100, 5)
    assert res.total_by_quantile.shape == (5,)


def test_quantile_returns_returns_dataclass():
    F = np.random.default_rng(0).standard_normal((50, 10))
    R = np.random.default_rng(1).standard_normal((50, 10))
    assert isinstance(mean_return_by_quantile(F, R), QuantileReturnsResult)


def test_quantile_returns_rejects_too_few_quantiles():
    F = np.zeros((5, 5))
    with pytest.raises(KuantValueError):
        mean_return_by_quantile(F, F, n_quantiles=1)


# ---------- quantile_spread --------------------------------------------


def test_spread_positive_for_predictive_factor():
    rng = np.random.default_rng(0)
    F = rng.standard_normal((500, 50))
    R = 0.1 * F + rng.standard_normal((500, 50))
    res = quantile_spread(F, R, n_quantiles=5)
    assert res.mean > 0
    assert res.t_stat > 3.0


def test_spread_near_zero_on_null():
    rng = np.random.default_rng(0)
    F = rng.standard_normal((500, 50))
    R = rng.standard_normal((500, 50))
    res = quantile_spread(F, R, n_quantiles=5)
    # Under the null the spread has zero expectation.
    assert abs(res.t_stat) < 3.0


def test_spread_returns_dataclass():
    F = np.random.default_rng(0).standard_normal((100, 20))
    R = np.random.default_rng(1).standard_normal((100, 20))
    assert isinstance(quantile_spread(F, R), QuantileSpreadResult)


def test_spread_shape():
    F = np.random.default_rng(0).standard_normal((100, 25))
    R = np.random.default_rng(1).standard_normal((100, 25))
    res = quantile_spread(F, R, n_quantiles=5)
    assert res.spread.shape == (100,)


# ---------- quantile_turnover ------------------------------------------


def test_turnover_zero_on_frozen_factor():
    """If the factor never changes, no bucket names change → turnover = 0."""
    N = 30
    F = np.tile(np.arange(N, dtype=np.float64), (50, 1))
    res = quantile_turnover(F, n_quantiles=5)
    finite_top = res.top_turnover[np.isfinite(res.top_turnover)]
    assert finite_top.max() == 0.0


def test_turnover_high_on_random_factor():
    """Fresh random factor per period → most names churn."""
    F = np.random.default_rng(0).standard_normal((200, 50))
    res = quantile_turnover(F, n_quantiles=5)
    # ~80% churn expected for i.i.d. rows.
    assert res.top_mean > 0.6


def test_turnover_returns_dataclass():
    F = np.random.default_rng(0).standard_normal((50, 20))
    assert isinstance(quantile_turnover(F), QuantileTurnoverResult)


def test_turnover_leading_nan():
    """Position 0 has no prior period → NaN."""
    F = np.random.default_rng(0).standard_normal((50, 20))
    res = quantile_turnover(F, n_quantiles=5)
    assert np.isnan(res.top_turnover[0])
    assert np.isnan(res.bottom_turnover[0])


def test_turnover_rejects_bad_n_quantiles():
    F = np.zeros((10, 10))
    with pytest.raises(KuantValueError):
        quantile_turnover(F, n_quantiles=1)


# ---------- summaries render ------------------------------------------


def test_all_summaries_contain_type_name():
    F = np.random.default_rng(0).standard_normal((100, 20))
    R = np.random.default_rng(1).standard_normal((100, 20))
    assert "FactorICResult" in factor_ic(F, R).summary()
    assert "RankAutocorrResult" in factor_rank_autocorr(F).summary()
    assert "QuantileReturnsResult" in mean_return_by_quantile(F, R).summary()
    assert "QuantileSpreadResult" in quantile_spread(F, R).summary()
    assert "QuantileTurnoverResult" in quantile_turnover(F).summary()
