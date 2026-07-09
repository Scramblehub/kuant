"""Tests for kuant.options v0.6.0 batch 8: exotic option kernels."""

from __future__ import annotations

import numpy as np

from kuant.core._bs_common import finalize, prepare_bs
from kuant.core.normcdf import normcdf
from kuant.options import (
    chooserprice,
    digitalprice,
    gapprice,
    lookbackprice,
    powerprice,
)


def _bscall(S, K, T, r, sigma, q=0.0):
    c = prepare_bs(S, K, T, r, sigma, q)
    xp = c.xp
    price = c.S_safe * xp.exp(-c.q * c.T_safe) * normcdf(c.d1) - c.K_safe * xp.exp(
        -c.r * c.T_safe
    ) * normcdf(c.d2)
    c.out = xp.where((c.T > 0) & (c.sigma > 0) & (c.S > 0) & (c.K > 0), price, c.out)
    return finalize(c.out)


def _bsput(S, K, T, r, sigma, q=0.0):
    c = prepare_bs(S, K, T, r, sigma, q)
    xp = c.xp
    price = c.K_safe * xp.exp(-c.r * c.T_safe) * normcdf(-c.d2) - c.S_safe * xp.exp(
        -c.q * c.T_safe
    ) * normcdf(-c.d1)
    c.out = xp.where((c.T > 0) & (c.sigma > 0) & (c.S > 0) & (c.K > 0), price, c.out)
    return finalize(c.out)


# ---------- digitalprice ---------------------------------------------


class TestDigital:
    def test_atm_gaussian_reference(self):
        # At S=K, digital call = 0.5*exp(-r*T) minus small drift adjustment.
        p = digitalprice(100, 100, 1.0, 0.05, 0.20)
        assert 0.4 < p < 0.6

    def test_put_call_parity(self):
        # digital call + digital put = cash * exp(-r*T)
        cash = 1.0
        r, T = 0.05, 1.0
        c = digitalprice(100, 100, T, r, 0.20, cash=cash)
        p = digitalprice(100, 100, T, r, 0.20, cash=cash, is_call=False)
        assert abs(c + p - cash * np.exp(-r * T)) < 1e-10

    def test_deep_itm_call_approaches_cash(self):
        # S far above K, near-zero vol: digital call ~ cash * exp(-r*T)
        c = digitalprice(200, 100, 1.0, 0.05, 0.05)
        assert abs(c - np.exp(-0.05)) < 1e-6

    def test_deep_otm_call_near_zero(self):
        c = digitalprice(50, 100, 1.0, 0.05, 0.05)
        assert c < 0.05


# ---------- gapprice --------------------------------------------------


class TestGap:
    def test_triggers_match_recovers_vanilla(self):
        # K_trigger = K_payoff -> gap = vanilla
        g = gapprice(100, 100, 100, 1.0, 0.05, 0.20)
        v = _bscall(100, 100, 1.0, 0.05, 0.20)
        assert abs(g - v) < 1e-10

    def test_gap_pays_more_than_vanilla_when_payoff_below_trigger(self):
        # K_trigger=100, K_payoff=90: pays S_T - 90 when triggered
        g = gapprice(100, 100, 90, 1.0, 0.05, 0.20)
        v = _bscall(100, 100, 1.0, 0.05, 0.20)
        assert g > v

    def test_put(self):
        # Put version should be positive at ATM
        p = gapprice(100, 100, 100, 1.0, 0.05, 0.20, is_call=False)
        assert p > 0


# ---------- lookbackprice --------------------------------------------


class TestLookback:
    def test_fresh_call_positive_greater_than_vanilla(self):
        lc = lookbackprice(100, 100, 1.0, 0.05, 0.20)
        v = _bscall(100, 100, 1.0, 0.05, 0.20)
        assert lc > v

    def test_fresh_put_positive(self):
        lp = lookbackprice(100, 100, 1.0, 0.05, 0.20, is_call=False)
        assert lp > 0

    def test_mc_agreement_within_stderr(self):
        # Cross-check against 50k-path MC to within ~5%
        np.random.seed(0)
        n_paths, n_steps = 50_000, 252
        dt = 1.0 / n_steps
        sigma, r, q, S0 = 0.20, 0.05, 0.0, 100.0
        Z = np.random.standard_normal((n_paths, n_steps))
        paths = S0 * np.exp(
            np.cumsum((r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z, axis=1)
        )
        S_min = paths.min(axis=1)
        S_T = paths[:, -1]
        mc_call = np.exp(-r) * np.mean(S_T - S_min)
        analytic_call = lookbackprice(S0, S0, 1.0, r, sigma, q)
        assert abs(mc_call - analytic_call) / analytic_call < 0.07


# ---------- chooserprice ---------------------------------------------


class TestChooser:
    def test_t_choose_zero_equals_max_call_put(self):
        # Chooser at t_choose~0 is picking now: it's just max(call, put)
        ch = chooserprice(100, 100, 1.0, 0.001, 0.05, 0.20)
        c = _bscall(100, 100, 1.0, 0.05, 0.20)
        p = _bsput(100, 100, 1.0, 0.05, 0.20)
        assert abs(ch - max(c, p)) < 1e-3

    def test_t_choose_full_equals_call_plus_put(self):
        # Chooser at t_choose~T: holder must "prepay" for both, so it
        # approaches call + put (a straddle).
        ch = chooserprice(100, 100, 1.0, 0.999, 0.05, 0.20)
        c = _bscall(100, 100, 1.0, 0.05, 0.20)
        p = _bsput(100, 100, 1.0, 0.05, 0.20)
        assert abs(ch - (c + p)) < 0.02

    def test_chooser_bounded_by_straddle(self):
        # Chooser value <= straddle at any t_choose <= T
        ch = chooserprice(100, 100, 1.0, 0.5, 0.05, 0.20)
        straddle = _bscall(100, 100, 1.0, 0.05, 0.20) + _bsput(100, 100, 1.0, 0.05, 0.20)
        assert ch <= straddle + 1e-10


# ---------- powerprice ------------------------------------------------


class TestPower:
    def test_n_equals_one_recovers_vanilla(self):
        p1 = powerprice(100, 100, 1.0, 0.05, 0.20, n=1.0)
        v = _bscall(100, 100, 1.0, 0.05, 0.20)
        assert abs(p1 - v) < 1e-10

    def test_squared_positive(self):
        p2 = powerprice(100, 10000, 1.0, 0.05, 0.20, n=2.0)
        # Squared call ATM (K = S^2) should be well above vanilla.
        assert p2 > 100

    def test_put_positive_atm(self):
        pp = powerprice(100, 10000, 1.0, 0.05, 0.20, n=2.0, is_call=False)
        assert pp > 0

    def test_higher_n_amplifies_call(self):
        # For S=100, K=S^n at each n, call price scales up with n.
        p2 = powerprice(100, 100**2, 1.0, 0.05, 0.20, n=2.0)
        p3 = powerprice(100, 100**3, 1.0, 0.05, 0.20, n=3.0)
        # For a naively identical-moneyness comparison, higher-power calls
        # should have larger notional values.
        assert p3 > p2 * 100 / 5  # loose sanity check
