"""Tests for kuant.causal (v0.6.0 batch 10)."""

from __future__ import annotations

import numpy as np
import pytest

from kuant.causal import (
    iv,
    pcalgo,
    rdd,
    synthcontrol,
)
from kuant.errors import KuantNumericWarning, KuantValueError


# ---------- synthcontrol ----------


def test_synthcontrol_recovers_att_and_weights():
    rng = np.random.default_rng(0)
    T, t_treat = 40, 25
    donors = rng.standard_normal((T, 5)) * 0.5
    true_w = np.array([0.6, 0.4, 0.0, 0.0, 0.0])
    y_pre = donors[:t_treat] @ true_w + rng.standard_normal(t_treat) * 0.05
    y_post = donors[t_treat:] @ true_w + 3.0 + rng.standard_normal(T - t_treat) * 0.05
    y = np.concatenate([y_pre, y_post])
    res = synthcontrol(y, donors, t_treat)
    assert abs(res.att - 3.0) < 0.3
    assert res.weights.sum() == pytest.approx(1.0, abs=1e-6)
    assert (res.weights >= -1e-9).all()
    # Top two weights should dominate
    assert res.weights[0] + res.weights[1] > 0.85


def test_synthcontrol_shape_error():
    with pytest.raises(KuantValueError):
        synthcontrol(np.zeros(30), np.zeros(30), t_treat=15)  # 1D donors


def test_synthcontrol_len_mismatch():
    with pytest.raises(KuantValueError):
        synthcontrol(np.zeros(30), np.zeros((25, 3)), t_treat=15)


# ---------- iv ----------


def test_iv_recovers_true_beta_when_ols_is_biased():
    rng = np.random.default_rng(1)
    n = 2000
    u = rng.standard_normal(n)
    v = rng.standard_normal(n)
    z = rng.standard_normal(n)
    x = 0.8 * z + v
    y = 2 + 3 * x + u + 0.6 * v  # OLS would be biased upward by 0.6

    res = iv(y, x, z)
    assert abs(res.beta[0] - 3.0) < 0.2
    assert res.f_stat_stage1 > 50


def test_iv_underidentified_error():
    rng = np.random.default_rng(2)
    n = 200
    x = rng.standard_normal((n, 2))  # 2 endog
    z = rng.standard_normal((n, 1))  # 1 instr
    y = rng.standard_normal(n)
    with pytest.raises(KuantValueError):
        iv(y, x, z)


def test_iv_weak_instrument_warning():
    rng = np.random.default_rng(3)
    n = 500
    u = rng.standard_normal(n)
    v = rng.standard_normal(n)
    z = rng.standard_normal(n)
    x = 0.02 * z + v  # near-zero first-stage relevance
    y = 2 + 3 * x + u + 0.6 * v
    with pytest.warns(KuantNumericWarning, match="KW-IV-WEAK-INSTRUMENT"):
        iv(y, x, z)


# ---------- rdd ----------


def test_rdd_recovers_true_jump():
    rng = np.random.default_rng(4)
    n = 3000
    x = rng.uniform(-1, 1, n)
    y = 0.5 + 0.3 * x + 2.5 * (x >= 0) + rng.standard_normal(n) * 0.2
    res = rdd(x, y, cutoff=0.0)
    assert abs(res.tau - 2.5) < 0.15
    assert res.n_left > 20 and res.n_right > 20


def test_rdd_no_jump_null():
    rng = np.random.default_rng(5)
    n = 2000
    x = rng.uniform(-1, 1, n)
    y = 0.5 + 0.3 * x + rng.standard_normal(n) * 0.2
    res = rdd(x, y, cutoff=0.0)
    assert abs(res.tau) < 0.15


def test_rdd_shape_error():
    with pytest.raises(KuantValueError):
        rdd(np.zeros(200), np.zeros(300), cutoff=0.0)


def test_rdd_min_clean_error():
    with pytest.raises(KuantValueError):
        rdd(np.zeros(10), np.zeros(10), cutoff=0.0)


# ---------- pcalgo ----------


def test_pcalgo_recovers_chain_skeleton():
    rng = np.random.default_rng(6)
    n = 800
    A = rng.standard_normal(n)
    B = 0.9 * A + rng.standard_normal(n) * 0.2
    C = 0.9 * B + rng.standard_normal(n) * 0.2
    D = rng.standard_normal(n)  # independent
    data = np.column_stack([A, B, C, D])
    res = pcalgo(data, max_order=2)
    # Chain A-B-C present; D disconnected; A-C removed by conditioning on B
    assert res.adj[0, 1] == 1
    assert res.adj[1, 2] == 1
    assert res.adj[0, 2] == 0
    assert res.adj[0, 3] == 0 and res.adj[1, 3] == 0 and res.adj[2, 3] == 0


def test_pcalgo_fork_creates_two_edges():
    rng = np.random.default_rng(7)
    n = 800
    A = rng.standard_normal(n)
    B = 0.8 * A + rng.standard_normal(n) * 0.3  # A -> B
    C = 0.8 * A + rng.standard_normal(n) * 0.3  # A -> C, independent given A
    data = np.column_stack([A, B, C])
    res = pcalgo(data, max_order=1)
    assert res.adj[0, 1] == 1 and res.adj[0, 2] == 1
    assert res.adj[1, 2] == 0  # B and C independent given A


def test_pcalgo_alpha_out_of_range():
    rng = np.random.default_rng(8)
    with pytest.raises(KuantValueError):
        pcalgo(rng.standard_normal((100, 3)), alpha=1.5)


def test_pcalgo_too_few_variables():
    with pytest.raises(KuantValueError):
        pcalgo(np.zeros((100, 1)))
