"""Tests for kuant.nulltest.bootstrap."""

from __future__ import annotations

import numpy as np
import pytest

from kuant.errors import KuantValueError
from kuant.nulltest.bootstrap import (
    BootstrapICResult,
    bootstrap_ic,
    stationary_bootstrap,
)


# ---------- stationary_bootstrap ---------------------------------------


def test_stationary_bootstrap_shape_and_values():
    """Output shape matches input; every element is from the source."""
    x = np.arange(50.0)
    y = stationary_bootstrap(x, mean_block_length=5, seed=0)
    assert y.shape == x.shape
    assert set(y.tolist()).issubset(set(x.tolist()))


def test_stationary_bootstrap_deterministic_for_seed():
    x = np.arange(50.0)
    a = stationary_bootstrap(x, mean_block_length=5, seed=42)
    b = stationary_bootstrap(x, mean_block_length=5, seed=42)
    assert np.array_equal(a, b)


def test_stationary_bootstrap_different_seeds_differ():
    x = np.arange(100.0)
    a = stationary_bootstrap(x, mean_block_length=5, seed=1)
    b = stationary_bootstrap(x, mean_block_length=5, seed=2)
    assert not np.array_equal(a, b)


def test_stationary_bootstrap_rejects_bad_block_length():
    with pytest.raises(KuantValueError):
        stationary_bootstrap(np.arange(10.0), mean_block_length=0.5)


def test_stationary_bootstrap_rejects_too_short():
    with pytest.raises(KuantValueError):
        stationary_bootstrap(np.array([1.0]), mean_block_length=1)


# ---------- bootstrap_ic ------------------------------------------------


def test_bootstrap_ic_detects_real_signal():
    rng = np.random.default_rng(0)
    T = 500
    sig = rng.standard_normal(T)
    ret = 0.15 * sig + 0.5 * rng.standard_normal(T)
    r = bootstrap_ic(sig, ret, n_boot=300)
    assert r.point_estimate > 0.1
    assert r.p_value < 0.05


def test_bootstrap_ic_null_signal_has_ci_containing_zero():
    rng = np.random.default_rng(0)
    T = 500
    sig = rng.standard_normal(T)
    ret = rng.standard_normal(T)
    r = bootstrap_ic(sig, ret, n_boot=300)
    assert r.ci_low < 0 < r.ci_high


def test_bootstrap_ic_returns_dataclass():
    rng = np.random.default_rng(0)
    r = bootstrap_ic(rng.standard_normal(200), rng.standard_normal(200), n_boot=100)
    assert isinstance(r, BootstrapICResult)


def test_bootstrap_ic_ci_ordering():
    rng = np.random.default_rng(0)
    r = bootstrap_ic(rng.standard_normal(200), rng.standard_normal(200), n_boot=100)
    assert r.ci_low <= r.ci_high


def test_bootstrap_ic_p_value_in_unit_interval():
    rng = np.random.default_rng(0)
    r = bootstrap_ic(rng.standard_normal(200), rng.standard_normal(200), n_boot=100)
    assert 0.0 <= r.p_value <= 1.0


def test_bootstrap_ic_rejects_shape_mismatch():
    with pytest.raises(Exception):
        bootstrap_ic(np.arange(100.0), np.arange(50.0))


def test_bootstrap_ic_rejects_insufficient_clean_rows():
    with pytest.raises(KuantValueError):
        bootstrap_ic(np.array([1.0, np.nan]), np.array([1.0, np.nan]))
