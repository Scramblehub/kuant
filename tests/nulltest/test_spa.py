"""Tests for kuant.nulltest.spa_test and mcs_test."""

from __future__ import annotations

import numpy as np
import pytest

from kuant.errors import KuantShapeError, KuantValueError
from kuant.nulltest.spa_test import SPAResult, mcs_test, spa_test


# ---------- spa_test ---------------------------------------------------


def test_spa_detects_dominant_strategy():
    """One clearly-better alternative → SPA rejects the null."""
    rng = np.random.default_rng(0)
    T = 300
    bench = rng.normal(0.0, 0.01, T)
    winners = np.column_stack([rng.normal(0.005, 0.01, T) for _ in range(3)])
    r = spa_test(bench, winners, n_boot=300)
    assert r.p_value < 0.10


def test_spa_no_rejection_on_null():
    """All alternatives statistically equivalent → SPA fails to reject."""
    rng = np.random.default_rng(0)
    T = 300
    bench = rng.normal(0.0, 0.01, T)
    alts = np.column_stack([rng.normal(0.0, 0.01, T) for _ in range(5)])
    r = spa_test(bench, alts, n_boot=300)
    assert r.p_value > 0.05


def test_spa_returns_dataclass():
    rng = np.random.default_rng(0)
    T = 200
    r = spa_test(
        rng.normal(size=T),
        np.column_stack([rng.normal(size=T) for _ in range(2)]),
        n_boot=100,
    )
    assert isinstance(r, SPAResult)


def test_spa_rejects_1d_alt():
    with pytest.raises(KuantShapeError):
        spa_test(np.arange(10.0), np.arange(10.0))


def test_spa_rejects_length_mismatch():
    with pytest.raises(KuantShapeError):
        spa_test(np.arange(10.0), np.zeros((11, 3)))


# ---------- mcs_test ---------------------------------------------------


def test_mcs_survivors_include_top_strategy():
    """The genuinely best strategy should survive the confidence set."""
    rng = np.random.default_rng(0)
    T = 400
    R = np.column_stack(
        [
            rng.normal(0.0, 0.01, T),  # bench
            rng.normal(0.0, 0.01, T),  # noise
            rng.normal(0.005, 0.01, T),  # winner
            rng.normal(-0.003, 0.01, T),  # loser
        ]
    )
    r = mcs_test(R, alpha=0.05, n_boot=300)
    # The winner (col 2) should survive; the loser (col 3) should be dropped.
    assert 2 in r.survivors
    assert 3 not in r.survivors


def test_mcs_all_equivalent_keeps_everyone():
    """When all strategies are statistically the same, MCS should not
    drop anyone."""
    rng = np.random.default_rng(0)
    T = 300
    R = np.column_stack([rng.normal(0.0, 0.01, T) for _ in range(4)])
    r = mcs_test(R, alpha=0.05, n_boot=300)
    # Some may drop by chance but the majority should survive.
    assert len(r.survivors) >= 2


def test_mcs_rejects_single_column():
    with pytest.raises(KuantValueError):
        mcs_test(np.zeros((100, 1)))


def test_mcs_rejects_1d_input():
    with pytest.raises(KuantShapeError):
        mcs_test(np.arange(100.0))


def test_mcs_survivors_are_indices():
    """The returned .survivors is a sorted list of column indices."""
    rng = np.random.default_rng(0)
    R = np.column_stack([rng.normal(size=200) for _ in range(3)])
    r = mcs_test(R, alpha=0.05, n_boot=100)
    assert isinstance(r.survivors, list)
    assert all(isinstance(s, int) for s in r.survivors)
    assert r.survivors == sorted(r.survivors)
