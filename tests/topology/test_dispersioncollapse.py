"""Tests for kuant.topology.dispersioncollapse."""

from __future__ import annotations

import numpy as np
import pytest

from kuant.errors import KuantShapeError, KuantValueError
from kuant.topology.dispersioncollapse import dispersioncollapse


# ---------- shape and dtype invariants -------------------------------------


def test_returns_boolean_1d_of_correct_length():
    rng = np.random.default_rng(0)
    R = rng.normal(size=(200, 10))
    c = dispersioncollapse(R, window=30, n_consecutive=3)
    assert c.shape == (200,)
    assert c.dtype == bool


def test_return_dispersion_flag_gives_tuple():
    R = np.random.default_rng(0).normal(size=(100, 5))
    c, disp = dispersioncollapse(R, window=20, return_dispersion=True)
    assert c.shape == (100,) and c.dtype == bool
    assert disp.shape == (100,) and disp.dtype == np.float64


# ---------- warm-up + persistence semantics --------------------------------


def test_warmup_never_fires():
    rng = np.random.default_rng(0)
    R = rng.normal(size=(100, 10))
    c = dispersioncollapse(R, window=30, n_consecutive=5)
    # Anchor needs (window-1) history + (n_consecutive-1) prior low bars.
    # Anything before window + n_consecutive - 2 must be False by construction.
    warmup_end = 30 + 5 - 2
    assert not c[:warmup_end].any()


def test_short_n_consecutive_still_needs_low():
    """n_consecutive=1 → fires whenever the current bar is low."""
    rng = np.random.default_rng(0)
    R = rng.normal(size=(500, 20))
    c1 = dispersioncollapse(R, window=100, quantile=0.20, n_consecutive=1)
    c5 = dispersioncollapse(R, window=100, quantile=0.20, n_consecutive=5)
    # Longer persistence requirement → fewer or equal fires.
    assert c5.sum() <= c1.sum()


# ---------- semantic correctness -------------------------------------------


def test_lockstep_block_fires_signal():
    """Insert a block where all names move identically → dispersion → 0."""
    rng = np.random.default_rng(0)
    n_names = 20
    normal_lo = rng.normal(0, 0.02, size=(200, n_names))
    normal_hi = rng.normal(0, 0.02, size=(200, n_names))
    # Lockstep block: all names get the same daily return.
    lockstep_days = rng.normal(0, 0.02, size=50)
    lockstep = np.tile(lockstep_days[:, None], (1, n_names))
    R = np.vstack([normal_lo, lockstep, normal_hi])

    c = dispersioncollapse(R, window=63, quantile=0.20, n_consecutive=5)
    # The lockstep window starts at index 200.
    fire_rate_lockstep = c[200:250].mean()
    fire_rate_pre = c[63:200].mean()
    assert fire_rate_lockstep > fire_rate_pre + 0.20


def test_uniform_random_rarely_fires():
    """Homogeneous noise → signal fires at most at its own definitional rate."""
    rng = np.random.default_rng(1)
    R = rng.normal(0, 0.02, size=(1000, 30))
    c = dispersioncollapse(R, window=63, quantile=0.20, n_consecutive=5)
    # With q=0.20 and n_consecutive=5, expected fire rate on i.i.d. dispersion
    # is at most 0.20^5 ≈ 0.03%. Give plenty of slack for finite-sample noise.
    assert c.mean() < 0.05


# ---------- NaN policy -----------------------------------------------------


def test_nan_row_produces_nan_dispersion():
    R = np.full((50, 3), np.nan)
    R[10, :] = 0.01  # not enough for cross-sectional std (need >=2 finite? this has 3)
    R[20, 0] = 0.01  # only 1 finite → NaN dispersion at that row
    _, disp = dispersioncollapse(R, window=10, return_dispersion=True)
    assert not np.isfinite(disp[0])  # all-NaN row
    assert np.isfinite(disp[10])  # 3 finite values
    assert not np.isfinite(disp[20])  # 1 finite value


def test_partial_nan_still_dispatches():
    """Some NaN in the panel is fine; dispersion uses finite cells per row."""
    rng = np.random.default_rng(0)
    R = rng.normal(size=(200, 20))
    R[::10, :5] = np.nan  # occasional NaN block
    c = dispersioncollapse(R, window=30, quantile=0.20, n_consecutive=3)
    # Just verify no crash and shape is right.
    assert c.shape == (200,)


# ---------- error contract -------------------------------------------------


def test_reject_1d_input():
    with pytest.raises(KuantShapeError):
        dispersioncollapse(np.zeros(100), window=20)


def test_reject_zero_window():
    with pytest.raises(KuantValueError) as exc:
        dispersioncollapse(np.zeros((50, 5)), window=0)
    assert "window" in str(exc.value)


def test_reject_out_of_range_quantile():
    with pytest.raises(KuantValueError) as exc:
        dispersioncollapse(np.zeros((50, 5)), window=20, quantile=1.5)
    assert "quantile" in str(exc.value)


def test_reject_zero_n_consecutive():
    with pytest.raises(KuantValueError) as exc:
        dispersioncollapse(np.zeros((50, 5)), window=20, n_consecutive=0)
    assert "n_consecutive" in str(exc.value)
