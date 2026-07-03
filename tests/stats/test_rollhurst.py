'''Test suite for kuant.stats.rollhurst.'''
from __future__ import annotations

import numpy as np
import pytest

from kuant.stats import rollhurst


def test_length_matches_input(rng):
    r = rng.standard_normal(600)
    H_t = rollhurst(r, window=252)
    assert H_t.size == 600


def test_first_window_minus_1_are_nan(rng):
    r = rng.standard_normal(600)
    H_t = rollhurst(r, window=252)
    assert np.all(np.isnan(H_t[:251]))


def test_finite_after_warmup(rng):
    r = rng.standard_normal(600)
    H_t = rollhurst(r, window=252)
    finite = np.isfinite(H_t[251:])
    assert finite.sum() >= 300


def test_brownian_noise_median_near_half(rng):
    r = rng.standard_normal(1500)
    H_t = rollhurst(r, window=300)
    med = np.nanmedian(H_t)
    assert 0.40 < med < 0.60


def test_2d_input_raises():
    with pytest.raises(ValueError, match='1D'):
        rollhurst(np.zeros((100, 5)))


def test_window_below_floor_raises():
    with pytest.raises(ValueError, match='window'):
        rollhurst(np.zeros(300), window=10, min_w=8)


def test_reproducible():
    rng1 = np.random.default_rng(7)
    rng2 = np.random.default_rng(7)
    a = rollhurst(rng1.standard_normal(600), window=252)
    b = rollhurst(rng2.standard_normal(600), window=252)
    np.testing.assert_array_equal(a, b)


def test_python_list_input(rng):
    r = list(rng.standard_normal(400))
    H_t = rollhurst(r, window=252)
    assert H_t.size == 400
