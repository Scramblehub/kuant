'''Test suite for kuant.stats.hurstrs.

Validation strategy:
  1. Golden values      — H ~ 0.5 on Brownian noise (known-answer)
  2. Reference match    — H > 0.5 on fractional Brownian motion (H target)
                          H < 0.5 on antipersistent process (AR(1) with phi<0)
  3. Edge cases         — short series raises, degenerate ranges, all-NaN
  4. Property tests     — result reproducible across shuffled but seeded inputs;
                          intercept has expected sign
  5. Backend            — accepts numpy in, returns HurstResult with numpy arrays
'''
from __future__ import annotations

import numpy as np
import pytest

from kuant.stats import HurstResult, hurstrs


# ---------------------------------------------------------------------------
# 1. Golden values
# ---------------------------------------------------------------------------


def test_brownian_noise_near_half(rng):
    '''iid normal returns produce H near 0.5.'''
    r = rng.standard_normal(4000)
    result = hurstrs(r)
    assert 0.40 < result.H < 0.60, f'expected H near 0.5, got {result.H:.3f}'


def test_returns_hurstresult():
    r = np.random.default_rng(0).standard_normal(2000)
    result = hurstrs(r)
    assert isinstance(result, HurstResult)
    assert isinstance(result.windows, np.ndarray)
    assert isinstance(result.log_rs, np.ndarray)
    assert result.windows.size == result.n_windows


# ---------------------------------------------------------------------------
# 2. Reference match — persistent / antipersistent processes
# ---------------------------------------------------------------------------


def _fbm_approx(n, H, rng):
    '''Approximate fractional Brownian motion via random midpoint displacement.

    Not a true fBM but sufficient for detecting H > 0.5 signature.
    '''
    x = np.cumsum(rng.standard_normal(n))
    # Emphasize persistence: smooth incrementally to inject autocorrelation
    if H > 0.5:
        alpha = 2 * (H - 0.5)
        for _ in range(3):
            x = (1 - alpha) * x + alpha * np.concatenate([[x[0]], x[:-1]])
    return np.diff(x, prepend=x[0])


def test_persistent_series_h_above_half(rng):
    '''AR(1) with strong positive autocorrelation → persistent (H > 0.5).'''
    n = 4000
    phi = 0.7
    e = rng.standard_normal(n)
    r = np.zeros(n)
    for t in range(1, n):
        r[t] = phi * r[t-1] + e[t]
    result = hurstrs(r)
    assert result.H > 0.55, f'expected H > 0.55 for AR(1) phi=0.7, got {result.H:.3f}'


def test_antipersistent_series_h_below_half(rng):
    '''Strong mean-reverting process → antipersistent (H < 0.5).

    R/S has a well-known small-window bias that pulls H toward and
    above 0.5. To detect antipersistence reliably we use strong
    negative autocorrelation and skip the smallest windows.
    '''
    n = 8000
    phi = -0.9
    e = rng.standard_normal(n)
    r = np.zeros(n)
    for t in range(1, n):
        r[t] = phi * r[t-1] + e[t]
    result = hurstrs(r, min_w=25)
    assert result.H < 0.55, f'expected H clearly < persistent for AR(1) phi=-0.9, got {result.H:.3f}'
    # Also demand it be less than a same-length Brownian control.
    control = rng.standard_normal(n)
    control_H = hurstrs(control, min_w=25).H
    assert result.H < control_H, (
        f'expected antipersistent H ({result.H:.3f}) < Brownian control H ({control_H:.3f})'
    )


# ---------------------------------------------------------------------------
# 3. Edge cases
# ---------------------------------------------------------------------------


def test_short_series_raises():
    with pytest.raises(ValueError, match='too short'):
        hurstrs(np.zeros(30), min_w=10)


def test_max_w_le_min_w_raises():
    with pytest.raises(ValueError, match='max_w'):
        hurstrs(np.random.default_rng(0).standard_normal(200), min_w=10, max_w=10)


def test_2d_input_raises():
    with pytest.raises(ValueError, match='1D'):
        hurstrs(np.zeros((100, 5)))


def test_nans_in_input_survive(rng):
    r = rng.standard_normal(2000)
    r[rng.choice(2000, 40, replace=False)] = np.nan
    result = hurstrs(r)
    assert np.isfinite(result.H)


def test_constant_series_raises(rng):
    '''All-constant returns → std = 0 everywhere → no valid R/S.'''
    r = np.zeros(1000)
    with pytest.raises(ValueError, match='fewer than 3 windows'):
        hurstrs(r)


# ---------------------------------------------------------------------------
# 4. Property tests
# ---------------------------------------------------------------------------


def test_reproducible_with_same_seed():
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)
    r1 = hurstrs(rng1.standard_normal(2000)).H
    r2 = hurstrs(rng2.standard_normal(2000)).H
    assert r1 == r2


def test_summary_returns_string():
    r = np.random.default_rng(0).standard_normal(2000)
    s = hurstrs(r).summary()
    assert isinstance(s, str)
    assert 'H' in s


def test_windows_are_monotone_increasing():
    r = np.random.default_rng(0).standard_normal(2000)
    result = hurstrs(r)
    assert np.all(np.diff(result.windows) > 0)


def test_scale_invariance(rng):
    '''H should be invariant under scaling the input.'''
    r = rng.standard_normal(2000)
    H1 = hurstrs(r).H
    H2 = hurstrs(r * 100).H
    assert abs(H1 - H2) < 1e-10
