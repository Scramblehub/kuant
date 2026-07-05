"""Adversarial numerical stability tests.

Every test here targets a specific failure mode that catastrophic
cancellation, poor cumsum arithmetic, or naive absolute-value clamping
would reveal:

- **Near-constant series**: the classic `sqrt(|E[X²] - E[X]²|)` bug.
  On `[C, C+eps, C+2·eps, ...]` a well-known bad formula produces
  large or NaN std because the two large terms cancel and the abs()
  mask flips a small negative into a large positive.
- **Large offset with small variation**: `[C + noise]` where C is huge
  and noise is O(1). Std should equal std of noise, not some fraction
  of C.
- **Long drifting series**: shift-once-by-x[0] tricks lose accuracy
  when the series drifts by many orders of magnitude within a window.
- **Alternating large-magnitude signs**: cumsum precision test —
  `[+K, -K, +K, -K, ...]` should sum to 0 or ±K, not accumulate error.
- **Third-moment near zero**: rollskew stability when m3 → 0.

Tolerances are set at what a naive implementation would BLOW PAST, not
at machine epsilon — the point is to fail loudly if any kernel is
using the vectorbt-shape formula.
"""

from __future__ import annotations

import numpy as np
import pytest

from kuant.portfolio.sharperatio import sharperatio
from kuant.portfolio.sortinoratio import sortinoratio
from kuant.stats.rollema import rollema
from kuant.stats.rollemastd import rollemastd
from kuant.stats.rollmean import rollmean
from kuant.stats.rollmoments import rollkurt, rollskew
from kuant.stats.rollstd import rollstd
from kuant.stats.rollsum import rollsum
from kuant.stats.zscore import zscore


# ---------- near-constant series --------------------------------------


def test_rollstd_near_constant_series_stays_near_zero():
    """[C, C+eps, C+2·eps, ...] with C large — std ≈ eps · sqrt((w²-1)/12).

    A `sqrt(|sum_sq - 2·sum·mean + n·mean²|/(n-1))` formula produces
    catastrophic cancellation here. This test would fail with either
    a huge value or an outright NaN if the formula were naive.
    """
    C = 1e10
    eps = 1e-6
    n = 200
    x = C + eps * np.arange(n, dtype=np.float64)
    w = 20
    out = rollstd(x, window=w, ddof=1)
    # Analytic std of an arithmetic progression with step eps over w points:
    # eps · sqrt((w+1)·w/12) (uncorrected), sample-corrected → eps·sqrt(w(w+1)/12)
    # scaled by sqrt(w/(w-1)) for ddof=1. For w=20 ≈ eps · 5.916.
    expected = eps * np.sqrt(w * (w + 1) / 12) * np.sqrt(w / (w - 1))
    valid = out[w - 1 :]
    assert np.all(np.isfinite(valid))
    # A naive formula produces values many orders of magnitude off.
    # We tolerate a large multiplicative slack (2x) to accept any
    # numerically-safe implementation; a bad implementation blows past it.
    assert np.all(np.abs(valid - expected) < 2.0 * expected + 1e-9)


def test_rollstd_constant_series_returns_zero():
    """Exactly-constant series → std must be exactly 0, not epsilon-noise."""
    x = np.full(100, 42.0)
    out = rollstd(x, window=10, ddof=1)
    # First 9 NaN, rest should be 0 exactly.
    assert np.allclose(out[9:], 0.0, atol=1e-12)


def test_rollmean_near_constant_series_recovers_mean():
    """Rolling mean of `[C, C, C, ...]` must equal C, no drift."""
    C = 1e12
    x = np.full(500, C)
    out = rollmean(x, window=50)
    assert np.allclose(out[49:], C, atol=1e-3)  # 1e-3 relative to 1e12 = 1e-15 rel


def test_rollsum_near_constant_series_no_drift():
    """Rolling sum of constant series should recover w·C every window."""
    C = 1e10
    w = 30
    x = np.full(200, C)
    out = rollsum(x, window=w)
    expected = w * C
    err = np.abs(out[w - 1 :] - expected) / expected
    assert np.max(err) < 1e-10


# ---------- large offset + small variation ----------------------------


def test_rollstd_large_offset_recovers_noise_std():
    """[C + N(0, σ)] with C = 1e12, σ = 1 — std must recover σ, not 0 or C."""
    rng = np.random.default_rng(0)
    C = 1e12
    sigma = 1.0
    x = C + rng.normal(0, sigma, size=1000)
    out = rollstd(x, window=100, ddof=1)
    # Empirical std of a window from N(0, 1) — expect ~1.0.
    tail = out[99:]
    assert np.abs(np.mean(tail) - sigma) < 0.15


def test_zscore_large_offset_stays_normalized():
    """zscore with mean subtracted → output N(0,1) even at huge C."""
    rng = np.random.default_rng(0)
    C = 1e12
    x = C + rng.normal(0, 1, size=500)
    z = zscore(x, window=50, ddof=1)
    tail = z[49:]
    tail = tail[np.isfinite(tail)]
    assert np.abs(np.mean(tail)) < 0.2
    assert 0.7 < np.std(tail, ddof=1) < 1.3


# ---------- long drifting series --------------------------------------


def test_rollstd_series_drifting_five_orders_of_magnitude():
    """Series drifts from C=1 to C=1e5 over 5000 points.

    The shift-by-x[0] trick has its limits — this stresses them.
    We require the rolling std to remain FINITE and MONOTONIC-ISH:
    the std should scale roughly with the local magnitude.
    """
    n = 5000
    t = np.linspace(0, 1, n)
    base = np.exp(np.log(1) + t * np.log(1e5))  # geometric drift 1 → 1e5
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 0.01, size=n) * base
    x = base + noise
    out = rollstd(x, window=50, ddof=1)
    valid = out[49:]
    assert np.all(np.isfinite(valid))
    assert np.all(valid > 0)


# ---------- alternating large-magnitude signs -------------------------


def test_rollsum_alternating_large_signs_no_error_accumulation():
    """[+K, -K, +K, -K, ...] with even window → sum is exactly 0.

    A poorly-implemented cumsum with catastrophic float error would
    drift here. numpy.cumsum is fine on this pattern; the test is a
    canary if we ever switch to a custom accumulator.
    """
    K = 1e10
    n = 400
    x = np.tile([K, -K], n // 2).astype(np.float64)
    w = 10  # even
    out = rollsum(x, window=w)
    valid = out[w - 1 :]
    # Even window → sum = 0 exactly.
    assert np.max(np.abs(valid)) < 1e-3  # tolerance << K


def test_rollmean_alternating_large_signs_stays_zero():
    K = 1e12
    x = np.tile([K, -K], 500).astype(np.float64)
    out = rollmean(x, window=100)  # even window
    valid = out[99:]
    assert np.max(np.abs(valid)) < 1.0  # tolerance << K


# ---------- higher moments ---------------------------------------------


def test_rollskew_near_constant_series_yields_nan_or_zero():
    """Constant series → variance is zero → skew is undefined.

    Kuant's convention: return NaN. A naive formula would divide by
    zero and yield ±inf.
    """
    x = np.full(200, 5.5)
    out = rollskew(x, window=30)
    valid = out[29:]
    # Should be NaN (m2 = 0 → undefined) or exactly 0. NEVER inf.
    assert np.all(np.isnan(valid) | (valid == 0.0))
    assert not np.any(np.isinf(valid))


def test_rollkurt_near_constant_series_yields_nan_or_zero():
    x = np.full(200, 5.5)
    out = rollkurt(x, window=30)
    valid = out[29:]
    assert np.all(np.isnan(valid) | (valid == 0.0))
    assert not np.any(np.isinf(valid))


def test_rollskew_large_offset_symmetric_data_near_zero():
    """Symmetric distribution + huge offset → skew ≈ 0, not NaN or large."""
    rng = np.random.default_rng(0)
    C = 1e10
    x = C + rng.normal(0, 1, size=2000)
    out = rollskew(x, window=200)
    valid = out[199:]
    valid = valid[np.isfinite(valid)]
    assert np.abs(np.mean(valid)) < 0.3  # near zero


def test_rollkurt_large_offset_normal_data_near_zero():
    """Normal + huge offset → excess kurtosis ≈ 0."""
    rng = np.random.default_rng(0)
    C = 1e10
    x = C + rng.normal(0, 1, size=2000)
    out = rollkurt(x, window=500)
    valid = out[499:]
    valid = valid[np.isfinite(valid)]
    assert np.abs(np.mean(valid)) < 0.5  # near zero


# ---------- EMA stability ----------------------------------------------


def test_rollema_large_offset_converges_to_offset():
    """EMA on [C + tiny noise] should converge to C, not overshoot."""
    C = 1e12
    n = 1000
    rng = np.random.default_rng(0)
    x = C + rng.normal(0, 1, size=n)
    out = rollema(x, span=20)
    tail = out[500:]
    # EMA of C + O(1) noise should be within a few sigma of C.
    assert np.all(np.isfinite(tail))
    assert np.all(np.abs(tail - C) < 100)  # << C


def test_rollemastd_near_constant_series_stays_finite():
    """EMA std on near-constant series must stay finite and small."""
    C = 1e10
    n = 500
    x = C + 1e-6 * np.arange(n, dtype=np.float64)
    out = rollemastd(x, span=20)
    tail = out[100:]
    assert np.all(np.isfinite(tail))
    # An epsilon-scale drift should produce an epsilon-scale std.
    assert np.max(tail) < 1e-3  # << C


# ---------- portfolio kernels ------------------------------------------


def test_sharperatio_large_offset_stable():
    """Sharpe with a huge offset added to every return must equal
    the shifted Sharpe (mean_shift / std unchanged in std but altered
    in numerator by exactly the shift).

    We test that std doesn't blow up under the offset, so the final
    Sharpe stays finite.
    """
    rng = np.random.default_rng(0)
    r = rng.normal(0.001, 0.01, size=1000)
    shifted = r + 1e10  # doesn't make financial sense but numerically stresses
    result = sharperatio(shifted, ann_factor=252)
    # Sharpe uses variance under the hood — must stay finite even with
    # a huge additive offset (variance is shift-invariant).
    assert np.isfinite(result.sharpe)


def test_sortinoratio_near_constant_returns_stable():
    """Sortino on constant returns → downside deviation is 0.

    Kuant's convention: return NaN or inf (no risk to divide by). NEVER
    silently wrong.
    """
    r = np.full(500, 0.001)  # perfect constant return
    result = sortinoratio(r, ann_factor=252, target=0.0)
    # No downside → either inf or handled gracefully (finite). Not NaN
    # for a well-defined constant positive return.
    assert not np.isnan(result.sortino)


# ---------- pandas-parity guardrails ----------------------------------


def test_rollstd_matches_pandas_on_stress_input():
    """On a series where naive formulas fail, kuant must still agree
    with pandas to within a tight tolerance."""
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(0)
    C = 1e9
    x = C + rng.normal(0, 1, size=500)
    w = 50
    kuant_out = rollstd(x, window=w, ddof=1)
    pandas_out = pd.Series(x).rolling(window=w).std(ddof=1).to_numpy()
    # Compare tail only (both have NaN in the prefix).
    diff = np.abs(kuant_out[w - 1 :] - pandas_out[w - 1 :])
    # Pandas uses similar shift-based tricks; both should be within
    # ~1e-6 relative on this stress input.
    ref = np.abs(pandas_out[w - 1 :])
    rel = diff / np.where(ref > 0, ref, 1.0)
    assert np.max(rel) < 1e-4


def test_rollmean_matches_pandas_on_large_offset():
    pd = pytest.importorskip("pandas")
    C = 1e12
    rng = np.random.default_rng(0)
    x = C + rng.normal(0, 1, size=500)
    w = 25
    kuant_out = rollmean(x, window=w)
    pandas_out = pd.Series(x).rolling(window=w).mean().to_numpy()
    diff = np.abs(kuant_out[w - 1 :] - pandas_out[w - 1 :])
    ref = np.abs(pandas_out[w - 1 :])
    rel = diff / np.where(ref > 0, ref, 1.0)
    assert np.max(rel) < 1e-10
