"""Tests for kuant.signals.icdecay."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scipy")

from kuant.errors import KuantNumericWarning, KuantValueError  # noqa: E402
from kuant.signals.icdecay import ICDecayResult, icdecay  # noqa: E402


# ---------- known-truth recovery ----------------------------------------


def test_positive_ic_when_signal_predicts_returns():
    """Synthetic setup: signal[t] predicts returns[t+1] via a lag-1 shift.
    icdecay correlates signal[t] with cumulative returns over [t+1..t+h]."""
    rng = np.random.default_rng(0)
    T = 500
    signal = rng.standard_normal(T)
    forward_ret = np.zeros(T)
    forward_ret[1:] = 0.10 * signal[:-1] + 0.02 * rng.standard_normal(T - 1)
    r = icdecay(signal, forward_ret, horizons=(1,))
    assert r.ic[0] > 0.5  # strong positive IC at h=1


def test_ic_decays_with_horizon():
    """Signal only predicts NEXT return; longer horizons dilute the edge
    via the added noise of the intervening periods."""
    rng = np.random.default_rng(0)
    T = 2000
    signal = rng.standard_normal(T)
    forward_ret = np.zeros(T)
    # Only returns[t+1] correlates with signal[t]; returns[t+2..] are independent.
    forward_ret[1:] = 0.10 * signal[:-1] + 0.5 * rng.standard_normal(T - 1)
    r = icdecay(signal, forward_ret, horizons=(1, 5, 21))
    # h=1 gets the raw signal; longer horizons dilute it with more noise.
    assert abs(r.ic[0]) > abs(r.ic[1])
    assert abs(r.ic[1]) >= abs(r.ic[2]) - 0.02


def test_zero_ic_on_pure_noise():
    """Signal uncorrelated with returns → IC near zero."""
    rng = np.random.default_rng(0)
    T = 1000
    signal = rng.standard_normal(T)
    forward_ret = rng.standard_normal(T)
    r = icdecay(signal, forward_ret, horizons=(1, 5, 21))
    # With T=1000, stderr at h=1 is ~1/√999 ≈ 0.03.
    # Pure noise should be well within a few stderr.
    assert abs(r.ic[0]) < 0.15


# ---------- return object contract --------------------------------------


def test_returns_dataclass():
    signal = np.arange(100.0)
    returns = np.arange(100.0)
    assert isinstance(icdecay(signal, returns, horizons=(1,)), ICDecayResult)


def test_arrays_have_horizon_length():
    signal = np.arange(100.0)
    returns = np.random.default_rng(0).standard_normal(100)
    r = icdecay(signal, returns, horizons=(1, 5, 21))
    assert r.ic.shape == (3,)
    assert r.ic_stderr.shape == (3,)
    assert r.ic_tstat.shape == (3,)
    assert r.n.shape == (3,)


def test_summary_string():
    signal = np.random.default_rng(0).standard_normal(200)
    returns = np.random.default_rng(1).standard_normal(200)
    r = icdecay(signal, returns, horizons=(1, 5))
    s = r.summary()
    assert "ICDecayResult" in s
    assert "horizon" in s
    assert "peak" in s


def test_to_parquet_roundtrip(tmp_path):
    pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    signal = np.random.default_rng(0).standard_normal(200)
    returns = np.random.default_rng(1).standard_normal(200)
    r = icdecay(signal, returns, horizons=(1, 5, 21))
    path = tmp_path / "ic.parquet"
    r.to_parquet(path)
    cols = set(pq.read_table(path).column_names)
    assert cols == {"horizon", "ic", "ic_stderr", "ic_tstat", "n"}


# ---------- stderr / t-stat -------------------------------------------


def test_tstat_is_ic_over_stderr():
    signal = np.random.default_rng(0).standard_normal(500)
    returns = 0.1 * signal + np.random.default_rng(1).standard_normal(500) * 0.5
    r = icdecay(signal, returns, horizons=(1,))
    assert abs(r.ic_tstat[0] - r.ic[0] / r.ic_stderr[0]) < 1e-9


def test_stderr_shrinks_with_sample_size():
    """Longer samples give tighter stderr (1/√n)."""
    rng = np.random.default_rng(0)
    small = icdecay(rng.standard_normal(200), rng.standard_normal(200), horizons=(1,))
    large = icdecay(rng.standard_normal(2000), rng.standard_normal(2000), horizons=(1,))
    assert large.ic_stderr[0] < small.ic_stderr[0]


# ---------- NaN handling ----------------------------------------------


def test_nan_in_signal_dropped():
    rng = np.random.default_rng(0)
    T = 500
    signal = rng.standard_normal(T)
    returns = 0.1 * signal + 0.05 * rng.standard_normal(T)
    signal[100:120] = np.nan
    r = icdecay(signal, returns, horizons=(1,))
    # Fewer than T-1 clean pairs because of the NaN block.
    assert r.n[0] < T - 1


def test_nan_in_returns_propagates_to_forward_sum():
    rng = np.random.default_rng(0)
    T = 500
    signal = rng.standard_normal(T)
    returns = rng.standard_normal(T)
    returns[100:110] = np.nan
    r = icdecay(signal, returns, horizons=(1, 5))
    # Every t whose 5-day forward-sum window covers the NaN block is
    # dropped → fewer usable pairs at h=5 than h=1.
    assert r.n[1] < r.n[0]


# ---------- noise-floor warning ---------------------------------------


def test_low_ic_high_stderr_warns():
    """Small sample + weak signal → IC below noise floor → warning."""
    rng = np.random.default_rng(0)
    signal = rng.standard_normal(60)
    returns = rng.standard_normal(60)
    with pytest.warns(KuantNumericWarning) as record:
        icdecay(signal, returns, horizons=(1, 5))
    assert any("KW-IC-NOISE-FLOOR" in str(w.message) for w in record)


# ---------- error contract --------------------------------------------


def test_reject_empty_horizons():
    with pytest.raises(KuantValueError):
        icdecay(np.arange(100.0), np.arange(100.0), horizons=())


def test_reject_horizon_too_large():
    """Horizon >= T means no overlap."""
    with pytest.raises(KuantValueError):
        icdecay(np.arange(50.0), np.arange(50.0), horizons=(50,))


def test_reject_length_mismatch():
    with pytest.raises(Exception):
        icdecay(np.arange(100.0), np.arange(80.0), horizons=(1,))


def test_reject_negative_horizon():
    with pytest.raises(KuantValueError):
        icdecay(np.arange(100.0), np.arange(100.0), horizons=(-1,))


def test_reject_2d_signal():
    with pytest.raises(Exception):
        icdecay(np.zeros((100, 3)), np.arange(100.0), horizons=(1,))
