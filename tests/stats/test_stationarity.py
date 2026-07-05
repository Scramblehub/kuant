"""Tests for kuant.stats.stationarity."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("statsmodels")

from kuant.errors import KuantValueError  # noqa: E402
from kuant.stats.stationarity import (  # noqa: E402
    StationarityResult,
    adftest,
    kpsstest,
)

_HAS_ARCH = False
try:
    import arch  # noqa: F401

    _HAS_ARCH = True
except ImportError:
    pass


# ---------- ADF -------------------------------------------------------


def test_adf_rejects_null_on_iid_noise():
    rng = np.random.default_rng(0)
    iid = rng.standard_normal(500)
    r = adftest(iid)
    assert r.is_stationary
    assert r.p_value < 0.05


def test_adf_fails_to_reject_on_random_walk():
    rng = np.random.default_rng(0)
    rw = np.cumsum(rng.standard_normal(500))
    r = adftest(rw)
    assert not r.is_stationary
    assert r.p_value > 0.10


def test_adf_result_type_and_metadata():
    rng = np.random.default_rng(0)
    r = adftest(rng.standard_normal(200))
    assert isinstance(r, StationarityResult)
    assert r.test == "adftest"
    assert "unit root" in r.null_hypothesis.lower()


def test_adf_rejects_too_short():
    with pytest.raises(KuantValueError):
        adftest(np.arange(10.0))


# ---------- KPSS ------------------------------------------------------


def test_kpss_on_iid_noise_stationary():
    rng = np.random.default_rng(0)
    iid = rng.standard_normal(500)
    r = kpsstest(iid)
    # IID → truly stationary; KPSS should NOT reject the null.
    assert r.is_stationary


def test_kpss_on_random_walk_rejects_stationarity():
    rng = np.random.default_rng(0)
    rw = np.cumsum(rng.standard_normal(500))
    r = kpsstest(rw)
    # Random walk → NOT stationary. KPSS should reject its null.
    assert not r.is_stationary


def test_kpss_metadata():
    rng = np.random.default_rng(0)
    r = kpsstest(rng.standard_normal(200))
    assert r.test == "kpsstest"
    assert "stationary" in r.null_hypothesis.lower()


# ---------- adf and kpss agree on the easy cases ---------------------


def test_adf_and_kpss_agree_on_stationary_series():
    rng = np.random.default_rng(0)
    iid = rng.standard_normal(500)
    assert adftest(iid).is_stationary
    assert kpsstest(iid).is_stationary


def test_adf_and_kpss_agree_on_random_walk():
    rng = np.random.default_rng(0)
    rw = np.cumsum(rng.standard_normal(500))
    assert not adftest(rw).is_stationary
    assert not kpsstest(rw).is_stationary


# ---------- Phillips-Perron and variance ratio (arch dep) ------------


@pytest.mark.skipif(not _HAS_ARCH, reason="arch package not installed")
def test_phillipsperron_on_random_walk():
    from kuant.stats.stationarity import phillipsperrontest

    rng = np.random.default_rng(0)
    rw = np.cumsum(rng.standard_normal(500))
    r = phillipsperrontest(rw)
    assert not r.is_stationary


@pytest.mark.skipif(not _HAS_ARCH, reason="arch package not installed")
def test_variance_ratio_on_random_walk_null_holds():
    from kuant.stats.stationarity import varianceratiotest

    rng = np.random.default_rng(0)
    rw = np.cumsum(rng.standard_normal(500))
    r = varianceratiotest(rw, lags=2)
    # RW is exactly the null; VR should NOT reject at 5%.
    assert not r.is_stationary
