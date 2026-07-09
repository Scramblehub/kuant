"""Tests for kuant.sindy.chaos.entropy (5 kernels)."""

from __future__ import annotations

import numpy as np
import pytest

from kuant.errors import KuantValueError
from kuant.sindy.chaos.entropy import (
    ApproximateEntropyResult,
    DispersionEntropyResult,
    PermutationEntropyResult,
    SampleEntropyResult,
    TransferEntropyResult,
    approximateentropy,
    dispersionentropy,
    permutationentropy,
    sampleentropy,
    transferentropy,
)


# ---------- permutationentropy ----------------------------------------


class TestPermutationEntropy:
    def test_returns_result(self):
        rng = np.random.default_rng(0)
        r = permutationentropy(rng.normal(size=500))
        assert isinstance(r, PermutationEntropyResult)

    def test_monotone_sequence_zero_entropy(self):
        # A strictly monotone sequence has only one ordinal pattern.
        x = np.arange(500.0)
        r = permutationentropy(x, m=3)
        assert r.entropy < 1e-10
        assert r.n_patterns_seen == 1

    def test_gaussian_near_max_normalized(self):
        rng = np.random.default_rng(1)
        r = permutationentropy(rng.normal(size=2000), m=3)
        # Gaussian is nearly maximally disordered in ordinal patterns.
        assert r.normalized > 0.95

    def test_too_short_rejected(self):
        with pytest.raises(KuantValueError, match=r"finite values"):
            permutationentropy(np.array([1.0, 2.0, 3.0]), m=3, tau=1)

    def test_bad_m_rejected(self):
        rng = np.random.default_rng(2)
        with pytest.raises(KuantValueError):
            permutationentropy(rng.normal(size=500), m=1)

    def test_summary(self):
        rng = np.random.default_rng(3)
        r = permutationentropy(rng.normal(size=500))
        s = r.summary()
        assert "PermutationEntropyResult" in s


# ---------- sampleentropy ---------------------------------------------


class TestSampleEntropy:
    def test_returns_result(self):
        rng = np.random.default_rng(0)
        r = sampleentropy(rng.normal(size=200))
        assert isinstance(r, SampleEntropyResult)

    def test_positive_for_noise(self):
        rng = np.random.default_rng(1)
        r = sampleentropy(rng.normal(size=500))
        assert r.entropy > 0

    def test_smaller_for_more_regular(self):
        # Sinusoid (highly regular) should have LOWER SampEn than noise.
        rng = np.random.default_rng(2)
        noise = rng.normal(size=500)
        sinusoid = np.sin(2 * np.pi * np.arange(500) / 20.0)
        # Use r = 0.2 * std over the SAME reference so magnitudes are
        # comparable.
        r_noise = sampleentropy(noise, m=2)
        r_sin = sampleentropy(sinusoid, m=2)
        assert r_sin.entropy < r_noise.entropy

    def test_too_short_rejected(self):
        with pytest.raises(KuantValueError):
            sampleentropy(np.arange(20.0))

    def test_bad_r_rejected(self):
        rng = np.random.default_rng(3)
        with pytest.raises(KuantValueError):
            sampleentropy(rng.normal(size=200), r=-1.0)

    def test_summary(self):
        rng = np.random.default_rng(4)
        r = sampleentropy(rng.normal(size=200))
        assert "SampleEntropyResult" in r.summary()


# ---------- approximateentropy ----------------------------------------


class TestApproximateEntropy:
    def test_returns_result(self):
        rng = np.random.default_rng(0)
        r = approximateentropy(rng.normal(size=200))
        assert isinstance(r, ApproximateEntropyResult)

    def test_positive_for_noise(self):
        rng = np.random.default_rng(1)
        r = approximateentropy(rng.normal(size=500))
        assert r.entropy > 0

    def test_too_short_rejected(self):
        with pytest.raises(KuantValueError):
            approximateentropy(np.arange(20.0))


# ---------- dispersionentropy -----------------------------------------


class TestDispersionEntropy:
    def test_returns_result(self):
        rng = np.random.default_rng(0)
        r = dispersionentropy(rng.normal(size=500))
        assert isinstance(r, DispersionEntropyResult)

    def test_normalized_bounded(self):
        rng = np.random.default_rng(1)
        r = dispersionentropy(rng.normal(size=1000))
        assert 0.0 <= r.normalized <= 1.0

    def test_gaussian_near_max_normalized(self):
        rng = np.random.default_rng(2)
        r = dispersionentropy(rng.normal(size=2000), m=3, c=6)
        # Gaussian noise has broad dispersion-pattern coverage.
        assert r.normalized > 0.85

    def test_bad_c_rejected(self):
        rng = np.random.default_rng(3)
        with pytest.raises(KuantValueError):
            dispersionentropy(rng.normal(size=500), c=1)


# ---------- transferentropy -------------------------------------------


class TestTransferEntropy:
    def test_returns_result(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=500)
        y = rng.normal(size=500)
        r = transferentropy(x, y)
        assert isinstance(r, TransferEntropyResult)

    def test_independent_series_small_te(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=500)
        y = rng.normal(size=500)
        r = transferentropy(x, y, bins=6)
        # With histogram bias, independent series should still show
        # SMALL TE; guard against runaway.
        assert r.te < 0.5

    def test_coupled_series_larger_te(self):
        # y_{t+1} = 0.7 * x_t + 0.3 * noise. X drives Y.
        rng = np.random.default_rng(2)
        n = 800
        x = rng.normal(size=n)
        y = np.zeros(n)
        y[0] = rng.normal()
        for i in range(1, n):
            y[i] = 0.7 * x[i - 1] + 0.3 * rng.normal()
        r_xy = transferentropy(x, y, lag=1, bins=6)
        r_yx = transferentropy(y, x, lag=1, bins=6)
        # TE(X -> Y) should exceed TE(Y -> X) for a directed link.
        assert r_xy.te > r_yx.te

    def test_unequal_length_rejected(self):
        rng = np.random.default_rng(3)
        with pytest.raises(KuantValueError, match=r"equal length"):
            transferentropy(rng.normal(size=500), rng.normal(size=400))

    def test_too_short_rejected(self):
        rng = np.random.default_rng(4)
        with pytest.raises(KuantValueError):
            transferentropy(rng.normal(size=50), rng.normal(size=50))
