"""Tests for kuant.signals v0.6.0 batch 7: signal-processing transforms."""

from __future__ import annotations

import numpy as np
import pytest

from kuant.errors import KuantShapeError, KuantValueError
from kuant.signals import emd, ica, kernelpca, wavelet, whitening


def _correlated_matrix(n=300, d=4, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(d, d))
    return rng.normal(size=(n, d)) @ A


# ---------- whitening -------------------------------------------------


class TestWhitening:
    def test_zca_produces_identity_cov(self):
        X = _correlated_matrix()
        r = whitening(X, method="zca")
        C = r.X_white.T @ r.X_white / X.shape[0]
        assert np.allclose(C, np.eye(X.shape[1]), atol=1e-6)

    def test_pca_produces_identity_cov(self):
        X = _correlated_matrix()
        r = whitening(X, method="pca")
        C = r.X_white.T @ r.X_white / X.shape[0]
        assert np.allclose(C, np.eye(X.shape[1]), atol=1e-6)

    def test_bad_method_rejected(self):
        with pytest.raises(KuantValueError):
            whitening(_correlated_matrix(), method="bogus")

    def test_bad_shape_rejected(self):
        with pytest.raises(KuantShapeError):
            whitening(np.zeros(50))


# ---------- ica -------------------------------------------------------


try:
    import sklearn  # noqa: F401 - optional dependency for ICA

    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


@pytest.mark.skipif(not _HAS_SKLEARN, reason="sklearn not installed")
class TestIca:
    def test_returns_result(self):
        X = _correlated_matrix()
        r = ica(X, n_components=3, max_iter=200)
        assert r.sources.shape == (X.shape[0], 3)
        assert r.mixing.shape == (X.shape[1], 3)

    def test_sources_independent(self):
        # Recovered sources should have low pairwise correlation.
        X = _correlated_matrix(n=500, d=3, seed=1)
        r = ica(X, n_components=3)
        C = np.corrcoef(r.sources.T)
        off = C - np.diag(np.diag(C))
        assert np.max(np.abs(off)) < 0.15

    def test_bad_n_components_rejected(self):
        with pytest.raises(KuantValueError):
            ica(_correlated_matrix(), n_components=99)


# ---------- kernelpca -------------------------------------------------


class TestKernelPca:
    def test_returns_components(self):
        X = _correlated_matrix()
        r = kernelpca(X, n_components=2, kernel="rbf")
        assert r.components.shape == (X.shape[0], 2)

    def test_linear_kernel_matches_pca_variance_ordering(self):
        X = _correlated_matrix(seed=2)
        r = kernelpca(X, n_components=2, kernel="linear")
        # First component should carry more variance than second.
        assert np.var(r.components[:, 0]) >= np.var(r.components[:, 1])

    def test_bad_kernel_rejected(self):
        with pytest.raises(KuantValueError):
            kernelpca(_correlated_matrix(), n_components=2, kernel="bogus")


# ---------- wavelet ---------------------------------------------------


class TestWavelet:
    def test_haar_reconstructs_scale_count(self):
        rng = np.random.default_rng(0)
        r = wavelet(rng.normal(size=512), n_scales=5, kernel="haar")
        assert r.n_scales == 5
        assert len(r.details) == 5

    def test_variance_monotone_for_smooth_signal(self):
        # A smooth low-frequency signal should have lower-scale details
        # with smaller variance than higher-scale approximation.
        t = np.linspace(0, 4 * np.pi, 1024)
        sig = np.sin(t) + 0.05 * np.random.default_rng(0).normal(size=t.size)
        r = wavelet(sig, n_scales=6, kernel="haar")
        # Higher scales capture the sinusoid = larger variance.
        assert r.variances[-1] > r.variances[0]

    def test_bad_kernel_rejected(self):
        rng = np.random.default_rng(1)
        with pytest.raises(KuantValueError):
            wavelet(rng.normal(size=256), kernel="bogus")

    def test_too_short_rejected(self):
        with pytest.raises(KuantValueError):
            wavelet(np.arange(16.0))


# ---------- emd -------------------------------------------------------


class TestEmd:
    def test_returns_imfs(self):
        rng = np.random.default_rng(0)
        sig = (
            np.sin(np.linspace(0, 8 * np.pi, 512))
            + 0.3 * np.sin(np.linspace(0, 32 * np.pi, 512))
            + 0.1 * rng.normal(size=512)
        )
        r = emd(sig, max_imfs=5, sifting_iters=8)
        assert r.n_imfs >= 1
        # Sum of IMFs + residual should reconstruct signal.
        recon = sum(r.imfs) + r.residual
        assert np.allclose(recon, sig, atol=1e-6)

    def test_too_short_rejected(self):
        with pytest.raises(KuantValueError):
            emd(np.arange(30.0))

    def test_bad_max_imfs_rejected(self):
        rng = np.random.default_rng(1)
        with pytest.raises(KuantValueError):
            emd(rng.normal(size=200), max_imfs=0)
