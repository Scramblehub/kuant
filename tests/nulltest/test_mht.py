"""Tests for kuant.nulltest.mht_correction."""

from __future__ import annotations

import numpy as np
import pytest

from kuant.errors import KuantValueError
from kuant.nulltest.mht_correction import mht_correction


def test_bonferroni_multiplies_by_n():
    p = np.array([0.01, 0.03, 0.10])
    adj = mht_correction(p, method="bonferroni")
    assert np.allclose(adj, [0.03, 0.09, 0.30])


def test_bonferroni_caps_at_1():
    p = np.array([0.5, 0.9])
    adj = mht_correction(p, method="bonferroni")
    assert (adj <= 1.0).all()


def test_holm_monotone():
    p = np.array([0.001, 0.01, 0.03, 0.05, 0.20])
    adj = mht_correction(p, method="holm")
    # Sorted by original p; adjusted values must be non-decreasing
    # when re-sorted.
    order = np.argsort(p)
    assert (np.diff(adj[order]) >= -1e-9).all()


def test_bh_monotone_after_sort():
    p = np.array([0.001, 0.01, 0.03, 0.05, 0.20])
    adj = mht_correction(p, method="bh")
    order = np.argsort(p)
    assert (np.diff(adj[order]) >= -1e-9).all()


def test_bh_less_conservative_than_bonferroni():
    """BH controls FDR (less strict) → adjustments no larger than Bonferroni."""
    p = np.array([0.01, 0.02, 0.04, 0.10, 0.20])
    bh = mht_correction(p, method="bh")
    bon = mht_correction(p, method="bonferroni")
    assert (bh <= bon + 1e-9).all()


def test_scalar_input_returns_float():
    """Scalar in → scalar out."""
    out = mht_correction(0.02, method="bonferroni")
    assert isinstance(out, float)


def test_reject_out_of_range_p():
    with pytest.raises(KuantValueError):
        mht_correction(np.array([0.5, -0.1]))


def test_reject_bad_method():
    with pytest.raises(KuantValueError):
        mht_correction(np.array([0.01, 0.02]), method="bogus")


def test_single_p_no_correction_for_bh():
    """N=1 → BH adjustment = raw p * 1 / 1 = raw p."""
    p = np.array([0.03])
    adj = mht_correction(p, method="bh")
    assert adj[0] == pytest.approx(0.03)
