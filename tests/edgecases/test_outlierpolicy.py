"""Tests for kuant.edgecases.outlierpolicy."""

from __future__ import annotations

import numpy as np
import pytest

from kuant.edgecases.outlierpolicy import outlierpolicy
from kuant.errors import KuantNumericWarning, KuantValueError


# ---------- basic detection ----------------------------------------------


def test_mad_flags_lone_outlier():
    """One 100x outlier in an otherwise tight sample."""
    x = np.array([1.0, 2, 3, 4, 5, 100])
    mask = outlierpolicy(x, method="mad")
    assert mask[-1] and not mask[:-1].any()


def test_iqr_flags_lone_outlier():
    """Same setup with the IQR detector."""
    x = np.array([1.0, 2, 3, 4, 5, 100])
    mask = outlierpolicy(x, method="iqr")
    assert mask[-1] and not mask[:-1].any()


def test_zscore_flags_lone_outlier_in_large_clean_sample():
    """zscore CAN detect an outlier — as long as the clean sample is
    large enough that the outlier can't hide by inflating std. A tiny
    sample where the outlier dominates mean+std is the canonical
    zscore-blindspot case (see test_mad_more_robust_than_zscore)."""
    rng = np.random.default_rng(0)
    clean = rng.standard_normal(200)
    x = np.concatenate([clean, [10.0]])  # 10σ outlier on N(0, 1)
    mask = outlierpolicy(x, method="zscore", threshold=3.0)
    assert mask[-1]


def test_zscore_can_miss_outlier_in_small_sample():
    """The canonical zscore blindspot: 100 pulls both mean AND std
    upward, so |100 - mean| / std stays below the threshold.
    Documented failure mode — prefer MAD on small samples."""
    x = np.array([1.0, 2, 3, 4, 5, 100])
    mask_z = outlierpolicy(x, method="zscore", threshold=3.0)
    mask_m = outlierpolicy(x, method="mad", threshold=3.0)
    # MAD catches it; zscore misses.
    assert mask_m[-1]
    assert not mask_z[-1]


def test_returns_bool_mask_of_correct_length():
    x = np.arange(10.0)
    mask = outlierpolicy(x)
    assert mask.shape == (10,)
    assert mask.dtype == bool


# ---------- comparative behavior ----------------------------------------


def test_mad_more_robust_than_zscore():
    """With two moderately-extreme values, zscore's mean+std pull toward
    the outliers and hide them; MAD keeps flagging them."""
    x = np.array([1.0, 2, 3, 4, 5, 6, 40, 50])
    zscore_mask = outlierpolicy(x, method="zscore", threshold=3.0)
    mad_mask = outlierpolicy(x, method="mad", threshold=3.0)
    # MAD should flag both outliers; zscore may miss them.
    assert mad_mask[-2:].sum() == 2
    # MAD identifies at least as many outliers as zscore on skewed data.
    assert mad_mask.sum() >= zscore_mask.sum()


def test_symmetric_distribution_no_false_positives():
    """A standard-normal sample of 1000 shouldn't have many false positives."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(1000)
    mask = outlierpolicy(x, method="mad", threshold=3.0)
    # Rough sanity: fewer than 2% flagged as outliers under Gaussian.
    assert mask.mean() < 0.02


# ---------- NaN handling -------------------------------------------------


def test_nan_positions_stay_false_in_mask():
    """NaN is neither inlier nor outlier."""
    x = np.array([1.0, np.nan, 100, 2, 3])
    mask = outlierpolicy(x, method="mad")
    assert not mask[1]  # NaN position → False


def test_all_nan_returns_all_false():
    x = np.array([np.nan, np.nan, np.nan])
    mask = outlierpolicy(x, method="mad")
    assert not mask.any()


# ---------- degenerate scale warnings ------------------------------------


def test_constant_input_mad_warns_and_returns_all_false():
    x = np.ones(10)
    with pytest.warns(KuantNumericWarning) as record:
        mask = outlierpolicy(x, method="mad")
    assert any("KW-OUTLIER-DEGENERATE" in str(w.message) for w in record)
    assert not mask.any()


def test_constant_input_iqr_warns():
    x = np.ones(10)
    with pytest.warns(KuantNumericWarning):
        outlierpolicy(x, method="iqr")


def test_constant_input_zscore_warns():
    x = np.ones(10)
    with pytest.warns(KuantNumericWarning):
        outlierpolicy(x, method="zscore")


# ---------- error contract -----------------------------------------------


def test_reject_bad_method():
    with pytest.raises(KuantValueError) as exc:
        outlierpolicy(np.arange(5.0), method="chebyshev")
    m = str(exc.value)
    assert "method" in m and "mad" in m


def test_reject_negative_threshold():
    with pytest.raises(KuantValueError):
        outlierpolicy(np.arange(5.0), method="mad", threshold=-1)


def test_reject_zero_threshold():
    with pytest.raises(KuantValueError):
        outlierpolicy(np.arange(5.0), method="mad", threshold=0)


def test_reject_2d_input():
    with pytest.raises(Exception):
        outlierpolicy(np.zeros((5, 3)), method="mad")


# ---------- defaults + tuning --------------------------------------------


def test_method_defaults_populate_correctly():
    """Passing threshold=None triggers per-method defaults."""
    x = np.array([1.0, 2, 3, 4, 5, 100])
    # Just verify each method is callable with the default.
    for method in ("mad", "iqr", "zscore"):
        mask = outlierpolicy(x, method=method)
        assert mask.dtype == bool


def test_tighter_threshold_flags_more():
    """A tighter threshold should flag at least as many outliers."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(500)
    x[0] = 5.0  # obvious outlier
    tight = outlierpolicy(x, method="mad", threshold=1.0)
    loose = outlierpolicy(x, method="mad", threshold=4.0)
    assert tight.sum() >= loose.sum()
