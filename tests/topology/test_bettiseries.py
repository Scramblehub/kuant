"""Tests for kuant.topology.bettiseries."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("ripser")

from kuant.errors import KuantShapeError, KuantValueError  # noqa: E402
from kuant.topology.bettiseries import bettiseries  # noqa: E402


def test_warmup_is_nan():
    x = np.sin(np.linspace(0, 10 * np.pi, 300))
    b = bettiseries(x, window=100, dim=1)
    assert np.isnan(b[:99]).all()
    # First filled anchor.
    assert not np.isnan(b[99])


def test_length_matches_input():
    x = np.sin(np.linspace(0, 4 * np.pi, 200))
    b = bettiseries(x, window=50, dim=1)
    assert b.size == x.size


def test_sine_wave_produces_h1_signal():
    """A pure sine wave inside every window yields a persistent H1 loop."""
    t = np.linspace(0, 20 * np.pi, 600)
    x = np.sin(t)
    b = bettiseries(x, window=120, dim=1, embedding_dim=3, delay=6, min_persistence=0.2)
    filled = b[~np.isnan(b)]
    assert filled.size > 0
    # Most windows should surface at least one persistent loop.
    assert (filled >= 1).mean() > 0.5


def test_stride_skips_anchors():
    """Stride > 1 fills only every k-th slot; others stay NaN."""
    x = np.sin(np.linspace(0, 4 * np.pi, 200))
    b = bettiseries(x, window=50, dim=1, stride=10)
    # Anchor indices: 49, 59, 69, ..., in the post-warmup region.
    filled_idx = np.flatnonzero(~np.isnan(b))
    if filled_idx.size >= 2:
        gaps = np.diff(filled_idx)
        assert (gaps == 10).all()


def test_min_persistence_filters_features():
    """Raising min_persistence never increases the count at any anchor."""
    x = np.sin(np.linspace(0, 10 * np.pi, 400))
    b_all = bettiseries(x, window=100, dim=1, min_persistence=0.0)
    b_strong = bettiseries(x, window=100, dim=1, min_persistence=0.4)
    mask = ~np.isnan(b_all) & ~np.isnan(b_strong)
    assert (b_strong[mask] <= b_all[mask]).all()


def test_nan_segment_yields_nan_anchor():
    """A NaN inside the trailing window → NaN Betti at that anchor."""
    x = np.linspace(0, 1, 200)
    x[50] = np.nan
    b = bettiseries(x, window=40, dim=0)
    # Any anchor whose window covers index 50 must be NaN.
    for t in range(50, 50 + 40):
        if t < b.size:
            assert np.isnan(b[t]), f"anchor {t} should be NaN (window covers NaN at 50)"


def test_window_larger_than_input_returns_all_nan():
    x = np.arange(20.0)
    b = bettiseries(x, window=100, dim=0)
    assert np.isnan(b).all()
    assert b.size == 20


# ---------- error contract ------------------------------------------------


def test_reject_2d_input():
    with pytest.raises(KuantShapeError):
        bettiseries(np.zeros((100, 3)), window=20)


def test_reject_zero_window():
    with pytest.raises(KuantValueError) as exc:
        bettiseries(np.arange(50.0), window=0)
    assert "window" in str(exc.value)


def test_reject_zero_stride():
    with pytest.raises(KuantValueError) as exc:
        bettiseries(np.arange(50.0), window=10, stride=0)
    assert "stride" in str(exc.value)


def test_reject_negative_min_persistence():
    with pytest.raises(KuantValueError) as exc:
        bettiseries(np.arange(50.0), window=10, min_persistence=-0.1)
    assert "min_persistence" in str(exc.value)
