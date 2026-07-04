"""Tests for kuant.topology.wasserstein."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("persim")
pytest.importorskip("ripser")

from kuant.errors import KuantDependencyError, KuantValueError  # noqa: E402
from kuant.topology.persistenthomology import persistenthomology  # noqa: E402
from kuant.topology.wasserstein import wasserstein  # noqa: E402


# ---------- identity, symmetry, non-negativity -----------------------------


def test_identical_diagrams_zero_distance():
    d = np.array([[0.0, 1.0], [0.1, 0.5]])
    assert wasserstein(d, d) == 0.0


def test_empty_diagrams_zero_distance():
    empty = np.empty((0, 2))
    assert wasserstein(empty, empty) == 0.0


def test_symmetry():
    a = np.array([[0.0, 1.0], [0.1, 0.5]])
    b = np.array([[0.0, 1.2], [0.2, 0.6]])
    d_ab = wasserstein(a, b)
    d_ba = wasserstein(b, a)
    assert abs(d_ab - d_ba) < 1e-9


def test_nonnegative():
    a = np.array([[0.0, 1.0]])
    b = np.array([[0.5, 2.0]])
    assert wasserstein(a, b) >= 0.0


# ---------- metric contract: known cases -----------------------------------


def test_shifted_diagram_scales_with_shift():
    """Diagram displaced by δ has distance monotone in δ."""
    base = np.array([[0.0, 1.0], [0.2, 0.6]])
    d_small = wasserstein(base, base + 0.1)
    d_large = wasserstein(base, base + 0.3)
    assert d_small < d_large


def test_bottleneck_smaller_than_wasserstein_typically():
    """Bottleneck is the max matched cost; Wasserstein sums them."""
    a = np.array([[0.0, 1.0], [0.1, 0.4], [0.2, 0.5]])
    b = np.array([[0.05, 1.1], [0.12, 0.45], [0.25, 0.55]])
    w = wasserstein(a, b, metric="wasserstein")
    btl = wasserstein(a, b, metric="bottleneck")
    # For diffuse small shifts, bottleneck (max) ≤ Wasserstein-2 (aggregate).
    assert btl <= w + 1e-9


def test_sliced_wasserstein_finite_and_nonzero():
    a = np.array([[0.0, 1.0], [0.1, 0.4]])
    b = np.array([[0.0, 1.3], [0.15, 0.5]])
    d = wasserstein(a, b, metric="sliced_wasserstein", n_slices=100)
    assert np.isfinite(d) and d > 0.0


# ---------- inf-death handling ---------------------------------------------


def test_infinite_death_rows_dropped_before_metric():
    """A lone finite-death diff should determine the whole distance."""
    a = np.array([[0.0, 1.0], [0.2, np.inf]])
    b = np.array([[0.0, 1.5], [0.2, np.inf]])
    # Only finite pair remains: (0, 1) vs (0, 1.5), Δ = 0.5 → some positive dist.
    d = wasserstein(a, b, metric="bottleneck")
    assert d > 0.0
    # Bottleneck of (0,1) vs (0,1.5) is ≤ 0.5 (max coordinate diff).
    assert d < 1.0


def test_all_infinite_death_empty_after_filter():
    """Diagrams that are entirely inf-death → empty finite parts → 0 distance."""
    a = np.array([[0.0, np.inf]])
    b = np.array([[0.1, np.inf], [0.5, np.inf]])
    assert wasserstein(a, b) == 0.0


# ---------- integration with persistenthomology ----------------------------


def test_from_persistenthomology_circle_vs_stretched_circle():
    """Slight geometric perturbation → small but positive H1 distance."""
    t = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    a = np.stack([np.cos(t), np.sin(t)], axis=1)
    b = np.stack([np.cos(t), 1.10 * np.sin(t)], axis=1)
    d_a = persistenthomology(a, dim=1)
    d_b = persistenthomology(b, dim=1)
    dist = wasserstein(d_a.h1, d_b.h1)
    assert dist > 0.0


# ---------- error contract ------------------------------------------------


def test_bad_metric_rejected():
    with pytest.raises(KuantValueError) as exc:
        wasserstein(np.zeros((1, 2)), np.zeros((1, 2)), metric="chebyshev")
    m = str(exc.value)
    assert "metric" in m
    assert "wasserstein" in m and "bottleneck" in m


def test_bad_shape_rejected():
    with pytest.raises(KuantValueError) as exc:
        wasserstein(np.zeros((5, 3)), np.zeros((5, 2)))
    assert "diagram_a" in str(exc.value)


def test_zero_n_slices_rejected():
    with pytest.raises(KuantValueError):
        wasserstein(np.zeros((1, 2)), np.zeros((1, 2)), metric="sliced_wasserstein", n_slices=0)


# ---------- lazy-dep error message ----------------------------------------


def test_lazy_dep_message_is_actionable(monkeypatch):
    import builtins

    real = builtins.__import__

    def fake(name, *args, **kwargs):
        if name == "persim":
            raise ImportError("no module named persim")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(KuantDependencyError) as exc:
        wasserstein(np.zeros((1, 2)), np.zeros((1, 2)))
    m = str(exc.value)
    assert "pip install persim" in m
    assert "wasserstein" in m
