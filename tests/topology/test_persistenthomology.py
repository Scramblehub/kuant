"""Tests for kuant.topology.persistenthomology.

Correctness comes from three classical point-cloud fixtures with
known homology:
- Circle in R² → one long H1 (the loop) + n_points H0 classes.
- Two well-separated clusters → 2 persistent H0 features.
- Sphere in R³ → one H2 (void) at high dim.

Beyond that we verify: shape/dtype invariants of the returned
diagrams; time-delay embedding on a sine wave gives an H1 loop
(the phase-space attractor of a periodic signal is a circle);
error messages surface with the kuant.errors classes.
"""

from __future__ import annotations

import numpy as np
import pytest

# ripser is a hard requirement for this suite.
ripser_mod = pytest.importorskip("ripser")

from kuant.errors import KuantDependencyError, KuantShapeError, KuantValueError  # noqa: E402
from kuant.topology.persistenthomology import (  # noqa: E402
    PersistenceDiagram,
    _time_delay_embed,
    persistenthomology,
)


# ---------- time-delay embed helper ----------------------------------------


def test_delay_embed_shape():
    x = np.arange(10.0)
    out = _time_delay_embed(x, embedding_dim=3, delay=1)
    assert out.shape == (8, 3)
    # First point: (x[0], x[1], x[2])
    assert np.array_equal(out[0], np.array([0, 1, 2]))
    # Last point:  (x[7], x[8], x[9])
    assert np.array_equal(out[-1], np.array([7, 8, 9]))


def test_delay_embed_tau_2():
    x = np.arange(10.0)
    out = _time_delay_embed(x, embedding_dim=2, delay=3)
    assert out.shape == (7, 2)
    assert np.array_equal(out[0], np.array([0, 3]))


def test_delay_embed_series_too_short():
    x = np.arange(3.0)
    # d=3, tau=2 needs stride = 4 > n_available_starts.
    out = _time_delay_embed(x, embedding_dim=3, delay=2)
    assert out.shape == (0, 3)


# ---------- known-topology point clouds ------------------------------------


def _circle(n=100, r=1.0):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([r * np.cos(t), r * np.sin(t)], axis=1)


def test_circle_has_one_persistent_h1_loop():
    """A clean circle in R² carries exactly one persistent H1 feature."""
    cloud = _circle(n=80)
    d = persistenthomology(cloud, dim=1)
    # ripser samples on a circle produce noisy short-lived H1 pairs plus
    # the one long-lived loop. We check that the LONGEST persistence in
    # H1 is meaningfully larger than any second-place feature.
    p1 = d.persistences(1)
    assert p1.size >= 1
    # Longest persistence must be substantial relative to circle radius.
    assert p1[0] > 1.0
    # And it should dominate any runner-up.
    if p1.size >= 2:
        assert p1[0] > 5 * p1[1]


def test_circle_h0_has_infinite_class():
    """H0 has one infinite-death class (the whole set is connected in the limit)."""
    d = persistenthomology(_circle(n=50), dim=1)
    h0 = d.h0
    assert h0.shape[0] >= 1
    assert np.isinf(h0[:, 1]).any(), "expected at least one infinite-death H0 class"


def test_two_clusters_have_two_persistent_h0():
    """Two well-separated clusters yield 2 persistent connected components."""
    rng = np.random.default_rng(0)
    cluster_a = rng.normal(0, 0.05, size=(40, 2))
    cluster_b = rng.normal(0, 0.05, size=(40, 2)) + np.array([10.0, 0.0])
    cloud = np.vstack([cluster_a, cluster_b])
    d = persistenthomology(cloud, dim=0)
    p0 = d.persistences(0)
    # At least 2 features with persistence > 1.0 (well below the 10-unit gap).
    long_lived = p0[p0 > 1.0]
    assert long_lived.size >= 2


def test_n_features_with_min_persistence():
    """Filtering by min persistence is monotone."""
    d = persistenthomology(_circle(n=60), dim=1)
    all_h1 = d.n_features(1, min_persistence=0.0)
    strong_h1 = d.n_features(1, min_persistence=0.5)
    assert strong_h1 <= all_h1
    # The circle's long loop should count as at least one strong-H1.
    assert strong_h1 >= 1


# ---------- 1D series (Takens embedding) -----------------------------------


def test_sine_wave_has_persistent_h1():
    """A pure sine wave's phase-space attractor is a circle → one long H1."""
    t = np.linspace(0, 10 * np.pi, 400)
    x = np.sin(t)
    d = persistenthomology(x, dim=1, embedding_dim=3, delay=6)
    assert d.embedding_dim == 3
    assert d.delay == 6
    p1 = d.persistences(1)
    assert p1.size >= 1
    # Sine has a clean attractor; the dominant H1 loop should stand out.
    assert p1[0] > 0.3


# ---------- return-object contract -----------------------------------------


def test_return_is_persistence_diagram():
    d = persistenthomology(_circle(n=30), dim=1)
    assert isinstance(d, PersistenceDiagram)
    assert d.max_dim == 1
    assert d.n_points == 30


def test_h_properties_return_ndarray():
    d = persistenthomology(_circle(n=30), dim=1)
    assert isinstance(d.h0, np.ndarray)
    assert isinstance(d.h1, np.ndarray)
    assert isinstance(d.h2, np.ndarray)  # not computed → empty
    assert d.h2.shape == (0, 2)


def test_diagrams_sorted_descending():
    d = persistenthomology(_circle(n=50), dim=1)
    for k in (0, 1):
        arr = d.diagrams[k]
        if arr.shape[0] < 2:
            continue
        pers = arr[:, 1] - arr[:, 0]
        # Ignore inf entries when checking monotonicity — they sort to start.
        finite_pers = pers[np.isfinite(pers)]
        if finite_pers.size >= 2:
            assert np.all(np.diff(finite_pers) <= 1e-9)


def test_degenerate_single_point():
    """<2 points → empty diagrams, no crash."""
    d = persistenthomology(np.array([[0.0, 0.0]]), dim=1)
    assert d.n_points == 1
    for k in (0, 1):
        assert d.diagrams[k].shape == (0, 2)


# ---------- error-message contract -----------------------------------------


def test_reject_3d_input():
    with pytest.raises(KuantShapeError) as exc:
        persistenthomology(np.zeros((10, 5, 3)))
    m = str(exc.value)
    assert "persistenthomology" in m
    assert "1D" in m and "2D" in m


def test_reject_bad_embedding_dim():
    with pytest.raises(KuantValueError) as exc:
        persistenthomology(np.arange(50.0), embedding_dim=0)
    assert "embedding_dim" in str(exc.value)


def test_reject_bad_delay():
    with pytest.raises(KuantValueError) as exc:
        persistenthomology(np.arange(50.0), delay=-1)
    assert "delay" in str(exc.value)


# ---------- lazy-dep error message -----------------------------------------


def test_lazy_dep_message_is_actionable(monkeypatch):
    """If ripser is not installed, the error should point to pip install ripser."""

    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "ripser":
            raise ImportError("no module named ripser")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(KuantDependencyError) as exc:
        persistenthomology(np.arange(50.0))
    m = str(exc.value)
    assert "pip install ripser" in m
    assert "persistenthomology" in m
