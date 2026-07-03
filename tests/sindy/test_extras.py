"""Tests for pinnscan, symbolicscan, accelerationscan."""

from __future__ import annotations

import numpy as np
import pytest

from kuant.sindy import accelerationscan, pinnscan, symbolicscan


# ---------------------------------------------------------------------------
# pinnscan
# ---------------------------------------------------------------------------


def test_pinnscan_signal_beats_shuffle():
    """Nonlinear signal: y = 0.5·x1·x2 + noise → OOF R² > 0, p < 0.05."""
    rng = np.random.default_rng(0)
    n = 400
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 0.5 * x1 * x2 + rng.normal(scale=0.3, size=n)
    library = {"x1": x1, "x2": x2, "noise": rng.normal(size=n)}
    result = pinnscan(y, library, n_perms=50)
    assert result.r2_oof > 0.1
    assert result.permutation_p < 0.05
    assert "x1" in result.feature_importances
    assert "x2" in result.feature_importances


def test_pinnscan_null_signal_high_p():
    """y independent of features → high permutation p."""
    rng = np.random.default_rng(0)
    n = 300
    y = rng.normal(size=n)
    library = {f"x{i}": rng.normal(size=n) for i in range(4)}
    result = pinnscan(y, library, n_perms=50)
    # No genuine signal; p should NOT be < 0.05 (allow some slack for noise)
    assert result.permutation_p > 0.05 or abs(result.corr_oof) < 0.15


def test_pinnscan_summary_readable():
    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(size=n)
    y = x + rng.normal(scale=0.3, size=n)
    text = pinnscan(y, {"x": x}, n_perms=30).summary()
    assert "PINN" in text or "feature-library" in text
    assert "permutation" in text.lower()


# ---------------------------------------------------------------------------
# symbolicscan
# ---------------------------------------------------------------------------


def test_symbolicscan_recovers_pure_interaction():
    """y = 0.5·x1·x2 + noise → LASSO should pick up x1·x2 term."""
    rng = np.random.default_rng(0)
    n = 500
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 0.5 * x1 * x2 + rng.normal(scale=0.2, size=n)
    result = symbolicscan(y, {"x1": x1, "x2": x2}, degree=2)
    # 'x1 x2' is sklearn's PolynomialFeatures naming for interactions
    interaction_terms = [k for k in result.selected_terms if "x1" in k and "x2" in k]
    assert len(interaction_terms) > 0


def test_symbolicscan_null_selects_nothing():
    """Pure noise → no polynomial terms selected."""
    rng = np.random.default_rng(0)
    n = 400
    y = rng.normal(size=n)
    result = symbolicscan(y, {"x1": rng.normal(size=n), "x2": rng.normal(size=n)}, degree=2)
    assert len(result.selected_terms) == 0 or all(
        abs(c) < 0.05 for c in result.selected_terms.values()
    )


def test_symbolicscan_degree_1_matches_linear():
    """degree=1 should give a linear-only expansion."""
    rng = np.random.default_rng(0)
    n = 300
    x = rng.normal(size=n)
    y = 0.5 * x + rng.normal(scale=0.3, size=n)
    result = symbolicscan(y, {"x": x}, degree=1)
    # Only 'x' should appear (no x²)
    assert all("^2" not in k and "x^2" not in k for k in result.selected_terms)


def test_symbolicscan_bad_degree_raises():
    with pytest.raises(ValueError, match="'degree' must be"):
        symbolicscan(np.arange(50.0), {"x": np.arange(50.0)}, degree=0)


def test_symbolicscan_summary_readable():
    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(size=n)
    y = x + rng.normal(scale=0.3, size=n)
    text = symbolicscan(y, {"x": x}, degree=2).summary()
    assert "polynomial" in text.lower() or "symbolic" in text.lower()


# ---------------------------------------------------------------------------
# accelerationscan
# ---------------------------------------------------------------------------


def test_accelerationscan_detects_true_signal():
    """Construct y = 0.5·d²x + noise → smoothing=1 should have high |corr|."""
    rng = np.random.default_rng(0)
    n = 500
    x = np.cumsum(rng.normal(size=n))
    d2 = np.zeros(n)
    d2[2:] = x[2:] - 2 * x[1:-1] + x[:-2]
    y = 0.5 * d2 + rng.normal(scale=0.3, size=n)
    result = accelerationscan(x, y, smoothings=[1, 5])
    assert abs(result.correlations[1]) > result.noise_floor


def test_accelerationscan_null_signal():
    """x and y independent → correlations should sit near zero.

    Sample correlation std scales as 1/√n, so we need a big enough n for
    the null to actually be below the default noise floor of 0.025.
    """
    rng = np.random.default_rng(0)
    n = 5000
    x = np.cumsum(rng.normal(size=n))
    y = rng.normal(size=n)
    result = accelerationscan(x, y, smoothings=[1, 5, 21])
    # At n=5000, sample corr std ≈ 0.014. All 3 should be under the 0.025 floor.
    assert all(abs(c) < result.noise_floor for c in result.correlations.values())


def test_accelerationscan_length_mismatch_raises():
    with pytest.raises(ValueError, match="equal length"):
        accelerationscan(np.arange(100.0), np.arange(50.0))


def test_accelerationscan_negative_smoothing_raises():
    with pytest.raises(ValueError, match="'smoothing' must be"):
        accelerationscan(np.arange(50.0), np.arange(50.0), smoothings=[1, 0])


def test_accelerationscan_summary_readable():
    rng = np.random.default_rng(0)
    n = 200
    x = np.cumsum(rng.normal(size=n))
    y = rng.normal(size=n)
    text = accelerationscan(x, y).summary()
    assert "Acceleration" in text
