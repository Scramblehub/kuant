"""Tests for kuant.signals.neutralize."""

from __future__ import annotations

import numpy as np
import pytest

from kuant.errors import KuantNumericWarning, KuantShapeError, KuantValueError
from kuant.signals.neutralize import NeutralizeResult, neutralize


# ---------- known-truth recovery ----------------------------------------


def test_recovers_true_beta():
    """Synthetic setup: signal = 0.5 * factor + noise. β should be near 0.5."""
    rng = np.random.default_rng(0)
    factor = rng.standard_normal(500)
    signal = 0.5 * factor + 0.1 * rng.standard_normal(500)
    r = neutralize(signal, {"factor": factor})
    assert abs(r.betas["factor"] - 0.5) < 0.05
    assert r.r2 > 0.9


def test_intercept_added_by_default():
    signal = np.arange(100.0)
    factor = np.ones(100)  # constant → collinear with intercept
    # With intercept the constant factor is degenerate but neutralize
    # still runs (returns underdetermined-safe residuals). Skip this
    # pathological case and use a non-constant factor.
    factor = np.arange(100.0)
    r = neutralize(signal, {"factor": factor})
    assert "intercept" in r.betas
    assert "factor" in r.betas


def test_no_intercept_option():
    signal = np.arange(50.0)
    factor = np.arange(50.0)
    r = neutralize(signal, {"factor": factor}, add_intercept=False)
    assert "intercept" not in r.betas
    assert list(r.betas.keys()) == ["factor"]


# ---------- factor input formats ----------------------------------------


def test_dict_form_preserves_names():
    rng = np.random.default_rng(0)
    signal = rng.standard_normal(200)
    r = neutralize(signal, {"size": rng.standard_normal(200), "value": rng.standard_normal(200)})
    assert set(r.betas.keys()) >= {"size", "value"}


def test_2d_array_form_gives_generic_names():
    rng = np.random.default_rng(0)
    signal = rng.standard_normal(200)
    X = rng.standard_normal((200, 3))
    r = neutralize(signal, X)
    assert set(r.betas.keys()) >= {"factor0", "factor1", "factor2"}


def test_list_form_generic_names():
    rng = np.random.default_rng(0)
    signal = rng.standard_normal(200)
    r = neutralize(signal, [rng.standard_normal(200), rng.standard_normal(200)])
    assert set(r.betas.keys()) >= {"factor0", "factor1"}


def test_1d_factor_array_treated_as_single_column():
    rng = np.random.default_rng(0)
    signal = rng.standard_normal(200)
    factor = rng.standard_normal(200)
    r = neutralize(signal, factor)
    assert "factor0" in r.betas


# ---------- return object contract --------------------------------------


def test_returns_dataclass():
    signal = np.arange(50.0)
    factor = np.arange(50.0)
    assert isinstance(neutralize(signal, {"f": factor}), NeutralizeResult)


def test_residuals_have_same_length_as_signal():
    signal = np.arange(50.0)
    factor = np.arange(50.0)
    r = neutralize(signal, {"f": factor})
    assert r.residuals.shape == signal.shape


def test_summary_contains_metadata():
    signal = np.arange(50.0)
    factor = np.arange(50.0)
    s = neutralize(signal, {"f": factor}).summary()
    assert "NeutralizeResult" in s
    assert "R²" in s


# ---------- NaN handling -------------------------------------------------


def test_nan_rows_dropped_from_fit_and_output():
    signal = np.arange(100.0)
    factor = np.arange(100.0)
    signal[50] = np.nan
    factor[80] = np.nan
    r = neutralize(signal, {"f": factor})
    assert r.n_used == 98
    assert np.isnan(r.residuals[50])
    assert np.isnan(r.residuals[80])
    # Rows without NaN should have residuals filled.
    assert np.isfinite(r.residuals[0])


def test_reject_insufficient_clean_rows():
    """Fewer clean rows than params → underdetermined."""
    signal = np.array([1.0, np.nan, np.nan])
    factor = np.arange(3.0)
    with pytest.raises(KuantValueError) as exc:
        neutralize(signal, {"f": factor})
    assert "clean" in str(exc.value).lower() or "underdet" in str(exc.value).lower()


# ---------- collinearity warning ----------------------------------------


def test_collinear_factors_warn():
    """Two identical factors → condition number blows up, warning fires."""
    rng = np.random.default_rng(0)
    signal = rng.standard_normal(200)
    f1 = rng.standard_normal(200)
    f2 = f1.copy()  # identical
    with pytest.warns(KuantNumericWarning) as record:
        r = neutralize(signal, {"f1": f1, "f2": f2})
    assert any("KW-COLLINEAR-FACTORS" in str(w.message) for w in record)
    assert r.condition_number > 1e10


def test_orthogonal_factors_no_warning():
    """Orthogonal factors → clean fit, no warning."""
    rng = np.random.default_rng(0)
    signal = rng.standard_normal(500)
    import warnings as _w

    with _w.catch_warnings():
        _w.simplefilter("error", KuantNumericWarning)
        neutralize(
            signal,
            {
                "f1": rng.standard_normal(500),
                "f2": rng.standard_normal(500),
                "f3": rng.standard_normal(500),
            },
        )


# ---------- error contract ------------------------------------------------


def test_reject_length_mismatch():
    with pytest.raises(Exception):
        neutralize(np.arange(50.0), {"f": np.arange(60.0)})


def test_reject_empty_factors():
    with pytest.raises(KuantValueError):
        neutralize(np.arange(50.0), [])


def test_reject_wrong_factor_type():
    with pytest.raises(KuantValueError):
        neutralize(np.arange(50.0), "not a factor")


def test_reject_3d_factors():
    with pytest.raises(KuantShapeError):
        neutralize(np.arange(50.0), np.zeros((50, 2, 2)))


# ---------- realistic pattern: two-factor neutralization -----------------


def test_two_factor_beta_recovery():
    rng = np.random.default_rng(42)
    T = 1000
    f1 = rng.standard_normal(T)
    f2 = rng.standard_normal(T)
    signal = 0.3 * f1 - 0.5 * f2 + 0.05 * rng.standard_normal(T)
    r = neutralize(signal, {"f1": f1, "f2": f2})
    assert abs(r.betas["f1"] - 0.3) < 0.02
    assert abs(r.betas["f2"] - (-0.5)) < 0.02
    assert r.r2 > 0.99  # near-perfect explanation
