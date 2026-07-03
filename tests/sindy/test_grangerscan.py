"""Test suite for kuant.sindy.grangerscan."""

from __future__ import annotations

import numpy as np
import pytest

# statsmodels is an optional dep; skip the whole file cleanly if missing.
pytest.importorskip("statsmodels")

from kuant.sindy import grangerscan  # noqa: E402 — must come after importorskip


# ---------------------------------------------------------------------------
# 1. Basic detection: known Granger relationship should surface
# ---------------------------------------------------------------------------


def test_lag_1_relationship_is_detected(rng):
    """x Granger-causes y at lag 1 → 'x' appears in hits, 'noise' doesn't."""
    n = 500
    x = rng.normal(size=n)
    y = np.roll(x, 1) + rng.normal(scale=0.5, size=n)  # y_t = x_{t-1} + noise
    noise = rng.normal(size=n)

    result = grangerscan(y, {"x": x, "noise": noise}, horizons=[1, 2])
    hit_names = [h.candidate for h in result.hits]

    assert "x" in hit_names, "known Granger cause not detected"
    assert "noise" not in hit_names, "pure noise falsely flagged"


def test_lag_5_relationship_is_detected(rng):
    """x Granger-causes y at lag 5 → 'x' passes when we test horizon 5."""
    n = 600
    x = rng.normal(size=n)
    y = np.roll(x, 5) + rng.normal(scale=0.4, size=n)

    result = grangerscan(y, {"x": x}, horizons=[5])
    hit_horizons = [(h.candidate, h.horizon) for h in result.hits]

    assert ("x", 5) in hit_horizons


# ---------------------------------------------------------------------------
# 2. Result structure
# ---------------------------------------------------------------------------


def test_result_records_n_tests_and_bonferroni_alpha(rng):
    y = rng.normal(size=200)
    cands = {"a": rng.normal(size=200), "b": rng.normal(size=200)}
    result = grangerscan(y, cands, horizons=[1, 2, 5], alpha=0.05)

    assert result.n_tests == 2 * 3  # candidates × horizons
    assert result.bonferroni_alpha == pytest.approx(0.05 / 6)


def test_result_summary_prints(rng):
    """The summary() method should return a non-empty string."""
    y = rng.normal(size=200)
    result = grangerscan(y, {"a": rng.normal(size=200)}, horizons=[1])
    s = result.summary()
    assert isinstance(s, str)
    assert "Granger causality scan" in s
    assert "Bonferroni" in s


def test_hit_has_expected_fields(rng):
    """When there's a hit, it carries candidate/horizon/f_stat/p_value."""
    x = rng.normal(size=300)
    y = np.roll(x, 1) + rng.normal(scale=0.3, size=300)
    result = grangerscan(y, {"x": x}, horizons=[1])

    assert len(result.hits) > 0
    h = result.hits[0]
    assert h.candidate == "x"
    assert h.horizon == 1
    assert h.f_stat > 0
    assert 0.0 <= h.p_value <= 1.0


# ---------------------------------------------------------------------------
# 3. Edge cases
# ---------------------------------------------------------------------------


def test_too_few_observations_skipped():
    """Series with < 30 clean observations should be skipped, not crash."""
    y = np.arange(20, dtype=np.float64)
    result = grangerscan(y, {"x": np.arange(20, dtype=np.float64)}, horizons=[1])
    # No hits, no exception raised.
    assert isinstance(result.hits, list)


def test_nan_rows_are_dropped(rng):
    """NaN in either target or candidate should be filtered."""
    n = 300
    x = rng.normal(size=n)
    y = np.roll(x, 1) + rng.normal(scale=0.3, size=n)
    # Scatter ~10% NaN across both.
    idx_x = rng.choice(n, size=30, replace=False)
    idx_y = rng.choice(n, size=30, replace=False)
    x[idx_x] = np.nan
    y[idx_y] = np.nan

    # Should still detect the relationship on clean rows.
    result = grangerscan(y, {"x": x}, horizons=[1])
    assert any(h.candidate == "x" for h in result.hits)


def test_default_horizons():
    """When horizons=None, default = [1, 2, 5]."""
    rng = np.random.default_rng(42)
    y = rng.normal(size=200)
    result = grangerscan(y, {"a": rng.normal(size=200)}, horizons=None)
    assert result.n_tests == 3  # 1 candidate × 3 default horizons


def test_bonferroni_scales_with_test_count(rng):
    """Adding candidates tightens the per-test threshold."""
    y = rng.normal(size=200)
    single = grangerscan(y, {"a": rng.normal(size=200)}, horizons=[1], alpha=0.05)
    many = grangerscan(
        y,
        {f"c{i}": rng.normal(size=200) for i in range(10)},
        horizons=[1, 2, 5],
        alpha=0.05,
    )
    assert single.bonferroni_alpha > many.bonferroni_alpha


# ---------------------------------------------------------------------------
# 4. Verbose mode doesn't crash
# ---------------------------------------------------------------------------


def test_verbose_mode_runs(capsys, rng):
    x = rng.normal(size=200)
    y = np.roll(x, 1) + rng.normal(scale=0.3, size=200)
    grangerscan(y, {"x": x}, horizons=[1], verbose=True)
    captured = capsys.readouterr()
    # Should have printed something for each test.
    assert len(captured.out) > 0


def test_verbose_prints_skip_message_for_short_series(capsys):
    """Verbose mode reports the skipped series with reason."""
    y = np.arange(15, dtype=np.float64)
    grangerscan(y, {"x": np.arange(15, dtype=np.float64)}, horizons=[1], verbose=True)
    captured = capsys.readouterr()
    assert "too few" in captured.out.lower()
