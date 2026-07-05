"""Tests for kuant.data.corpaction."""

from __future__ import annotations

import numpy as np
import pytest

from kuant.data.corpaction import CorpActionResult, corpaction
from kuant.errors import KuantNumericWarning, KuantValueError


# ---------- split-only mode -----------------------------------------------


def test_split_backward_halves_pre_split_prices():
    """2-for-1 split at pos 2: rows [0, 1] get factor 0.5."""
    raw = np.array([100.0, 100, 50, 51, 52])
    r = corpaction(
        raw,
        split_positions=[2],
        split_ratios=[2.0],
        mode="split_only",
        direction="backward",
    )
    assert r.prices.tolist() == [50.0, 50.0, 50.0, 51.0, 52.0]
    assert r.cumulative_factor.tolist() == [0.5, 0.5, 1.0, 1.0, 1.0]
    assert r.n_splits_applied == 1


def test_split_forward_scales_post_split_prices():
    """Same split forward-adjusted: rows [2..] get factor 2.0."""
    raw = np.array([100.0, 100, 50, 51, 52])
    r = corpaction(
        raw,
        split_positions=[2],
        split_ratios=[2.0],
        mode="split_only",
        direction="forward",
    )
    assert r.prices.tolist() == [100.0, 100.0, 100.0, 102.0, 104.0]
    assert r.cumulative_factor.tolist() == [1.0, 1.0, 2.0, 2.0, 2.0]


def test_reverse_split_backward():
    """1-for-10 reverse split (ratio=0.1) at pos 2 multiplies pre-split by 10."""
    raw = np.array([1.0, 1, 10, 11, 12])
    r = corpaction(
        raw,
        split_positions=[2],
        split_ratios=[0.1],
        mode="split_only",
        direction="backward",
    )
    assert r.prices.tolist() == [10.0, 10.0, 10.0, 11.0, 12.0]


def test_multiple_splits_compose_multiplicatively():
    """Two 2-for-1 splits at pos 2 and pos 4 compound to factor 0.25 for rows [0..1]."""
    raw = np.array([400.0, 400, 200, 200, 100, 101])
    r = corpaction(
        raw,
        split_positions=[2, 4],
        split_ratios=[2.0, 2.0],
        mode="split_only",
        direction="backward",
    )
    # Row 0: 400 * 0.5 * 0.5 = 100
    # Row 2: 200 * 0.5 = 100
    # Row 4: 100 * 1 = 100
    assert r.prices.tolist() == [100.0, 100.0, 100.0, 100.0, 100.0, 101.0]


def test_split_only_ignores_dividend_inputs():
    """When mode='split_only', supplied dividend inputs are ignored."""
    raw = np.array([100.0, 100, 100])
    r = corpaction(
        raw,
        split_positions=[1],
        split_ratios=[2.0],
        dividend_positions=[2],
        dividend_amounts=[5.0],
        mode="split_only",
        direction="backward",
    )
    # Only the split is applied.
    assert r.n_dividends_applied == 0
    # Row 0: 100 * 0.5 = 50; Row 1: unchanged; Row 2: unchanged.
    assert r.prices.tolist() == [50.0, 100.0, 100.0]


# ---------- total-return mode ---------------------------------------------


def test_total_return_backward_dividend_adjusts_pre_ex():
    """Ex-div of $2 at close $100 → factor 98/100 = 0.98 on rows before ex."""
    raw = np.array([100.0, 100, 100, 100])
    r = corpaction(
        raw,
        dividend_positions=[2],
        dividend_amounts=[2.0],
        mode="total_return",
        direction="backward",
    )
    assert r.n_dividends_applied == 1
    # Rows 0, 1 get factor 98/100 = 0.98.
    assert np.allclose(r.prices, [98.0, 98.0, 100.0, 100.0])


def test_total_return_forward_dividend_scales_post_ex():
    """Forward direction: rows AT-OR-AFTER ex get factor c/(c-d) = 100/98."""
    raw = np.array([100.0, 100, 100, 100])
    r = corpaction(
        raw,
        dividend_positions=[2],
        dividend_amounts=[2.0],
        mode="total_return",
        direction="forward",
    )
    expected = np.array([100.0, 100.0, 100.0 * 100 / 98, 100.0 * 100 / 98])
    assert np.allclose(r.prices, expected)


def test_split_and_dividend_backward_compose():
    """Split at pos 2 + dividend at pos 3 both apply to pre-event rows."""
    raw = np.array([100.0, 100, 50, 50, 48])
    r = corpaction(
        raw,
        split_positions=[2],
        split_ratios=[2.0],
        dividend_positions=[3],
        dividend_amounts=[2.0],
        mode="total_return",
        direction="backward",
    )
    # Row 0: 100 * 0.5 (split at 2) * 48/50 (div at 3) = 48.
    # Row 1: 100 * 0.5 * 48/50 = 48.
    # Row 2: 50 * 48/50 = 48.
    # Row 3: 50 (unchanged; ex-div applies to rows [0..2]).
    # Row 4: 48.
    assert np.allclose(r.prices, [48.0, 48.0, 48.0, 50.0, 48.0])


def test_no_events_returns_raw_prices():
    raw = np.array([100.0, 101, 102, 103])
    r = corpaction(raw, mode="split_only")
    assert r.prices.tolist() == raw.tolist()
    assert r.cumulative_factor.tolist() == [1.0, 1.0, 1.0, 1.0]
    assert r.n_splits_applied == 0
    assert r.n_dividends_applied == 0


def test_empty_price_series():
    r = corpaction(np.array([], dtype=np.float64))
    assert r.prices.size == 0
    assert r.cumulative_factor.size == 0


# ---------- return object -------------------------------------------------


def test_returns_dataclass():
    r = corpaction(np.array([100.0, 101]))
    assert isinstance(r, CorpActionResult)


def test_summary_string():
    r = corpaction(
        np.array([100.0, 100, 50]),
        split_positions=[2],
        split_ratios=[2.0],
    )
    s = r.summary()
    assert "CorpActionResult" in s
    assert "n_splits_applied" in s


def test_to_parquet_roundtrip(tmp_path):
    pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    r = corpaction(
        np.array([100.0, 100, 50]),
        split_positions=[2],
        split_ratios=[2.0],
    )
    path = tmp_path / "adj.parquet"
    r.to_parquet(path)
    cols = pq.read_table(path).column_names
    assert set(cols) == {"prices", "cumulative_factor"}


# ---------- warnings ------------------------------------------------------


def test_extreme_split_ratio_warns():
    """Ratios >100 or <0.001 are the typo signature — warn."""
    raw = np.array([100.0, 100])
    with pytest.warns(KuantNumericWarning) as record:
        corpaction(
            raw,
            split_positions=[1],
            split_ratios=[1000.0],  # very extreme
            mode="split_only",
        )
    assert any("KW-SPLIT-EXTREME" in str(w.message) for w in record)


def test_extreme_low_split_ratio_warns():
    raw = np.array([100.0, 100])
    with pytest.warns(KuantNumericWarning):
        corpaction(
            raw,
            split_positions=[1],
            split_ratios=[0.0001],
            mode="split_only",
        )


def test_degenerate_dividend_warns_and_skips():
    """Dividend >= close would produce a non-positive factor — skip + warn."""
    raw = np.array([10.0, 10, 10])
    with pytest.warns(KuantNumericWarning) as record:
        r = corpaction(
            raw,
            dividend_positions=[2],
            dividend_amounts=[15.0],  # > close
            mode="total_return",
        )
    assert any("KW-DIV-DEGENERATE" in str(w.message) for w in record)
    # Prices should be unchanged since the degenerate dividend is skipped.
    assert r.prices.tolist() == raw.tolist()


# ---------- error contract ------------------------------------------------


def test_reject_bad_mode():
    with pytest.raises(KuantValueError) as exc:
        corpaction(np.array([100.0]), mode="jumbo")
    assert "mode" in str(exc.value)


def test_reject_bad_direction():
    with pytest.raises(KuantValueError) as exc:
        corpaction(np.array([100.0]), direction="sideways")
    assert "direction" in str(exc.value)


def test_reject_2d_prices():
    with pytest.raises(KuantValueError):
        corpaction(np.zeros((3, 2)))


def test_reject_split_mutex():
    """split_positions supplied without ratios (or vice versa) → error."""
    with pytest.raises(KuantValueError) as exc:
        corpaction(
            np.array([100.0, 100]),
            split_positions=[1],
            # missing split_ratios
        )
    assert "mutex" in str(exc.value).lower() or "MUTEX" in str(exc.value)


def test_reject_out_of_bounds_split_position():
    with pytest.raises(KuantValueError):
        corpaction(
            np.array([100.0, 100, 50]),
            split_positions=[10],  # beyond len(prices)
            split_ratios=[2.0],
        )


def test_reject_non_integer_split_position():
    with pytest.raises(KuantValueError):
        corpaction(
            np.array([100.0, 100]),
            split_positions=np.array([1.5]),  # not integer
            split_ratios=[2.0],
        )


# ---------- integration: sanity on a realistic pattern --------------------


def test_apple_style_2020_split():
    """Apple-style 4-for-1 split (ratio=4.0) on a 6-day toy: verify pre-split
    prices are quartered under backward adjustment."""
    # Fictional dates around a hypothetical 4-for-1: prices halve...err quarter
    # at the split boundary.
    raw = np.array([400.0, 404, 408, 100, 101, 102])
    r = corpaction(
        raw,
        split_positions=[3],
        split_ratios=[4.0],
        mode="split_only",
        direction="backward",
    )
    # Rows [0..2] scaled by 1/4.
    assert np.allclose(r.prices[:3], [100.0, 101.0, 102.0])
    # Post-split rows unchanged.
    assert r.prices[3:].tolist() == [100.0, 101.0, 102.0]
