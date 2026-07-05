"""Tests for kuant.text.tickernorm."""

from __future__ import annotations

import pytest

from kuant.errors import KuantEncodingError, KuantValueError
from kuant.text.tickernorm import TickerParts, tickernorm


# ---------- venue round-trips (BRK.B) ----------------------------------


@pytest.mark.parametrize(
    "src,venue,expected",
    [
        ("BRK.B", "yahoo", "BRK-B"),
        ("BRK-B", "canonical", "BRK.B"),
        ("BRK-B", "wiki", "BRK.B"),
        ("BRK/B", "yahoo", "BRK-B"),
        ("BRK.B", "google", "BRK/B"),
        ("BRK/B", "canonical", "BRK.B"),
        ("BF.B", "yahoo", "BF-B"),
        ("BF-B", "google", "BF/B"),
    ],
)
def test_class_share_venue_conversion(src, venue, expected):
    assert tickernorm(src, venue=venue) == expected


# ---------- CRSP permno suffix ------------------------------------------


def test_permno_stripped_for_non_crsp_venues():
    assert tickernorm("ACME.99999", venue="yahoo") == "ACME"
    assert tickernorm("ACME.99999", venue="wiki") == "ACME"
    assert tickernorm("ACME.99999", venue="google") == "ACME"


def test_permno_preserved_for_crsp_venue():
    assert tickernorm("ACME.99999", venue="crsp") == "ACME.99999"


def test_permno_detected_in_parts():
    p = tickernorm("ACME.99999")
    assert p.root == "ACME"
    assert p.permno == 99999
    assert p.share_class is None


def test_permno_min_4_digits_and_max_7():
    """The permno regex uses 4-7 digits; anything else is treated as a
    share class (or should not match at all)."""
    # 3-digit trailing block is not a permno (falls through to class share
    # detection).
    p = tickernorm("XYZ.123")
    assert p.permno is None
    assert p.share_class == "123"


# ---------- passthrough (no separator, no permno) ---------------------


def test_plain_ticker_uppercased():
    assert tickernorm("ibm", venue="yahoo") == "IBM"
    assert tickernorm("aapl", venue="canonical") == "AAPL"


def test_plain_ticker_all_venues_identity():
    for v in ("yahoo", "wiki", "google", "canonical", "crsp"):
        assert tickernorm("MSFT", venue=v) == "MSFT"


def test_leading_and_trailing_whitespace_stripped():
    assert tickernorm("  BRK.B  ", venue="yahoo") == "BRK-B"


# ---------- TickerParts contract ---------------------------------------


def test_returns_ticker_parts_when_no_venue():
    p = tickernorm("BRK.B")
    assert isinstance(p, TickerParts)
    assert p.root == "BRK"
    assert p.share_class == "B"
    assert p.permno is None
    assert p.original == "BRK.B"


def test_parts_render_at_call_time():
    p = tickernorm("BRK.B")
    assert p.render("yahoo") == "BRK-B"
    assert p.render("google") == "BRK/B"
    assert p.render("canonical") == "BRK.B"


def test_parts_render_rejects_bad_venue():
    p = tickernorm("BRK.B")
    with pytest.raises(KuantValueError):
        p.render("bloomberg")


# ---------- error contract ---------------------------------------------


def test_reject_empty_input():
    with pytest.raises(KuantValueError):
        tickernorm("   ")


def test_reject_bytes_input():
    with pytest.raises(KuantEncodingError):
        tickernorm(b"BRK.B")


def test_reject_multiple_separators():
    with pytest.raises(KuantValueError):
        tickernorm("A.B.C")


def test_reject_bad_characters():
    with pytest.raises(KuantValueError):
        tickernorm("BRK.B!")


def test_reject_bad_venue_in_top_level_call():
    with pytest.raises(KuantValueError):
        tickernorm("IBM", venue="bloomberg")


# ---------- realistic-shape inputs -------------------------------------


@pytest.mark.parametrize(
    "src,expected_root,expected_class,expected_permno",
    [
        # Yahoo / Wikipedia S&P 500 samples
        ("BRK.B", "BRK", "B", None),
        ("BF.B", "BF", "B", None),
        ("BRK-A", "BRK", "A", None),
        # Permno-tagged shape with placeholder identifiers
        ("XYZ.12345", "XYZ", None, 12345),
        ("QQQQ.67890", "QQQQ", None, 67890),
        # Plain
        ("SPY", "SPY", None, None),
        ("AAPL", "AAPL", None, None),
    ],
)
def test_realistic_inputs(src, expected_root, expected_class, expected_permno):
    p = tickernorm(src)
    assert p.root == expected_root
    assert p.share_class == expected_class
    assert p.permno == expected_permno
