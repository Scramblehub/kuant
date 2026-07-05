"""Tests for kuant.text.occparse."""

from __future__ import annotations

from datetime import date

import pytest

from kuant.errors import KuantEncodingError, KuantEncodingWarning, KuantValueError
from kuant.text.occparse import OCCSymbol, occparse


# ---------- happy path ---------------------------------------------------


def test_aapl_call():
    r = occparse("AAPL240119C00150000")
    assert r.underlying == "AAPL"
    assert r.expiry == date(2024, 1, 19)
    assert r.right == "C"
    assert r.strike == 150.0
    assert r.is_call and not r.is_put


def test_spy_put():
    r = occparse("SPY240315P00450500")
    assert r.underlying == "SPY"
    assert r.right == "P"
    assert r.strike == 450.5


def test_fractional_strike_penny_precision():
    r = occparse("SPY240315P00450125")
    assert r.strike == 450.125


def test_single_char_root():
    r = occparse("F240119C00012500")
    assert r.underlying == "F"
    assert r.strike == 12.5


def test_root_with_share_class():
    r = occparse("BRK.B240119C00500000")
    assert r.underlying == "BRK.B"
    assert r.strike == 500.0


def test_max_length_root():
    """OCC root allows up to 6 chars."""
    r = occparse("ABCDEF240119C00050000")
    assert r.underlying == "ABCDEF"


# ---------- fixed 21-char (space-padded) form ---------------------------


def test_fixed_21_char_form():
    """Root space-padded to 6 chars (older vendor format)."""
    r = occparse("AAPL  240119C00150000")
    assert r.underlying == "AAPL"
    assert r.strike == 150.0


def test_fixed_form_short_root():
    r = occparse("F     240119C00012500")
    assert r.underlying == "F"


# ---------- case normalization + trimming -------------------------------


def test_lowercase_normalized_to_upper():
    r = occparse("aapl240119c00150000")
    assert r.underlying == "AAPL"
    assert r.right == "C"


def test_whitespace_trimmed():
    r = occparse("  AAPL240119C00150000  ")
    assert r.underlying == "AAPL"


# ---------- return object contract --------------------------------------


def test_returns_dataclass():
    r = occparse("SPY240315C00450000")
    assert isinstance(r, OCCSymbol)


def test_original_preserved():
    orig = "AAPL240119C00150000"
    r = occparse(orig)
    assert r.original == orig


def test_summary_contains_metadata():
    s = occparse("AAPL240119C00150000").summary()
    assert "OCCSymbol" in s
    assert "AAPL" in s and "150" in s


# ---------- error contract ----------------------------------------------


def test_reject_bytes_input():
    with pytest.raises(KuantEncodingError) as exc:
        occparse(b"AAPL240119C00150000")
    assert "bytes" in str(exc.value)
    assert "KE-ENCODING-BYTES" in str(exc.value)


def test_reject_non_string_input():
    with pytest.raises(KuantValueError):
        occparse(12345)


def test_reject_malformed_missing_digit():
    with pytest.raises(KuantValueError):
        occparse("AAPL24019C00150000")  # 5 date digits instead of 6


def test_reject_bad_right():
    with pytest.raises(KuantValueError):
        occparse("AAPL240119X00150000")  # X instead of C/P


def test_reject_bad_root_leading_digit():
    with pytest.raises(KuantValueError):
        occparse("1AAPL240119C00150000")  # root must start with letter


def test_reject_impossible_date():
    with pytest.raises(KuantValueError) as exc:
        occparse("AAPL240230C00150000")  # Feb 30
    assert "calendar day" in str(exc.value)


def test_reject_zero_strike():
    with pytest.raises(KuantValueError):
        occparse("AAPL240119C00000000")


# ---------- encoding warning -------------------------------------------


def test_replacement_character_warns():
    """U+FFFD in the input signals a broken upstream decode."""
    with pytest.warns(KuantEncodingWarning) as record:
        # The kernel will emit the warning THEN fail to parse. Either
        # outcome is fine; we only care that the warning fired.
        try:
            occparse("AAPL2401�9C00150000")
        except Exception:
            pass
    assert any("KW-ENCODING-REPLACEMENT" in str(w.message) for w in record)


def test_nul_byte_warns():
    with pytest.warns(KuantEncodingWarning) as record:
        try:
            occparse("AAPL240119C00150000\x00")
        except Exception:
            pass
    assert any("KW-ENCODING-REPLACEMENT" in str(w.message) for w in record)


# ---------- realistic sample of live OCC symbols -----------------------


@pytest.mark.parametrize(
    "sym,expected",
    [
        ("AAPL240119C00150000", ("AAPL", date(2024, 1, 19), "C", 150.0)),
        ("SPY240315P00450500", ("SPY", date(2024, 3, 15), "P", 450.5)),
        ("TSLA250620C00500000", ("TSLA", date(2025, 6, 20), "C", 500.0)),
        ("QQQ261218P00300000", ("QQQ", date(2026, 12, 18), "P", 300.0)),
        ("META240119C99999999", ("META", date(2024, 1, 19), "C", 99999.999)),
    ],
)
def test_realistic_symbols(sym, expected):
    r = occparse(sym)
    assert (r.underlying, r.expiry, r.right, r.strike) == expected
