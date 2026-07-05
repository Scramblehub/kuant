"""Tests for kuant.text.cusipvalidate."""

from __future__ import annotations

import pytest

from kuant.errors import KuantEncodingError
from kuant.text.cusipvalidate import (
    CUSIPValidation,
    _char_value,
    _compute_check_digit,
    cusipvalidate,
)


# ---------- known-valid real-world CUSIPs ------------------------------


@pytest.mark.parametrize(
    "cusip",
    [
        "037833100",  # Apple Inc.
        "594918104",  # Microsoft
        "023135106",  # Amazon
        "88160R101",  # Tesla
        "02079K305",  # Alphabet Class A
        "30303M102",  # Meta Platforms
    ],
)
def test_real_world_valid_cusips(cusip):
    r = cusipvalidate(cusip)
    assert r.is_valid, f"{cusip} should validate: {r.reason}"


# ---------- checksum algorithm on synthetic cases ----------------------


def test_wrong_check_digit_fails():
    """Corrupt the check digit; the algorithm must catch it."""
    r = cusipvalidate("037833109")  # Apple's real is ...100
    assert not r.is_valid
    assert r.supplied_check_digit == 9
    assert r.computed_check_digit == 0


def test_normalized_is_uppercase_9char():
    r = cusipvalidate("037833100")
    assert r.normalized == "037833100"


def test_lowercase_normalized_to_upper():
    r = cusipvalidate("88160r101")
    assert r.is_valid
    assert r.normalized == "88160R101"


def test_leading_and_trailing_whitespace_stripped():
    r = cusipvalidate("  037833100  ")
    assert r.is_valid
    assert r.input == "037833100"


# ---------- issuer / issue extraction ----------------------------------


def test_issuer_and_issue_split():
    r = cusipvalidate("037833100")
    assert r.issuer == "037833"
    assert r.issue == "10"


# ---------- invalid-input paths (no exception, is_valid=False) ---------


def test_wrong_length_returns_invalid_not_exception():
    """<9 or >9 chars → is_valid=False with a reason, not an exception."""
    for cusip in ["12345", "12345678", "1234567890", ""]:
        r = cusipvalidate(cusip)
        assert not r.is_valid
        assert r.reason is not None
        assert "length" in r.reason


def test_illegal_character_returns_invalid():
    r = cusipvalidate("037833!00")  # ! not in CUSIP alphabet
    assert not r.is_valid
    assert "CUSIP alphabet" in r.reason


def test_non_digit_check_digit_returns_invalid():
    """The last position must be a plain digit."""
    r = cusipvalidate("03783310A")
    assert not r.is_valid
    assert "not a digit" in r.reason


# ---------- character-value helper -------------------------------------


def test_char_value_digits():
    for i in range(10):
        assert _char_value(str(i)) == i


def test_char_value_letters():
    assert _char_value("A") == 10
    assert _char_value("B") == 11
    assert _char_value("Z") == 35


def test_char_value_special_chars():
    assert _char_value("*") == 36
    assert _char_value("@") == 37
    assert _char_value("#") == 38


def test_char_value_bad_char():
    assert _char_value("!") is None


def test_compute_check_digit_apple():
    """Recompute Apple's check digit explicitly."""
    assert _compute_check_digit("03783310") == 0


# ---------- return object ----------------------------------------------


def test_returns_dataclass():
    assert isinstance(cusipvalidate("037833100"), CUSIPValidation)


def test_summary_string():
    s = cusipvalidate("037833100").summary()
    assert "CUSIPValidation" in s
    assert "037833100" in s


def test_summary_on_invalid_shows_reason():
    s = cusipvalidate("bad").summary()
    assert "length" in s.lower() or "reason" in s.lower()


# ---------- error contract ---------------------------------------------


def test_reject_bytes_input():
    with pytest.raises(KuantEncodingError):
        cusipvalidate(b"037833100")


def test_reject_non_string_input():
    with pytest.raises(Exception):
        cusipvalidate(37833100)


# ---------- comprehensive: recompute for a batch of real CUSIPs --------


def test_apple_full_algorithm_walkthrough():
    """Explicit worked example: Apple 037833100.

    Positions 0..7: '0','3','7','8','3','3','1','0'
    Values:         0, 3, 7, 8, 3, 3, 1, 0
    Odd-position doubled: [0, 6, 7, 16, 3, 6, 1, 0]
    Digit sums:           [0, 6, 7,  7, 3, 6, 1, 0]  # 16 → 1+6=7
    Sum:                  30
    Check:                (10 - 30 % 10) % 10 = 0
    """
    computed = _compute_check_digit("03783310")
    assert computed == 0
