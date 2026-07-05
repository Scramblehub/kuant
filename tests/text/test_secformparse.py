"""Tests for kuant.text.secformparse."""

from __future__ import annotations

import pytest

from kuant.errors import KuantEncodingError, KuantValueError
from kuant.text.secformparse import SECForm, secformparse


# ---------- category classification -------------------------------------


@pytest.mark.parametrize(
    "form,category",
    [
        ("10-K", "annual"),
        ("20-F", "annual"),
        ("10-Q", "quarterly"),
        ("8-K", "current"),
        ("6-K", "current"),
        ("S-1", "registration"),
        ("S-3", "registration"),
        ("S-4", "registration"),
        ("DEF 14A", "proxy"),
        ("PRE 14A", "proxy"),
        ("3", "insider"),
        ("4", "insider"),
        ("5", "insider"),
        ("13F-HR", "institutional"),
        ("SC 13G", "institutional"),
        ("SC 13D", "institutional"),
    ],
)
def test_known_forms_categorized(form, category):
    r = secformparse(form)
    assert r.category == category


def test_unknown_form_falls_to_other():
    r = secformparse("XYZ-9999")
    assert r.category == "other"


# ---------- amendment detection -----------------------------------------


def test_amendment_suffix_stripped_from_base():
    r = secformparse("10-K/A")
    assert r.is_amendment is True
    assert r.base_form == "10-K"
    assert r.form_type == "10-K/A"
    assert r.category == "annual"


def test_non_amendment_is_amendment_false():
    r = secformparse("10-K")
    assert r.is_amendment is False
    assert r.base_form == "10-K"


def test_amendment_of_registration():
    r = secformparse("S-1/A")
    assert r.is_amendment is True
    assert r.category == "registration"


def test_insider_amendment():
    r = secformparse("4/A")
    assert r.is_amendment is True
    assert r.category == "insider"


# ---------- normalization ------------------------------------------------


def test_lowercase_normalized_to_upper():
    r = secformparse("10-k")
    assert r.form_type == "10-K"
    assert r.category == "annual"


def test_whitespace_trimmed_and_collapsed():
    r = secformparse("  DEF  14A  ")
    assert r.form_type == "DEF 14A"
    assert r.category == "proxy"


def test_original_preserved():
    r = secformparse("  10-K  ")
    assert r.original == "10-K"


# ---------- return object ------------------------------------------------


def test_returns_dataclass():
    assert isinstance(secformparse("10-K"), SECForm)


def test_summary_contains_metadata():
    s = secformparse("10-K/A").summary()
    assert "SECForm" in s
    assert "10-K/A" in s
    assert "annual" in s


# ---------- error contract ----------------------------------------------


def test_reject_empty_string():
    with pytest.raises(KuantValueError):
        secformparse("   ")


def test_reject_bytes_input():
    with pytest.raises(KuantEncodingError):
        secformparse(b"10-K")


def test_reject_int_input():
    with pytest.raises(KuantValueError):
        secformparse(10)
