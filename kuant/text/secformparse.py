"""Parse SEC form types into normalized components.

The SEC filing universe has a few dozen form types plus a suffix
convention for amendments (`/A`). Users routinely re-implement the
same normalization: strip trailing whitespace, upper-case, detect the
amendment suffix, bucket into a category (annual, quarterly, current,
registration, proxy, insider, institutional, other).

This kernel encapsulates that once.

Design: docs/kernels/text/secformparse.md.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from kuant.errors import KuantValueError

from .occparse import _check_str_input

# Category map. Order is deterministic; earlier matches win.
_CATEGORY_MAP = {
    "annual": frozenset({"10-K", "10-KT", "20-F", "40-F", "11-K"}),
    "quarterly": frozenset({"10-Q", "10-QT"}),
    "current": frozenset({"8-K", "6-K"}),
    "registration": frozenset(
        {
            "S-1",
            "S-3",
            "S-4",
            "S-8",
            "S-11",
            "F-1",
            "F-3",
            "F-4",
            "424B1",
            "424B2",
            "424B3",
            "424B4",
            "424B5",
        }
    ),
    "proxy": frozenset(
        {"DEF 14A", "PRE 14A", "DEFA14A", "DEFR14A", "DEFC14A", "DEF 14C", "PRE 14C"}
    ),
    "insider": frozenset({"3", "4", "5", "3/A", "4/A", "5/A"}),
    "institutional": frozenset(
        {"13F-HR", "13F-NT", "SC 13G", "SC 13D", "13F-HR/A", "SC 13G/A", "SC 13D/A"}
    ),
    "prospectus": frozenset({"485BPOS", "497", "497K", "N-CSR", "N-CSRS", "NPX", "N-Q"}),
}

# Reverse index: form → category.
_FORM_TO_CAT = {}
for cat, forms in _CATEGORY_MAP.items():
    for f in forms:
        # Insider forms in the list already carry /A; treat those as
        # amendments of the base form.
        _FORM_TO_CAT[f] = cat


@dataclass
class SECForm:
    """Parsed SEC form.

    Attributes
    ----------
    form_type : str
        Normalized form (upper-cased, whitespace trimmed). Includes the
        `/A` suffix if the input carried one.
    base_form : str
        The form without the amendment suffix. e.g. `'10-K'` for a
        `'10-K/A'`.
    is_amendment : bool
        True iff the input ended in `/A` (or is one of the insider form
        amendments 3/A, 4/A, 5/A which are also in the base list).
    category : str
        One of `'annual'`, `'quarterly'`, `'current'`, `'registration'`,
        `'proxy'`, `'insider'`, `'institutional'`, `'prospectus'`,
        `'other'`. `'other'` is used for anything not recognized.
    original : str
        The exact string the user passed in (whitespace-stripped).
    """

    form_type: str
    base_form: str
    is_amendment: bool
    category: str
    original: str

    def summary(self) -> str:
        parts = [
            "=== SECForm ===",
            f"form_type:     {self.form_type}",
            f"base_form:     {self.base_form}",
            f"is_amendment:  {self.is_amendment}",
            f"category:      {self.category}",
        ]
        return "\n".join(parts)


# Collapse repeated whitespace inside a form (e.g. 'DEF  14A' → 'DEF 14A').
_WS_RE = re.compile(r"\s+")


def secformparse(form) -> SECForm:
    """Parse and normalize a SEC form type string.

    Parameters
    ----------
    form : str
        Form type, as it appears in EDGAR (`'10-K'`, `'10-K/A'`,
        `'DEF 14A'`, `'8-K'`, `'S-1/A'`, ...). Case-insensitive.

    Returns
    -------
    SECForm

    Raises
    ------
    KuantEncodingError
        If `form` is bytes or contains a U+FFFD / NUL byte.
    KuantValueError
        If `form` is empty after whitespace strip.

    Examples
    --------
    >>> secformparse('10-K').category
    'annual'
    >>> r = secformparse('10-K/A')
    >>> r.is_amendment, r.base_form, r.category
    (True, '10-K', 'annual')
    >>> secformparse('def 14a').form_type
    'DEF 14A'
    >>> secformparse('Unknown-Type').category
    'other'
    """
    _check_str_input(form, "form", kernel="secformparse")

    normalized = _WS_RE.sub(" ", form.strip().upper())
    if not normalized:
        raise KuantValueError(
            f"kuant.secformparse: input {form!r} is empty after whitespace "
            f"strip.  [KE-VAL-RANGE]\n"
            f"  → Fix: pass a non-empty form type like '10-K' or 'DEF 14A'"
        )

    is_amendment = normalized.endswith("/A")
    base_form = normalized[:-2] if is_amendment else normalized

    # Look up category. Check the normalized form first (which may carry
    # /A for insider forms present in the base list). Then fall back to
    # the base_form for the general case.
    if normalized in _FORM_TO_CAT:
        category = _FORM_TO_CAT[normalized]
        # If we matched via /A (insider), treat as amendment.
        if normalized.endswith("/A"):
            is_amendment = True
    elif base_form in _FORM_TO_CAT:
        category = _FORM_TO_CAT[base_form]
    else:
        category = "other"

    return SECForm(
        form_type=normalized,
        base_form=base_form,
        is_amendment=is_amendment,
        category=category,
        original=form.strip(),
    )


__all__ = ["secformparse", "SECForm"]
