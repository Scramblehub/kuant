"""Validate a CUSIP checksum and normalize the input.

A CUSIP is a 9-character alphanumeric identifier issued by CUSIP Global
Services for North American securities. The last character is a
mod-10 check digit computed from the first 8.

Structure:

    Positions 0-5  Issuer code (6 chars)
    Positions 6-7  Issue code (2 chars: security type + variant)
    Position  8    Check digit (1 char, always numeric)

Character encoding for the check-digit algorithm:

    '0'-'9'  → 0-9
    'A'-'Z'  → 10-35 (A=10, B=11, ...)
    '*'      → 36
    '@'      → 37
    '#'      → 38

Algorithm:

    for each of positions 0..7:
        v = numeric value of that character
        if position is odd (1, 3, 5, 7):
            v = v * 2
        sum the digits of v; accumulate
    check = (10 - accumulator % 10) % 10

The kernel returns a `CUSIPValidation` dataclass with the parsed
issuer / issue codes, the computed vs supplied check digits, and an
`is_valid` bool.

Design: docs/kernels/text/cusipvalidate.md.
"""

from __future__ import annotations

from dataclasses import dataclass

from kuant.errors import KuantValueError

from .occparse import _check_str_input

_ALLOWED_ALPHA_MAP = {chr(ord("A") + i): 10 + i for i in range(26)}
_ALLOWED_SPECIAL_MAP = {"*": 36, "@": 37, "#": 38}


@dataclass
class CUSIPValidation:
    """Result of `cusipvalidate`.

    Attributes
    ----------
    input : str
        The exact string the user passed in (whitespace-stripped).
    normalized : str or None
        9-character uppercase form. `None` if the input length or
        character set is invalid.
    is_valid : bool
        True iff the input has a valid character set, 9-character
        length, and the supplied check digit matches the computed one.
    issuer : str or None
        First 6 characters of the normalized form.
    issue : str or None
        Characters 7-8 of the normalized form.
    supplied_check_digit : int or None
        The 9th character interpreted as a digit. `None` if it's not
        a digit.
    computed_check_digit : int or None
        The check digit computed from the first 8 characters. `None`
        if computation was not possible (bad chars).
    reason : str or None
        Human-readable diagnostic when `is_valid` is False.
    """

    input: str
    normalized: str | None
    is_valid: bool
    issuer: str | None
    issue: str | None
    supplied_check_digit: int | None
    computed_check_digit: int | None
    reason: str | None

    def summary(self) -> str:
        parts = [
            "=== CUSIPValidation ===",
            f"input:                {self.input!r}",
            f"normalized:           {self.normalized!r}",
            f"is_valid:             {self.is_valid}",
            f"issuer:               {self.issuer!r}",
            f"issue:                {self.issue!r}",
            f"supplied check:       {self.supplied_check_digit!r}",
            f"computed check:       {self.computed_check_digit!r}",
        ]
        if self.reason:
            parts.append(f"reason:               {self.reason}")
        return "\n".join(parts)


def _char_value(c: str) -> int | None:
    """Return the CUSIP-check numeric value of one character, or None."""
    if c.isdigit():
        return int(c)
    if c in _ALLOWED_ALPHA_MAP:
        return _ALLOWED_ALPHA_MAP[c]
    if c in _ALLOWED_SPECIAL_MAP:
        return _ALLOWED_SPECIAL_MAP[c]
    return None


def _compute_check_digit(first8: str) -> int | None:
    """Compute the CUSIP check digit from the first 8 chars.

    Returns None if any character is outside the allowed alphabet.
    """
    total = 0
    for i, c in enumerate(first8):
        v = _char_value(c)
        if v is None:
            return None
        if i % 2 == 1:  # odd positions (1, 3, 5, 7) double
            v *= 2
        total += (v // 10) + (v % 10)
    return (10 - total % 10) % 10


def cusipvalidate(cusip) -> CUSIPValidation:
    """Validate a CUSIP checksum and normalize the input.

    Parameters
    ----------
    cusip : str
        A 9-character CUSIP. Case-insensitive; leading and trailing
        whitespace is stripped.

    Returns
    -------
    CUSIPValidation

    Raises
    ------
    KuantEncodingError
        If `cusip` is bytes or contains a U+FFFD / NUL byte.

    Notes
    -----
    - The result's `.is_valid` is False rather than raising for bad-
      length or bad-character inputs. That matches how users
      typically iterate over vendor data: log the bad rows, keep going.
      Callers who prefer hard failure can `assert r.is_valid`.

    Examples
    --------
    >>> # Apple: known valid CUSIP.
    >>> r = cusipvalidate('037833100')
    >>> r.is_valid, r.issuer, r.issue
    (True, '037833', '10')
    >>> # Same input with the wrong check digit.
    >>> cusipvalidate('037833109').is_valid
    False
    """
    _check_str_input(cusip, "cusip", kernel="cusipvalidate")

    stripped = cusip.strip()
    upper = stripped.upper()

    if len(upper) != 9:
        return CUSIPValidation(
            input=stripped,
            normalized=None,
            is_valid=False,
            issuer=None,
            issue=None,
            supplied_check_digit=None,
            computed_check_digit=None,
            reason=f"length {len(upper)} != 9",
        )

    # Verify every character is in the allowed alphabet.
    for c in upper:
        if _char_value(c) is None:
            return CUSIPValidation(
                input=stripped,
                normalized=None,
                is_valid=False,
                issuer=None,
                issue=None,
                supplied_check_digit=None,
                computed_check_digit=None,
                reason=f"character {c!r} not in CUSIP alphabet",
            )

    supplied = upper[-1]
    if not supplied.isdigit():
        # The check digit slot must be a plain digit.
        return CUSIPValidation(
            input=stripped,
            normalized=upper,
            is_valid=False,
            issuer=upper[:6],
            issue=upper[6:8],
            supplied_check_digit=None,
            computed_check_digit=_compute_check_digit(upper[:8]),
            reason=f"check digit {supplied!r} is not a digit",
        )

    computed = _compute_check_digit(upper[:8])
    if computed is None:
        # Shouldn't hit; every char passed _char_value above.
        raise KuantValueError(
            f"kuant.cusipvalidate: internal error computing check digit for "
            f"{upper[:8]!r}.  [KE-VAL-RANGE]\n"
            f"  → Fix: this is a bug; please report"
        )
    supplied_int = int(supplied)
    is_valid = supplied_int == computed
    reason = None if is_valid else f"check digit mismatch: expected {computed}, got {supplied_int}"

    return CUSIPValidation(
        input=stripped,
        normalized=upper,
        is_valid=is_valid,
        issuer=upper[:6],
        issue=upper[6:8],
        supplied_check_digit=supplied_int,
        computed_check_digit=computed,
        reason=reason,
    )


__all__ = ["cusipvalidate", "CUSIPValidation"]
