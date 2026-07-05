"""Normalize a ticker string across venue and vendor conventions.

The same equity gets written differently by every data source:

    Vendor / venue        BRK class B      BF class B    "IBM"
    Wikipedia / academic  BRK.B            BF.B          IBM
    Yahoo Finance         BRK-B            BF-B          IBM
    Google / some feeds   BRK/B            BF/B          IBM
    With numeric ID tag   BRK.99999        BF.99999      IBM.99999

Users routinely re-implement this. `tickernorm` ships the canonical
mappings and a two-step API: parse to a `TickerParts` record, then
render into any target venue.

Design: docs/kernels/text/tickernorm.md.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from kuant.errors import KuantValueError

from .occparse import _check_str_input

_VENUES = ("yahoo", "wiki", "google", "crsp", "canonical")

# Match a trailing '.digits' block (the CRSP permno suffix pattern).
# Root is anything before the last dot; permno is 4-7 trailing digits.
_PERMNO_RE = re.compile(r"^(?P<root>.+)\.(?P<permno>\d{4,7})$")


@dataclass
class TickerParts:
    """Parsed ticker components.

    Attributes
    ----------
    root : str
        Base ticker, uppercased. e.g. `'BRK'`, `'IBM'`.
    share_class : str or None
        Share class suffix if present. e.g. `'B'` for `BRK.B`. `None`
        if the ticker has no class marker.
    permno : int or None
        CRSP permno tag if the input was in `TICKER.PERMNO` form.
        Preserved so callers can round-trip.
    original : str
        The user's input after whitespace strip.
    """

    root: str
    share_class: str | None
    permno: int | None
    original: str

    def render(self, venue: str = "canonical") -> str:
        """Render back to a target venue's convention.

        Parameters
        ----------
        venue : {'canonical', 'yahoo', 'wiki', 'google', 'crsp'}
            - `'canonical'` and `'wiki'`: dot separator (`BRK.B`).
            - `'yahoo'`: hyphen (`BRK-B`).
            - `'google'`: slash (`BRK/B`).
            - `'crsp'`: dot; permno restored if present (`BRK.B` OR
              `BRK.99999` if the parsed form had a permno instead of
              a share class).
        """
        if venue not in _VENUES:
            raise KuantValueError(
                f"kuant.tickernorm.render: 'venue' must be one of "
                f"{_VENUES}, got {venue!r}.  [KE-VAL-RANGE]\n"
                f"  → Fix: pick one of {_VENUES}"
            )
        # If the input carried a permno, the class-share slot is not the
        # class letter but the CRSP tag. Preserve it in crsp venue only.
        if self.permno is not None:
            if venue == "crsp":
                return f"{self.root}.{self.permno}"
            # Non-crsp target: drop the permno tag (it's a data-vendor
            # annotation, not a real venue symbol).
            return self.root
        if self.share_class is None:
            return self.root
        sep = {"yahoo": "-", "google": "/", "canonical": ".", "wiki": ".", "crsp": "."}[venue]
        return f"{self.root}{sep}{self.share_class}"


def tickernorm(
    ticker,
    venue: str | None = None,
) -> TickerParts | str:
    """Parse and optionally render a ticker string.

    Parameters
    ----------
    ticker : str
        Ticker in any supported format (see module docstring).
        Case-insensitive; whitespace trimmed.
    venue : str, optional
        If given, immediately render to the target venue's format and
        return the string. If `None`, return the parsed `TickerParts`
        for the caller to render however they like.

    Returns
    -------
    TickerParts or str
        `TickerParts` if `venue is None`, otherwise the rendered string.

    Raises
    ------
    KuantEncodingError
        If `ticker` is bytes or contains U+FFFD / NUL.
    KuantValueError
        If the input is empty or contains characters other than
        letters, digits, dot, hyphen, slash.

    Examples
    --------
    >>> tickernorm('BRK.B', venue='yahoo')
    'BRK-B'
    >>> tickernorm('BRK-B', venue='canonical')
    'BRK.B'
    >>> tickernorm('BRK/B', venue='crsp')
    'BRK.B'
    >>> # Permno-tagged form: strip the tag when rendering to yahoo.
    >>> tickernorm('ACME.99999', venue='yahoo')
    'ACME'
    >>> # No class share: passthrough with uppercase.
    >>> tickernorm('ibm', venue='yahoo')
    'IBM'
    """
    _check_str_input(ticker, "ticker", kernel="tickernorm")

    trimmed = ticker.strip()
    if not trimmed:
        raise KuantValueError(
            "kuant.tickernorm: input is empty after whitespace strip.  "
            "[KE-VAL-RANGE]\n"
            "  → Fix: pass a non-empty ticker like 'BRK.B' or 'IBM'"
        )

    upper = trimmed.upper()
    # Sanity: only letters, digits, and the three separator characters.
    if not re.fullmatch(r"[A-Z0-9./\-]+", upper):
        raise KuantValueError(
            f"kuant.tickernorm: input {ticker!r} contains characters "
            f"outside `[A-Z0-9./-]`.  [KE-VAL-RANGE]\n"
            f"  → Fix: strip non-alphanumeric chars other than . / - "
            f"before calling"
        )

    # Detect CRSP permno suffix first (a purely numeric trailing block).
    permno_match = _PERMNO_RE.match(upper)
    if permno_match:
        return _finalize(
            root=permno_match.group("root"),
            share_class=None,
            permno=int(permno_match.group("permno")),
            original=trimmed,
            venue=venue,
        )

    # Otherwise look for a share-class separator (., -, /).
    for sep in (".", "-", "/"):
        if sep in upper:
            parts = upper.split(sep)
            if len(parts) == 2 and parts[0] and parts[1]:
                return _finalize(
                    root=parts[0],
                    share_class=parts[1],
                    permno=None,
                    original=trimmed,
                    venue=venue,
                )
            # More than one separator, or empty on either side. Fall
            # through and treat as no-class-suffix, or reject.
            raise KuantValueError(
                f"kuant.tickernorm: input {ticker!r} has multiple or "
                f"empty separator segments; can't disambiguate.  "
                f"[KE-VAL-RANGE]\n"
                f"  → Fix: pass a ticker with at most one dot/hyphen/"
                f"slash separator (or no separator)"
            )

    # No separator, no permno: plain ticker.
    return _finalize(
        root=upper,
        share_class=None,
        permno=None,
        original=trimmed,
        venue=venue,
    )


def _finalize(
    *,
    root: str,
    share_class: str | None,
    permno: int | None,
    original: str,
    venue: str | None,
) -> TickerParts | str:
    parts = TickerParts(
        root=root,
        share_class=share_class,
        permno=permno,
        original=original,
    )
    if venue is None:
        return parts
    return parts.render(venue)


__all__ = ["tickernorm", "TickerParts"]
