"""Parse OCC option symbols into their four components.

An OCC symbol packs four fields into a single string:

    UNDERLYING + YYMMDD + [C|P] + STRIKE_x_1000_zero_padded_to_8

For example `AAPL240119C00150000` decodes to
`(underlying='AAPL', expiry=date(2024, 1, 19), right='C', strike=150.0)`.

Two formats are common in vendor feeds:

- **Compact** (recommended): `AAPL240119C00150000`. The root is 1-6
  uppercase alnum chars; no padding.
- **Fixed 21-char**: `AAPL  240119C00150000`. Root is space-padded to
  6 characters. Accepted; padding is stripped.

Design: docs/kernels/text/occparse.md.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from kuant.errors import KuantEncodingError, KuantValueError

# Compact form: root of 1-6 uppercase alnum (plus '.' for share class),
# then YYMMDD, C or P, then 8-digit strike.
_OCC_RE = re.compile(
    r"^"
    r"(?P<root>[A-Z][A-Z0-9.]{0,5})"  # underlying
    r"(?P<yy>\d{2})"
    r"(?P<mm>\d{2})"
    r"(?P<dd>\d{2})"
    r"(?P<right>[CP])"
    r"(?P<strike>\d{8})"
    r"$"
)


@dataclass
class OCCSymbol:
    """Parsed OCC option symbol.

    Attributes
    ----------
    underlying : str
        The root symbol (e.g. `'AAPL'`, `'BRK.B'`).
    expiry : datetime.date
        Contract expiration.
    right : str
        `'C'` for call, `'P'` for put.
    strike : float
        Strike price. The raw OCC field is strike-x-1000 zero-padded;
        this attribute has already divided out the 1000.
    original : str
        The exact string the user passed in (whitespace-stripped).
    """

    underlying: str
    expiry: date
    right: str
    strike: float
    original: str

    @property
    def is_call(self) -> bool:
        return self.right == "C"

    @property
    def is_put(self) -> bool:
        return self.right == "P"

    def summary(self) -> str:
        parts = [
            "=== OCCSymbol ===",
            f"underlying:  {self.underlying}",
            f"expiry:      {self.expiry.isoformat()}",
            f"right:       {self.right} ({'call' if self.is_call else 'put'})",
            f"strike:      {self.strike:g}",
        ]
        return "\n".join(parts)


def occparse(symbol) -> OCCSymbol:
    """Parse an OCC option symbol into its four components.

    Parameters
    ----------
    symbol : str
        OCC symbol. Compact (`'AAPL240119C00150000'`) or fixed-21-char
        (`'AAPL  240119C00150000'`). Leading and trailing whitespace
        is stripped. Case is normalized to upper.

    Returns
    -------
    OCCSymbol
        Parsed fields plus `.original` (the input after whitespace
        strip), `.is_call`/`.is_put`, `.summary()`.

    Raises
    ------
    KuantEncodingError
        If `symbol` is bytes instead of str, or contains a Unicode
        `REPLACEMENT CHARACTER` (U+FFFD) or NUL byte indicating a
        broken upstream decode.
    KuantValueError
        If the string does not match the OCC layout, the date is not
        a real calendar day, or the strike is zero.

    Examples
    --------
    >>> from datetime import date
    >>> r = occparse('AAPL240119C00150000')
    >>> r.underlying, r.expiry, r.right, r.strike
    ('AAPL', datetime.date(2024, 1, 19), 'C', 150.0)
    >>> occparse('SPY   240315P00450500').strike
    450.5
    """
    _check_str_input(symbol, "symbol", kernel="occparse")

    # Compact by removing internal spaces (handles the 21-char padded form).
    normalized = symbol.strip().replace(" ", "").upper()

    match = _OCC_RE.match(normalized)
    if not match:
        raise KuantValueError(
            f"kuant.occparse: input {symbol!r} does not match the OCC "
            f"layout `ROOT+YYMMDD+[C|P]+STRIKE(8)`.  [KE-VAL-RANGE]\n"
            f"  → Fix: expected e.g. 'AAPL240119C00150000'; check for "
            f"missing digits, wrong letter (must be C or P), or a strike "
            f"not padded to 8 digits"
        )

    yy = int(match.group("yy"))
    mm = int(match.group("mm"))
    dd = int(match.group("dd"))
    # OCC's 2-digit year: dates are strictly in the range [2000, 2099]. This
    # is safe until 2100 when the standard will presumably move to 4-digit
    # years. Alert if needed then.
    year = 2000 + yy
    try:
        expiry = date(year, mm, dd)
    except ValueError as exc:
        raise KuantValueError(
            f"kuant.occparse: expiry date {year:04d}-{mm:02d}-{dd:02d} is "
            f"not a real calendar day.  [KE-VAL-RANGE]\n"
            f"  → Fix: check the YYMMDD segment; e.g. February 30th is "
            f"a common upstream typo"
        ) from exc

    strike_int = int(match.group("strike"))
    if strike_int == 0:
        raise KuantValueError(
            "kuant.occparse: strike is zero, which is not a valid OCC "
            "strike.  [KE-VAL-POSITIVE]\n"
            "  → Fix: verify the strike field; it should be positive "
            "and encoded as strike-x-1000 zero-padded to 8 digits"
        )
    strike = strike_int / 1000.0

    return OCCSymbol(
        underlying=match.group("root"),
        expiry=expiry,
        right=match.group("right"),
        strike=strike,
        original=symbol.strip(),
    )


def _check_str_input(value, name: str, *, kernel: str) -> None:
    """Reject bytes / non-str; warn on replacement chars and NUL."""
    if isinstance(value, bytes):
        raise KuantEncodingError(
            f"kuant.{kernel}: '{name}' was passed as bytes; text kernels "
            f"expect str.  [KE-ENCODING-BYTES]\n"
            f"  → Fix: decode with an explicit encoding first, e.g. "
            f"`{name}.decode('utf-8')`"
        )
    if not isinstance(value, str):
        raise KuantValueError(
            f"kuant.{kernel}: '{name}' must be a str, got "
            f"{type(value).__name__}.  [KE-VAL-RANGE]\n"
            f"  → Fix: pass a Python str"
        )
    if "�" in value or "\x00" in value:
        # Import lazily to avoid a circular reference between errors and
        # _validation on module load.
        from kuant._validation import warn_kuant
        from kuant.errors import KuantEncodingWarning

        n_repl = value.count("�")
        n_nul = value.count("\x00")
        detail = []
        if n_repl:
            detail.append(f"{n_repl} REPLACEMENT CHARACTER(s) (U+FFFD)")
        if n_nul:
            detail.append(f"{n_nul} NUL byte(s)")
        warn_kuant(
            kernel=kernel,
            code="KW-ENCODING-REPLACEMENT",
            what=(
                f"'{name}' contains " + " and ".join(detail) + "; upstream "
                "decode may have silently failed"
            ),
            fix=(
                "re-decode the source bytes with the correct encoding "
                "(vendor feeds sometimes ship latin-1 or cp1252, not UTF-8)"
            ),
            category=KuantEncodingWarning,
        )


__all__ = ["occparse", "OCCSymbol", "_check_str_input"]
