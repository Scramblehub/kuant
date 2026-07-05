"""kuant.text - deterministic parsing of financial text symbols.

Not NLP: this subpackage handles the surface-level parsing you do BEFORE
any model touches text. Symbol validation, form classification, and
CUSIP-style checksum arithmetic. Reference implementations that get the
edge cases right.

Kernels:

- `occparse`: OCC option symbol -> (underlying, expiry, right, strike).
- `secformparse`: SEC form type -> normalized form + amendment flag + category.
- `cusipvalidate`: 9-character CUSIP checksum + normalization.

Every kernel here checks its input with a shared string-input helper,
so bytes-where-str-expected raises `KuantEncodingError` and inputs with
U+FFFD or NUL bytes emit `KuantEncodingWarning` (broken-upstream-decode
signal).
"""

from kuant.text.cusipvalidate import CUSIPValidation, cusipvalidate
from kuant.text.occparse import OCCSymbol, occparse
from kuant.text.secformparse import SECForm, secformparse
from kuant.text.tickernorm import TickerParts, tickernorm

__all__ = [
    "CUSIPValidation",
    "OCCSymbol",
    "SECForm",
    "TickerParts",
    "cusipvalidate",
    "occparse",
    "secformparse",
    "tickernorm",
]
