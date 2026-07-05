# kuant.text: Deterministic parsing of financial text symbols

Not NLP. This subpackage handles the surface-level parsing that runs
BEFORE any model touches text: symbol validation, form-type
classification, and checksum arithmetic. Reference implementations
that get the edge cases right.

## Kernels

- **`occparse`**: OCC option-symbol parser. Turns a 21-character
  OCC-standard string into `(underlying, expiry, right, strike)`.
- **`secformparse`**: SEC form-type classifier. Normalizes forms
  like `10-K`, `10-K/A`, `SC 13D/A` to a canonical form, an
  amendment flag, and a coarse category (`periodic`, `beneficial`,
  `registration`, etc.).
- **`cusipvalidate`**: 9-character CUSIP checksum with normalization
  (uppercase, leading-zero pad, alnum-only). Returns a
  `CUSIPValidation` record with the corrected CUSIP and a boolean
  `is_valid`.
- **`tickernorm`**: Ticker normalization across venue conventions
  (canonical, Yahoo, Google, and CRSP-style numeric ID suffixes).

## Shared error contract

Every kernel here runs its input through `_check_str_input` before
parsing. That helper:

- Rejects `bytes` outright with `KuantEncodingError`. Callers must
  decode first; kuant will not guess an encoding.
- Emits `KuantEncodingWarning` when the string contains U+FFFD
  (replacement character) or NUL bytes. Those are strong signals
  that some upstream decode is broken; the parse continues, but
  the caller is told.

Missing / empty strings raise `KuantValueError`. Structurally
invalid inputs (wrong length, bad characters) raise `KuantValueError`
with a stable code and a one-line fix hint.

## Design theme

Every kernel returns a small `dataclass` rather than a tuple. Field
access is more readable than positional unpacking, and the dataclass
provides a `.summary()` method for quick eyeballing in notebooks.
Every kernel is pure Python; no NumPy backend switch, no GPU path.

## Individual pages

- [`tickernorm.md`](tickernorm.md): venue-aware ticker parser and
  renderer.
- `occparse.md`, `secformparse.md`, `cusipvalidate.md`: coming soon.

## Related subpackages

- `kuant.data`: where parsed symbols typically get consumed
  (universe joins, corporate-action lookups).
