# tickernorm: Normalize a ticker across venue conventions

## Purpose

The same equity is written differently by every data source:

| Venue / vendor     | BRK class B | BF class B | Plain "IBM" |
| ------------------ | ----------- | ---------- | ----------- |
| Canonical / wiki   | `BRK.B`     | `BF.B`     | `IBM`       |
| Yahoo Finance      | `BRK-B`     | `BF-B`     | `IBM`       |
| Google / some feeds| `BRK/B`     | `BF/B`     | `IBM`       |
| Numeric ID suffix  | `BRK.12345` | `BF.12345` | `IBM.12345` |

Users routinely re-implement this parser. `tickernorm` ships the
canonical mapping and a two-step API: parse to a `TickerParts`
record, then render into the target venue.

## Public API

```python
from kuant.text import tickernorm, TickerParts

parts = tickernorm('BRK.B')                    # -> TickerParts
sym   = tickernorm('BRK.B', venue='yahoo')     # -> 'BRK-B'
sym2  = parts.render(venue='google')           # -> 'BRK/B'
```

- `ticker`: `str`, case-insensitive, whitespace trimmed.
- `venue` (optional):
  - `None` (default) → return `TickerParts`.
  - `'canonical'` or `'wiki'` → dot separator (`BRK.B`).
  - `'yahoo'` → hyphen (`BRK-B`).
  - `'google'` → slash (`BRK/B`).
  - `'crsp'` → dot; numeric ID suffix restored if the parsed form had
    one instead of a share-class letter (`BRK.B` OR `XYZ.12345`).

Raises `KuantEncodingError` on bytes input or U+FFFD / NUL bytes;
`KuantValueError` on empty strings or characters outside
`[A-Z0-9./-]`.

## Design decisions

### 1. Parse first, render second

Every venue conversion is a two-step round trip: input string is
parsed into a `TickerParts` record, then rendered by walking a small
separator lookup table. Users who need multiple output venues from
one input parse once and call `.render(venue=...)` for each. The
`tickernorm(..., venue=...)` shorthand collapses the two calls when
only one output is needed.

### 2. Numeric ID suffix takes precedence over separator sniffing

A trailing `.<digits>` block with 4 to 7 digits is treated as a
data-vendor annotation, not a share-class letter. A precompiled
regex `^(?P<root>.+)\.(?P<numeric_id>\d{4,7})$` catches this pattern
first, before the share-class separator scan. Rationale: share
classes are letters (`A`, `B`, `C`, ...) or short letter groups,
never 4+ digit numbers, so the two forms are unambiguous.

For example, `XYZ.12345` parses as `root='XYZ'`, numeric ID = 12345,
and `XYZ.B` parses as `root='XYZ'`, `share_class='B'`.

### 3. Separator search order: dot, hyphen, slash

If no numeric ID suffix is found, we scan for exactly one of `.`,
`-`, `/` in that order and split on the first hit. A ticker with
zero separators is a plain root (`IBM`). A ticker with more than one
separator or an empty segment on either side raises
`KuantValueError`: the ambiguity is real and forcing the caller to
clean the string is safer than guessing.

### 4. Uppercase everything

`.upper()` runs on the whitespace-trimmed input before any parsing.
`brk.b`, `Brk.B`, and `BRK.B` all parse identically. The `original`
field on `TickerParts` preserves the whitespace-stripped input as
the user provided it so callers can round-trip if needed.

### 5. Numeric ID suffix is dropped on non-CRSP venues

`.render(venue='yahoo')` on a suffix-tagged input returns just the
root: the numeric ID has no meaning to Yahoo Finance. Only
`venue='crsp'` restores it. This keeps output symbols valid at the
target venue rather than propagating a data-vendor annotation into a
market ticker.

### 6. Shared string-input helper

`_check_str_input` is imported from `kuant.text.occparse` and shared
across the subpackage. It rejects `bytes` outright
(`KuantEncodingError`), and emits a `KuantEncodingWarning` on
U+FFFD or NUL bytes (a signal that some upstream decode is broken).
`tickernorm` calls this first before any parsing runs.

## Return shape / dataclass fields

**TickerParts**

| Field | Type | Meaning |
| --- | --- | --- |
| `root` | `str` | Base ticker, uppercased (`'BRK'`, `'IBM'`) |
| `share_class` | `str or None` | Class letter (`'B'` for `BRK.B`) |
| `permno` | `int or None` | Numeric ID suffix if present |
| `original` | `str` | Whitespace-stripped user input |

`.render(venue='canonical')` returns the venue-specific string.

## Examples

```python
>>> from kuant.text import tickernorm
>>> tickernorm('BRK.B', venue='yahoo')
'BRK-B'
>>> tickernorm('BRK-B', venue='canonical')
'BRK.B'
>>> tickernorm('BRK/B', venue='crsp')
'BRK.B'
>>> # Numeric-ID-tagged form: drop the tag on non-CRSP venues.
>>> tickernorm('XYZ.12345', venue='yahoo')
'XYZ'
>>> # Numeric-ID-tagged form: preserve the tag on CRSP.
>>> tickernorm('XYZ.12345', venue='crsp')
'XYZ.12345'
>>> # No share class: passthrough with uppercase.
>>> tickernorm('ibm', venue='yahoo')
'IBM'
>>> # Parse first, then render multiple venues.
>>> parts = tickernorm('BRK.B')
>>> parts.root, parts.share_class
('BRK', 'B')
>>> parts.render('google')
'BRK/B'
```

## Related kernels

- `kuant.text.occparse`: OCC option-symbol parser; the two often
  chain (parse an option symbol, then normalize its underlying).
- `kuant.text.secformparse`: SEC form-type parser; unrelated but
  ships the same `_check_str_input` contract.
- `kuant.text.cusipvalidate`: 9-character CUSIP checksum, for
  identifier-normalization pipelines that feed the same downstream
  stores.
