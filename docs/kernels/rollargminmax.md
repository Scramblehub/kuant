# rollargmin / rollargmax — Position of window extreme

## Purpose

`rollargmax(x, w)[i]` = index within window `x[i-w+1 : i+1]` where the
maximum occurs. `rollargmin` is the analogous min. Return values are
in `[0, w-1]` where `0` is the oldest bar in the window and `w-1` is
the newest (i.e., `x[i]`).

Direct use: "days since max" style signals, `abs(x - x_at_max) /
sigma` decay signals, or absolute indexing via `i - (w-1) +
rollargmax[i]`.

## Public API

```python
from kuant.stats import rollargmax, rollargmin

pos = rollargmax(x, window)
```

Returns float array (to allow NaN); integer values in `[0, w-1]`
otherwise.

## Design decisions

### 1. Same sliding-view pattern as rollminmax

`sliding_window_view(x, w)` + `xp.argmax` (or `argmin`) along axis 1.
Zero-copy on numpy; may materialize on cupy.

### 2. Return float, not int

numpy's `argmax` returns `int64`, but we need to allow NaN for the
"partial window" and "NaN-containing window" cases. Casting to float
throughout keeps the mask semantics uniform with the rest of
`kuant.stats`.

### 3. Explicit NaN row mask required

`argmax` of an all-NaN row silently returns 0 (not NaN), so a naive
approach would report a bogus position. We build an
`xp.any(xp.isnan(window), axis=1)` mask and set those rows to NaN.

### 4. Convention on ties: first-occurrence (numpy default)

numpy's argmax returns the FIRST index where the max value appears.
kuant inherits this. Different tie-breaking (e.g. last-occurrence)
would need a separate kernel or a `keepdim` flag; not implemented in
V1.

## Edge cases

| Condition | Output |
|---|---|
| `window == 1` | all 0 (single-element window's argmax is 0) |
| `window > len(x)` | all NaN |
| `window <= 0` | raises `ValueError` |
| `x.ndim != 1` | raises `ValueError` |
| NaN in window | NaN |
| All values tied | 0 (first occurrence) |

## Cross-check tests

- Hand-computed argmin/argmax on a small example
- Ascending/descending inputs give predictable positions
- Tie-first convention verified
- Range check: `0 <= result <= w-1` for finite outputs
- Symmetry: `argmax(-x) == argmin(x)`

## Test coverage (14 tests)

Golden values, ascending/descending baselines, tie handling, NaN
propagation, edge cases (window bounds, 2D input, dtype), property
tests (bounds, negation symmetry), CPU==GPU parity.

## Direct usage in kuant

- "Bars since max" = `(w - 1) - rollargmax(x, w)` — how long ago the
  window's high was set
- Adaptive stop distance = time-decay applied to
  `(w - 1) - rollargmax(spot, w)`
- Signal decay: fade a signal that was set at the argmax of a fresh
  window

## Related kernels

- `kuant.stats.rollminmax` — value at the extreme
- `kuant.stats.rollrank` — rank of current value (not position of
  extreme, but same sliding-view family)
