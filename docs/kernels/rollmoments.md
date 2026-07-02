# rollskew / rollkurt — Rolling higher moments (3rd and 4th)

## Purpose

`rollskew(x, w)[i]` = sample skewness of `x[i-w+1 : i+1]` (adjusted
Fisher-Pearson, matches pandas).

`rollkurt(x, w)[i]` = sample excess kurtosis of the same window
(Fisher, subtract 3, matches pandas).

Direct use in kuant: tail-risk detection, distribution regime shifts,
gates that fire when skew changes sign or kurtosis exceeds a threshold.

## Public API

```python
from kuant.stats import rollskew, rollkurt

skew = rollskew(x, window)   # requires window >= 3
kurt = rollkurt(x, window)   # requires window >= 4
```

## Design decisions

### 1. Cumsum trick on x, x², x³ (and x⁴ for kurt)

Central moments have closed-form expressions in terms of the raw
window sums. Cumsum computes each raw sum in O(n); the central
moment is a per-window rearrangement of the raw sums.

```math
S_k = sum(x^k) over the window
μ   = S1 / w
m2  = S2/w - μ²
m3  = (S3 - 3μ·S2 + 2w·μ³) / w
m4  = (S4 - 4μ·S3 + 6μ²·S2 - 3w·μ⁴) / w
```

### 2. Shift trick — same idea as `rollstd`

`y = x - x[0]` before the cumsums. Central moments are shift-invariant,
so the answer is unchanged, but working in the small-magnitude y-space
avoids catastrophic cancellation. Critical because m3 and m4 involve
higher powers where cancellation compounds.

### 3. Bias corrections match pandas

```math
skew = √(w(w-1)) / (w-2)      · m3 / m2^{3/2}     (require w >= 3)
kurt = (w-1) / ((w-2)(w-3)) · ((w+1)·(m4/m2²) - 3(w-1))     (require w >= 4)
```

Both are Fisher / adjusted-Fisher-Pearson estimators — the same
default that pandas uses for `.skew()` and `.kurt()`.

### 4. Zero-variance guard

If `m2 == 0` (constant window), skew and kurt are undefined → NaN.
Guard via `xp.where(m2 > 0, ..., NaN)`.

### 5. Shared setup helper

`_rolling_moments_setup(x, w, up_to_order)` computes the cumsums and
central moments up to the requested order, then hands back everything
both `rollskew` and `rollkurt` need. Kurt asks for order 4 and skips
`m3`. Skew asks for order 3 and skips `m4`.

Passing the order lets us skip the `x⁴` cumsum in `rollskew` — cheap
but not free on huge inputs.

## Edge cases

| Condition | Output (both) |
| --- | --- |
| `window < 3` (skew) or `window < 4` (kurt) | all NaN |
| `window > len(x)` | all NaN |
| `window <= 0` | raises `ValueError` |
| `x.ndim != 1` | raises `ValueError` |
| NaN in window | NaN |
| Constant window (m2 == 0) | NaN |

## Cross-check tests

- `test_skew_matches_pandas_uniform`, `test_kurt_matches_pandas_uniform`
  — 500 random samples, atol=1e-10 (skew), atol=1e-9 (kurt)
- `test_matches_pandas_large_magnitude` — price-scale, atol=1e-6
- `test_shift_invariance`, `test_scale_invariance_positive` —
  mathematical properties of central-moment statistics
- `test_skew_negation_flips_sign` — `skew(-x) == -skew(x)`
- Skew signs on hand-picked series (symmetric → 0, right-tail → +, left-tail → -)

## Test coverage (23 tests)

Sign tests (symmetric, left-skewed, right-skewed), pandas reference
(uniform + NaN + price scale), invariance properties (shift, scale,
negation), window bounds (< 3 for skew, < 4 for kurt), zero-variance
handling, dtype preservation, CPU==GPU parity.

## Related kernels

- `kuant.stats.rollmean` — 1st central moment (μ)
- `kuant.stats.rollstd` — square root of 2nd central moment (m2)
- **Future**: `kuant.stats.rollmoment(x, w, order)` — general k-th moment
  if we ever need order 5+
