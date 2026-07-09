# crossrecurrence, jointrecurrence: cross- and joint-recurrence quantification

## Purpose

Two coupling diagnostics that extend single-series RQA to a pair `(x, y)`:

- `crossrecurrence(x, y, ...)` marks `(i, j)` where the delay-embedded
  state of `x` at time `i` sits within distance `epsilon` of the
  delay-embedded state of `y` at time `j`. Diagonal structure in the CRP
  reveals synchronization; alignment above or below the main diagonal
  reveals lagged coupling.
- `jointrecurrence(x, y, ...)` marks `(i, j)` where BOTH series are
  recurrent within their own state spaces at the same pair of times.
  It is the intersection of two same-series recurrence plots and picks
  out simultaneous recurrence events, a coupling signature distinct
  from cross-recurrence (Marwan-Romano-Thiel-Kurths 2007).

Both return the standard RQA measures (recurrence rate, determinism,
laminarity, longest diagonal, entropy of diagonals) computed over the
appropriate recurrence matrix.

## Public API

```python
from kuant.sindy.chaos import crossrecurrence, jointrecurrence

cr = crossrecurrence(x, y, tau=1, m=3, recurrence_rate_target=0.10)
print(cr.summary())
cr.determinism           # fraction of recurrent points on diagonals
cr.longest_diagonal      # length of the longest diagonal line

jr = jointrecurrence(x, y, tau=1, m=3, recurrence_rate_target=0.10)
jr.recurrence_rate       # ~ RR_x * RR_y for independent x, y
```

Signatures:

```python
crossrecurrence(
    x, y, *, tau=1, m=5,
    epsilon=None, recurrence_rate_target=0.10, l_min=2,
)

jointrecurrence(
    x, y, *, tau=1, m=5,
    epsilon_x=None, epsilon_y=None,
    recurrence_rate_target=0.10, l_min=2,
)
```

Returns `CrossRecurrenceResult` or `JointRecurrenceResult` with
`recurrence_rate`, `determinism`, `laminarity`, `longest_diagonal`,
`entropy_diagonal`, `l_min`, `embed_dim`, `embed_tau`, plus the
thresholds actually used (`epsilon` for CRP, `epsilon_x` / `epsilon_y`
for JRP).

## Design decisions

### Diagonal scan covers both sub- and super-diagonals

The cross-recurrence matrix `R[i, j] = 1 iff dist(x_i, y_j) <= epsilon`
is asymmetric: line structure lives on both sides of the main diagonal.
Walking only `k >= 0` (upper triangle) would halve determinism, longest
diagonal, and diagonal entropy for a CRP.

The internal `_diagonal_lengths` helper walks every offset
`k in [-(n-1), n-1]`, keeps run-length statistics per diagonal, and
returns the combined list. For the symmetric joint-recurrence matrix
the two triangles mirror each other, but the same scan is correct
there: the caller's `exclude_loi` flag drops only `k == 0` (the line
of identity), which is the piece that differs between CRP and JRP.

### `epsilon` auto-picked from a quantile

If `epsilon` is None, `crossrecurrence` sets it to the
`recurrence_rate_target` quantile of the full cross-distance matrix,
which pins the achieved recurrence rate near the requested target
(within about a percentage point on 500 points, see
`test_recurrence_rate_hits_target`).

`jointrecurrence` picks `epsilon_x` and `epsilon_y` independently from
the off-diagonal upper-triangle distances of each own-series RP, so
each marginal RP hits `recurrence_rate_target`. For independent inputs
the joint rate is then approximately `target * target`, giving the
product-rule sanity check exercised in
`test_independent_series_product_rule`.

### No line-of-identity exclusion for CRP

Cross-recurrence between two DIFFERENT series has no built-in main
diagonal to remove: `x_i` matching `y_i` is a real coincidence, not a
tautology. `_rqa_from_matrix` is called with `exclude_loi=False`, and
the denominator for `recurrence_rate` is the full `n * n`.

For JRP the diagonal is removed (`np.fill_diagonal(r_x, 0)` and
`r_y`), and the RQA denominator becomes `n * n - n`. This matches
Marwan 2007's convention for jointly-defined RPs.

### 2000-sample cap

The distance-matrix construction is O(N^2 * m). Above 2000 aligned
finite samples the trailing 2000 are kept. Small-attractor RQA
statistics are stable at that length and memory stays under a few
hundred MB.

### 100-sample floor

Below 100 paired finite values the RQA statistics are dominated by
noise. Enforced with `KE-VAL-MIN-CLEAN`.

## Error codes

- `KE-SHAPE-EQUAL-LEN`: `x` and `y` differ in length.
- `KE-VAL-MIN-CLEAN`: fewer than 100 paired finite values.
- Standard range / positivity errors from `_validation` for `tau`,
  `m` (bounded to `[2, 50]`), `recurrence_rate_target` (bounded to
  `[0.01, 0.9]`), and `l_min`.

## Edge cases

| Condition | Behavior |
| --- | --- |
| `x is y` | CRP recurrence rate is very high (self-matches). |
| Independent `x`, `y`, `target = 0.10` | JRP `recurrence_rate ~ 0.01`. |
| Constant `x` or `y` | epsilon collapses to 0; RR degenerates. |
| Fewer than 100 finite pairs | `KE-VAL-MIN-CLEAN`. |
| Length mismatch | `KE-SHAPE-EQUAL-LEN`. |
| `m == 1`, `recurrence_rate_target > 0.9` | range error from validation. |

## When it fires

- Cross-market synchronization: run `crossrecurrence` on two return
  series and read `determinism`; sustained diagonal structure is the
  signature of common driving.
- Coupling asymmetry: pair CRP with `transferentropy` for direction,
  since CRP alone is symmetric under swapping x and y.
- Joint regime detection: JRP recurrence rate spikes when two series
  simultaneously revisit their own recent states, a proxy for "both
  markets are re-entering a familiar regime at once."

## Cross-check tests

`tests/sindy/chaos/test_crossrecurrence.py`:

- `test_recurrence_rate_hits_target`: auto-`epsilon` places RR within
  0.01 of the target.
- `test_identical_series_high_rr`: CRP on `(x, x)` gives high recurrence.
- `test_independent_series_product_rule`: JRP with target 0.10 on
  independent inputs sits in `(0.001, 0.04)`.
- `test_identical_series_matches_own_rqa`: JRP on `(x, x)` reduces to
  the single-series RP with RR near the marginal target.
- Rejection tests for unequal length (`KE-SHAPE-EQUAL-LEN`) and
  under-length inputs (`KE-VAL-MIN-CLEAN`).

## Related kernels

- `kuant.sindy.chaos.rqa`: single-series RQA. CRP and JRP reuse the
  same run-length and entropy machinery.
- `kuant.sindy.chaos.embedding._embed`: delay embedding shared with
  all chaos kernels.
- `kuant.sindy.chaos.ccm`, `transferentropy`: directional coupling
  complements to the symmetric CRP / JRP diagnostics.

## References

- Marwan, Romano, Thiel & Kurths 2007, "Recurrence plots for the
  analysis of complex systems," Physics Reports 438.
