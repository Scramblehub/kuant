# Numerical correctness guarantees for rolling statistics

This page documents what kuant promises about the arithmetic of its
rolling-window kernels, why the promise is worth making, and how a
new kernel joins the guarantee. It is the design record behind
`tests/stats/test_numerical_stability.py`, and behind the v0.3.2 fix
in `kuant.stats.rollemastd`.

The audience is contributors adding a new kernel, and readers auditing
whether kuant is safe to use on inputs whose magnitude is nowhere near
`O(1)`.


## 1. The failure mode we are closing

The formula every rolling-variance implementation starts from is the
algebraic identity

```
Var(x) = E[X²] - E[X]²
```

Applied naively on float64, this identity is a numerical trap. Consider
a window of length `w` on the near-constant series

```
x_i = C + eps · i,  i = 0, 1, ..., w-1
```

with `C = 1e10` and `eps = 1e-6`. The true variance is on the order
of `eps² · w²`, roughly `1e-10`. But `E[X²]` and `E[X]²` are both on the
order of `C² = 1e20`. Subtracting two float64 numbers near `1e20`
produces a result whose absolute error floor is approximately
`1e20 · eps_machine`, where `eps_machine ≈ 2.22e-16`, so about
`2.2e4`. The arithmetic error dominates the true answer by roughly
34 orders of magnitude. The subtraction can, and often does, come out
negative.

The visible failure mode is that the variance goes negative and the
square root produces NaN. The more dangerous mode is invisible: some
libraries defuse the NaN by applying `abs()` before the `sqrt`. That
takes a large negative value produced entirely by cancellation and
returns its square root as a "std." On the input above, that path
returns a std around 150 for a series whose analytic std is around
`1e-6`. Callers who trust `.std()` cannot see the bug from the output.
The number is finite, positive, plausibly-shaped, and completely wrong.

Kuant treats the `abs()`-then-`sqrt()` pattern as a bug, not a fix.
Where the identity is unsafe we do not evaluate it; where FP rounding
can produce a small negative from a genuine zero, we clamp with
`xp.maximum(x, 0)`, which flushes the noise-negative to exactly zero
without turning cancellation into signal.


## 2. The shift trick

Variance is shift-invariant:

```
Var(x) = Var(x - c)  for any constant c
```

If we pick `c ≈ x[0]` (or any other value close to the window mean),
then in `y = x - c` the window values are `O(y)` rather than `O(x)`.
On typical financial inputs `y` is 6 to 10 orders of magnitude smaller
than `x`. The intermediate quantities in `E[Y²] - E[Y]²` shrink by
`|x/y|²`, so the absolute cancellation error shrinks by the same
factor. We recover a variance that is correct to a small multiple of
`eps_machine · Var(x)`.

The diff pattern kuant uses is the same in every kernel that needs
it. From `kuant/stats/rollstd.py`:

```python
shift_val = float(x_safe[0])
if not np.isfinite(shift_val):
    shift_val = 0.0
shift = xp.asarray(shift_val, dtype=out_dtype)
y = x_safe - shift

csum_y    = xp.concatenate([zero_pad, xp.cumsum(y)])
csum_y_sq = xp.concatenate([zero_pad, xp.cumsum(y * y)])

window_sum_y    = csum_y[w:]    - csum_y[:-w]
window_sum_y_sq = csum_y_sq[w:] - csum_y_sq[:-w]

ssq = window_sum_y_sq - window_sum_y * window_sum_y / w
ssq = xp.maximum(ssq, zero_scalar)   # noise-floor clamp, not abs()
std = xp.sqrt(ssq / (w - ddof))
```

The v0.3.2 fix in `kuant/stats/rollemastd.py` is the same pattern
applied to the EMA recurrence:

```python
shift_val = float(arr_np[0])
if not np.isfinite(shift_val):
    shift_val = 0.0
y = arr_np - shift_val

m1 = _ema_via_lfilter(y,     alpha_val)
m2 = _ema_via_lfilter(y * y, alpha_val)

var_biased = m2 - m1 * m1
var_biased = np.maximum(var_biased, 0.0)  # not abs()
```

Two things worth calling out:

1. **Fallback for non-finite `x[0]`.** If the first sample is NaN
   or inf, we shift by 0 instead. The trick becomes suboptimal but
   stays correct. It never propagates a non-finite shift.
2. **The clamp is `maximum`, not `abs`.** A near-zero true variance
   can round negative by a few ulps. Clamping to zero preserves the
   answer. `abs()` would turn a `-1e-14` produced by rounding into
   a `+1e-14`, which is fine here but conceals `-1e18` cancellation
   elsewhere. We reserve `maximum(x, 0)` for the noise floor and
   consider `abs()` on a variance intermediate to be a defect.


## 3. What kuant currently guarantees

The suite in `tests/stats/test_numerical_stability.py` is the executable
form of these guarantees. Each row below reflects at least one test in
that file; empty cells indicate no test targets that combination
(usually because the failure mode does not apply to that kernel).

| Kernel | Near-constant series | Large offset | Long drift | Alternating signs | Constant series boundary |
|---|---|---|---|---|---|
| `rollstd` | passes | passes | finite and positive | n/a | exactly zero |
| `rollmean` | passes | passes | n/a | passes | n/a |
| `rollsum` | passes | n/a | n/a | passes | n/a |
| `rollskew` | NaN or zero, never inf | passes | n/a | n/a | NaN or zero |
| `rollkurt` | NaN or zero, never inf | passes | n/a | n/a | NaN or zero |
| `rollema` | n/a | passes | n/a | n/a | n/a |
| `rollemastd` | passes (v0.3.2+) | n/a | n/a | n/a | n/a |
| `zscore` | n/a | passes | n/a | n/a | n/a |
| `sharperatio` | n/a | passes | n/a | n/a | n/a |
| `sortinoratio` | passes (finite result on zero downside) | n/a | n/a | n/a | n/a |

"Near-constant series" refers to `[C, C + eps, C + 2·eps, ...]` with
`C` at least `1e10` and `eps` at most `1e-6`: the classic cancellation
input. "Large offset" refers to `C + noise` with `C` at least `1e9` and
`noise` at unit scale: the classic zscore/mean-normalized input.
"Long drift" refers to a series whose magnitude spans five orders
within the observation window.

Additional pandas-parity guardrails: `rollstd` and `rollmean` agree
with `pandas.Series.rolling(...).std()` and `.mean()` on the same
stress inputs to within `1e-4` and `1e-10` relative respectively.
This is not a comparative claim about which library is "better", it
is a compatibility guarantee: if you swap kuant for the pandas rolling
kernel on a stability-sensitive input, the answer does not change.


## 4. What kuant does NOT guarantee

The shift-by-`x[0]` trick has a residual failure mode when `x[0]` is
not close to the values inside the current window. On a 25-year price
series with a two-order-of-magnitude drift, a rolling window near the
end of the series sees `y` values that are themselves large. The
cancellation error is smaller than it would be without the shift, but
it is not near machine epsilon.

Concretely: the test `test_rollstd_series_drifting_five_orders_of_magnitude`
constructs a series with a geometric drift from `1` to `1e5` over 5000
points and checks only that the output is finite and positive. It does
not require agreement with a Welford or Kahan-compensated reference at
machine precision. A caller who needs that guarantee on a long
drifting series should either normalize the input first (log-transform,
detrend) or wait for an opt-in true-Welford path (see section 5).

Second gap: **mixed-backend float32.** The float64 guarantees above do
not carry to float32 inputs, and they do not survive silent up- or
down-casting between numpy and cupy code paths. Dtype rounding at
`eps_machine ≈ 1.2e-7` for float32 dominates the cancellation error
we are otherwise controlling. Kuant preserves input dtype but does
not promise numerical stability on float32; callers who care should
cast to float64 before entering the kernel.

Third gap: **the sortino divide-by-zero on true zero downside.** The
kernel returns a finite value or an infinity for a series with no
downside, and this is intentional and documented as such. It is not
a numerical bug. If your caller cannot handle a non-finite Sortino,
guard it upstream rather than expecting the kernel to hide the
degenerate case.


## 5. How to write a new kernel that meets these guarantees

The checklist for a new rolling-window kernel:

1. **If the kernel computes a variance, a moment, or any centered
   sum of squares, subtract a stable reference value before the
   accumulation.** Use `x[0]` if you have no better estimate. Cast
   the shift through `float` and guard the non-finite case with a
   fallback to zero. Do the accumulation in `y = x - shift` space.
   Read the variance in `y` space. It equals the variance in `x`
   space by shift-invariance.

2. **Guard the floating-point-negative case with a clamp, never with
   `abs()`.** Use `xp.maximum(var, 0.0)`. If you see `abs()` on a
   variance intermediate anywhere in a PR, treat it as a review
   blocker; it hides the failure mode this page exists to close.

3. **Add a test in `tests/stats/test_numerical_stability.py` for
   each failure mode that applies to the kernel.** The template
   inputs are the ones in section 3: near-constant with a huge `C`,
   large offset plus unit-scale noise, and (if the kernel accumulates
   in a cumsum) alternating large-magnitude signs. The tolerance
   should be set to what a naive implementation would blow past,
   not to machine epsilon. The test is a canary, not a precision
   benchmark.

4. **If the kernel genuinely needs Welford or Kahan compensation
   rather than shift-plus-cumsum, expose it as an opt-in flag and
   document the cost.** Welford is `O(n · w)` in the strict rolling
   case, versus `O(n)` for the cumsum-with-shift path. That is a
   substantial regression on wide panels. The path is welcome, but
   it does not become the default without a design-page justification
   for why the shift trick is inadequate on the specific input class
   the kernel targets.

5. **Do not add a pandas-parity test that is tighter than pandas'
   own numerical guarantee.** The `rollstd`-vs-pandas tolerance of
   `1e-4` relative on the C = 1e9 input is not arbitrary; it is
   what pandas achieves. A test that requires a tighter bound will
   fail intermittently as pandas versions move.

A kernel that passes these five checks slots into the table in
section 3 with a new row, and the table becomes the new contract.
