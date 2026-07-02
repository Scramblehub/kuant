# kuant.core kernels

Foundation numerical kernels — the primitives everything else in kuant
composes on. All kernels:

- Preserve input backend (numpy in → numpy out; cupy in → cupy out)
- Preserve input dtype (float32 in → float32 out; ints promote to float64)
- Preserve input shape (broadcasting, scalar-in/scalar-out)
- Propagate NaN cleanly
- Have CPU and GPU implementations verified for parity

## Naming convention

**`bs` prefix** — Black-Scholes family (European options on dividend-paying
underlying).

**Direction in the name** — only where the *math* differs for calls vs
puts. That is: for **delta** and **rho** (opposite signs, different `Φ`
arguments), we ship separate `bsput...` and `bscall...` kernels. For
**gamma** and **vega** (put-call symmetric — same value for call and put
with identical inputs), we ship one kernel and drop the direction from the
name.

**`norm` prefix** — standard normal primitives (CDF, PDF). Foundation used
by the whole BS family.

## Kernels shipped

### Foundation

| Kernel | Formula | Doc |
| --- | --- | --- |
| [`normcdf`](normcdf.md) | `Φ(x) = P[Z ≤ x]`, Z ~ N(0,1) | 28 tests |
| [`normpdf`](normpdf.md) | `φ(x) = exp(-x²/2) / √(2π)` | 14 tests |

### Prices

| Kernel | Formula | Doc |
| --- | --- | --- |
| [`bsput`](bsput.md) | `K·e^(-r·T)·Φ(-d2) - S·e^(-q·T)·Φ(-d1)` | 23 tests |
| [`bscall`](bscall.md) | `S·e^(-q·T)·Φ(d1)  - K·e^(-r·T)·Φ(d2)` | 24 tests |

Related by put-call parity: `C - P = S·e^(-q·T) - K·e^(-r·T)`. Verified to
machine epsilon in `test_put_call_parity_random`.

### First-order Greeks — direction-specific

| Kernel | Formula | Range | Doc |
| --- | --- | --- | --- |
| [`bsputdelta`](bsputdelta.md) | `-e^(-q·T)·Φ(-d1)` | `[-1, 0]` | 24 tests |
| [`bscalldelta`](bscalldelta.md) | `+e^(-q·T)·Φ(d1)` | `[0, 1]` | 21 tests |
| [`bsputrho`](bsputrho.md) | `-T·K·e^(-r·T)·Φ(-d2)` | `(-∞, 0]` | 18 tests |
| [`bscallrho`](bscallrho.md) | `+T·K·e^(-r·T)·Φ(d2)` | `[0, +∞)` | 19 tests |

Parity identities (both verified to machine epsilon):

- `delta_call - delta_put = e^(-q·T)`
- `rho_call - rho_put = T·K·e^(-r·T)`

### Second-order Greeks — put-call symmetric (one kernel serves both)

| Kernel | Formula | Range | Doc |
| --- | --- | --- | --- |
| [`bsgamma`](bsgamma.md) | `e^(-q·T)·φ(d1) / (S·σ·√T)` | `[0, +∞)` | 16 tests |
| [`bsvega`](bsvega.md) | `S·e^(-q·T)·φ(d1)·√T` | `[0, +∞)` | 15 tests |

## Shared setup — `_bs_common.prepare_bs`

Every BS kernel shares the same 30 lines of boilerplate: backend detection,
broadcasting, dtype policy (with the subtle fix that `q`'s Python-scalar
default doesn't promote everything to float64), NaN-init output for free
NaN propagation, safe-value substitution in edge cells, and `d1/d2/normal`
mask precomputation.

That boilerplate lives in `kuant/core/_bs_common.py::prepare_bs()`. Each
kernel becomes ~20 lines of formula on top:

```python
c = prepare_bs(S, K, T, r, sigma, q)
formula = <one-liner using c.d1, c.d2, c.S_safe, ...>
out = c.xp.where(c.normal, formula, c.out)
# kernel-specific edge cases
return finalize(out)
```

## Cross-check tests

Each kernel is validated three ways:

1. **Golden values** — hand-picked scipy-derived reference points, pasted
   into `pytest.parametrize`. Catches typos in the formula.
2. **1000-point random reference match** — 1000 random `(S,K,T,r,σ,q)`
   tuples matched against an independent scipy-based implementation.
3. **Cross-kernel identities** — put-call parity for price/delta/rho;
   finite-difference cross-checks between price/delta/gamma/vega/rho form
   a closed derivative chain. Drift anywhere surfaces here.

Total across the family: **207 tests, all passing**.

## Related

- Setup helper: `kuant/core/_bs_common.py`
- Hardware throttle used by `normcdf`: `kuant/queueing/`
- Tests: `tests/core/test_*.py` — 1:1 with kernel files
