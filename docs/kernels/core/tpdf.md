# tpdf — Student-t probability density function

## Purpose

PDF of the Student-t distribution:

```math
f(x; \nu) = \frac{\Gamma\!\left(\frac{\nu+1}{2}\right)}
                 {\sqrt{\nu \pi} \, \Gamma\!\left(\frac{\nu}{2}\right)}
            \left(1 + \frac{x^2}{\nu}\right)^{-\frac{\nu+1}{2}}
```

Fat-tail cousin of the Gaussian. `ν` = degrees of freedom controls
tail heaviness: `ν → 1` is Cauchy (heavy), `ν → ∞` is Gaussian.

## Public API

```python
from kuant.core import tpdf

f = tpdf(x, df)
```

- `x` — scalar or array.
- `df` — scalar or array of degrees of freedom (> 0). Broadcasts with x.

## Design decisions

### Log-space evaluation

Direct evaluation of `Γ((ν+1)/2)` overflows at large ν. We compute:

```
log f = gammaln((ν+1)/2) - gammaln(ν/2) - 0.5·(log ν + log π)
        - (ν+1)/2 · log1p(x²/ν)
```

Then `exp` once. `log1p` handles `x²/ν → 0` precisely.

### Backend bridge

`gammaln` routed via `kuant.core._special_bridge`:
- numpy input → `scipy.special.gammaln`
- cupy input → `cupyx.scipy.special.gammaln` if available, else
  H↔D fallback via `.get()` and `cp.asarray()`.

### Extreme-df limit

At `df > 1e5`, tpdf converges to normpdf. Our gammaln-based path
matches scipy's alternate to ~1e-10 in this regime (both are Gaussian-
limit approximations of the same formula).

## Related

- `kuant.core.tcdf` — CDF
- `kuant.core.tppf` — inverse CDF
- `scipy.stats.t.pdf` — reference implementation
