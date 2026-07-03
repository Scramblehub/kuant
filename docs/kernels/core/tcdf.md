# tcdf — Student-t cumulative distribution function

## Purpose

CDF of the Student-t distribution:

```math
F(x; \nu) = \Pr(T \le x)
```

Computed via the regularized incomplete beta identity:

```math
F(x; \nu) = \begin{cases}
  1 - \tfrac{1}{2} \, I_{z}\!\left(\tfrac{\nu}{2}, \tfrac{1}{2}\right) & x > 0 \\
  \tfrac{1}{2} \, I_{z}\!\left(\tfrac{\nu}{2}, \tfrac{1}{2}\right) & x \le 0
\end{cases}
```

where `z = ν / (ν + x²)` in `[0, 1]`.

## Public API

```python
from kuant.core import tcdf

p = tcdf(x, df)
```

## Design decisions

### Incomplete-beta route

Using `betainc` rather than integrating tpdf: closed-form-ish and
already well-implemented in scipy / cupyx.scipy.

### Backend bridge

`betainc` routed via `_special_bridge`:
- numpy → `scipy.special.betainc`
- cupy → `cupyx.scipy.special.betainc` if available, else H↔D fallback

## Related

- `tpdf`, `tppf`
- `scipy.stats.t.cdf`
