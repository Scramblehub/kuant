# mintorsion: Principal-components torsion

## Purpose

Compute an orthonormal torsion matrix `T` such that `f = T r` produces
uncorrelated factors (`T Sigma T'` is diagonal), then report the
Meucci 2009 effective number of bets on a supplied portfolio.

This is the principal-components torsion (PCT) surrogate for the full
Meucci-Santangelo-Deguest 2013 minimum-torsion decomposition. PCT
matches the exact min-torsion answer when correlation is dominated by
a single principal axis, and is the standard cheap approximation
when only diversification content is needed. The iterative exact
solver from Meucci 2013 Section 2.5 is tracked as a follow-up.

## Public API

```python
from kuant.portfolio import mintorsion

r = mintorsion(cov)                        # equal-weight effective bets
r = mintorsion(cov, weights=w)             # portfolio-specific
r.torsion          # (n, n) orthonormal, diag(V) >= 0 sign convention
r.factor_cov       # T Sigma T', diagonal up to FP noise
r.effective_bets   # scalar in (0, n]
print(r.summary())
```

- `cov` — 2D (n, n) covariance.
- `weights` — optional (n,) portfolio weights. Equal weights if
  omitted.

## Design decisions

### 1. Symmetric eigendecomposition, descending order

`np.linalg.eigh(Sigma)` returns eigenvalues in ascending order.
Reverse so the largest-variance factor is first, matching the
standard PCA convention. Eigenvalues clipped to `1e-14` from below
to guard against negative FP dust for near-singular `Sigma`.

### 2. "Diag positive" sign convention

Eigenvectors are only defined up to sign. We orient each column so
that the eigenvector entry with the largest absolute value is
positive. This picks the sign that keeps each factor as close as
possible to the identity mapping: a factor associated with a
particular asset points the same way as that asset. This makes the
factor interpretation stable across seeds and small `Sigma`
perturbations.

### 3. Effective number of bets (Meucci 2009)

Portfolio exposure in factor space is `p = T w`. Factor variance
contributions are `p_k^2 * var(f_k)`; normalize to `p_norm` summing
to 1. The effective number of bets is the exponential of the entropy
of that distribution:

```math
N_eff = exp(-sum_k p_norm_k * log p_norm_k)
```

Bounded in `(0, n]`. Uniform contributions give `N_eff = n`; a
one-factor portfolio gives `N_eff = 1`.

### 4. PCT vs true min-torsion

PCT solves for uncorrelated factors but does NOT solve the Meucci
2013 min-Frobenius-distance-to-identity criterion. On highly
non-PCA-like `Sigma` the true min-torsion produces different (and
better) factors. PCT is the accepted approximation for diversification
diagnostics; use the min-torsion iterative solver when the exact
Frobenius-minimizing `T` is required.

## Edge cases / errors

| Condition | Behavior |
| --- | --- |
| Non-square `cov` | `KuantShapeError [KE-SHAPE-2D]` |
| `weights.size != n` | `KuantShapeError [KE-SHAPE-EQUAL-LEN]` |
| Zero total variance contribution | `effective_bets = NaN` |
| Rank-deficient `Sigma` | eigenvalues clipped to `1e-14`, torsion still returned |

## Cross-check tests

- `test_returns_result_with_torsion` — factor covariance off-diagonal
  norm no worse than raw covariance.
- `test_effective_bets_bounded` — `0 < N_eff <= n`.
- `test_weights_size_mismatch` — raises `KuantShapeError`.

`tests/portfolio/test_construction_batch6.py::TestMinTorsion`.

## References

- Meucci 2009, "Managing diversification," Risk Magazine.
- Meucci, Santangelo & Deguest 2013, "Risk budgeting and
  diversification based on optimized uncorrelated factors,"
  arXiv:1305.5850.

## Related kernels

- `kuant.portfolio.riskparity` — related "each contribution equal"
  idea, but on asset space rather than uncorrelated-factor space.
- `kuant.portfolio.hrp` — feed the HRP weights into `mintorsion` to
  check whether hierarchical bisection produced a genuinely
  diversified bet distribution.
