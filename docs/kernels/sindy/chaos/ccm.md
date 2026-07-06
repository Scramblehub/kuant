# ccm: Sugihara convergent cross-mapping

## Purpose

Test for nonlinear causality between two series `x` and `y`. Embed
each into its own shadow manifold. Ask: can `y_t` be predicted from
the nearest neighbors of `x_t` in `x`'s manifold, via simplex
projection? If prediction skill `rho_xy(L)` rises with the library
size `L` and lifts above a threshold, `y`'s information lives in
`x`'s history, which implies the causal direction `y -> x` (y
drives x).

The rising-with-`L` (convergent) behavior is the causality signal.
Correlation at a single library size is not enough: noise can hit a
lucky small-library `rho` that never grows. Convergence with `L`
is what distinguishes a real dynamical coupling from a coincidence.

CCM sidesteps two problems that break Granger causality:

- **Nonlinear couplings**: Granger's linear autoregressions miss
  them. CCM does not care about linearity.
- **Common-driver confounds**: if `x` and `y` share a hidden third
  driver, Granger flags both directions. CCM's convergence check
  distinguishes true dynamical coupling from shared-driver
  structure.

CCM has failure modes: it assumes deterministic dynamics on a
low-dim attractor. Noisy stochastic data can produce false-positive
`rho` values at small `L` that resolve at large `L`, so the
convergence check must not be skipped.

## Public API

```python
from kuant.sindy.chaos import ccm

c = ccm(x, y, tau=1, m=5, n_seeds=5, convergence_threshold=0.1)
print(c.summary())
c.convergence_xy    # bool: y -> x is convergent?
c.convergence_yx    # bool: x -> y is convergent?
c.rho_xy, c.rho_yx  # full skill curves vs library size
```

Signature:

```python
ccm(
    x, y, *, tau=1, m=5,
    lib_sizes=None, n_seeds=5,
    convergence_threshold=0.1, seed=0,
)
```

Returns `CCMResult` with `lib_sizes`, `rho_xy`, `rho_yx`, the two
convergence booleans, and the threshold used.

## Design decisions

### Simplex-projection prediction with `k = m + 1` neighbors

Each target value is predicted from its `m + 1` nearest neighbors in
the shadow manifold, weighted `exp(-d / d_min)`. Sugihara's original
choice; adequate at the sample sizes here without introducing S-map
tuning parameters.

### Default `lib_sizes` = 5 log-spaced

`np.logspace(log10(2 * (m + 1)), log10(N), 5)`. Small enough to see
the undersaturated regime, large enough that a real coupling has
room to converge. Callers who need a finer curve should pass an
explicit list.

### `n_seeds=5` per library size

Each library size averages `rho` over 5 random library draws to
damp seed-dependent variance. Fully deterministic given the
top-level `seed`.

### Convergence rule

`rho[-1] - rho[0] > convergence_threshold`. Simple and reproducible.
Callers who want a monotonicity check across the whole curve should
inspect `rho_xy` and `rho_yx` directly.

### 200-observation floor

Below 200 finite pair-rows, the cross-mapping is dominated by
finite-library variance and the convergence signal is unreadable.

## When it fires

- Whenever "does X drive Y?" is the question and the coupling is
  suspected nonlinear or subject to common-driver confounding.
- Not called by `chaosscan` (which scans a single series). Use it
  alongside `chaosscan` when you have two series to compare.

## References

- Sugihara et al 2012, "Detecting causality in complex ecosystems."
