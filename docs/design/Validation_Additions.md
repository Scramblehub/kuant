# Validation additions — audit of missing validators & warnings

Audit of `kuant/` against the two-line informative-error contract:

```
kuant.<kernel>: <what went wrong, with actual values>.  [<code>]
  → Fix: <copy-pasteable remedy>.
```

Coverage of the current framework in `kuant/_validation.py` is broad — shape,
value-range, NaN/finite, dependencies, and non-convergence are all handled.
The gaps below are places where a silent NaN or a garbage result can still
propagate into user code, and places where the user should be told the
returned answer is unreliable even though nothing "went wrong".

Numbers throughout: 39 remaining `raise Kuant*` sites across 12 files,
0 `warnings.warn()` calls emitted by kuant kernels, no `KuantWarning` class
in `kuant/errors.py`.

---

## Priority A — Missing validators (safety)

Silent-failure cases where the user WILL get garbage without a validator.
All 8 items are gaps we've either confirmed can produce silent NaN or where
the kernel proceeds as if valid on inputs that are semantically broken.

### A1. HMM `pi`/`A`/`B` never validated to be stochastic

- File: [kuant/qm/hmm/forward.py:44](kuant/qm/hmm/forward.py#L44), `_prepare_hmm_inputs`
- Also flows into `backward`, `posterior`, `viterbi`, `baumwelch` (all use the same helper).
- What breaks: user passes an unnormalized `A` (rows sum to 0.9) or a `pi` with a negative entry. `np.log()` still produces a finite matrix (or `-inf`), forward pass runs, and returns a "log-likelihood" that is not a log-likelihood. The Baum-Welch monotonicity check will trip much later with a mysterious `KE-CONV-MONOTONE` error whose actual cause is the bad initial guess.
- Also affects: [kuant/qm/ghmm/common.py:29](kuant/qm/ghmm/common.py#L29) (`_prepare_ghmm_inputs`) for `pi` and `A`.
- Fix (add helper — see Section D1): at the top of `_prepare_hmm_inputs` and `_prepare_ghmm_inputs`, insert
  ```python
  require_stochastic(pi_arr, "pi", kernel="hmm.forward")
  require_stochastic_rows(A_arr, "A", kernel="hmm.forward")
  require_stochastic_rows(B_arr, "B", kernel="hmm.forward")  # hmm only
  ```
  This is the single most impactful gap in the whole library — every HMM entry point silently accepts junk parameters.

### A2. `impvol` / `impvolbisection` — `max_iter`, `tol` never validated

- Files: [kuant/options/impvol.py:57](kuant/options/impvol.py#L57), [kuant/options/impvolbisection.py:51](kuant/options/impvolbisection.py#L51)
- What breaks: `impvol(price, S, K, T, r, max_iter=0)` skips the Newton loop entirely, then the "final validation" check declares every element out-of-tolerance and the kernel returns all-NaN. Same silent-all-NaN for `tol=0` (never satisfied) or `tol=np.nan` (comparison is always False).
- The impvol module has ZERO validators today (`grep require_ kuant/options/impvol.py` finds nothing).
- Fix — insert near top of both functions:
  ```python
  require_positive(max_iter, "max_iter", kernel="impvol", kind="int")
  require_positive(tol, "tol", kernel="impvol")
  ```
- Also for `impvolbisection`: `require_positive(sigma_lo, ...)`, and enforce `sigma_lo < sigma_hi` (see D2 for the helper if we choose to add one; inline check is fine).

### A3. HMM/Baum-Welch — user-supplied init arrays never shape/probability-validated

- File: [kuant/qm/hmm/baumwelch.py:195](kuant/qm/hmm/baumwelch.py#L195)
- What breaks: warm-start with `pi_init=[0.1, 0.5]`, `A_init=np.eye(3)`, `B_init=np.eye(2,4)`. Sizes disagree, but the code just reads `N = pi.size = 2` and then indexes `A` and `B` as if they matched. Downstream errors are cryptic (shape mismatch inside `posterior`).
- Fix: right after the `if have_init:` block loads pi/A/B, insert:
  ```python
  require_expected_shape(A, "A_init", (N, N), kernel="baumwelch")
  require_expected_shape(B, "B_init", (N, "M"), kernel="baumwelch")
  require_stochastic(pi, "pi_init", kernel="baumwelch")
  require_stochastic_rows(A, "A_init", kernel="baumwelch")
  require_stochastic_rows(B, "B_init", kernel="baumwelch")
  ```
  Same for `kuant/qm/ghmm/baumwelch.py:212` on `pi_init` and `A_init`.

### A4. `sindylasso` / `pinnscan` / `symbolicscan` — no int-kind validation on `n_splits`, `max_iter`, `n_estimators`

- Files: [kuant/sindy/sindylasso.py:86](kuant/sindy/sindylasso.py#L86), [kuant/sindy/pinnscan.py:85](kuant/sindy/pinnscan.py#L85), [kuant/sindy/symbolicscan.py:93](kuant/sindy/symbolicscan.py#L93)
- What breaks: `n_splits=1` produces a sklearn error (`KFold` requires ≥ 2) that leaks a stdlib traceback the user has to decode. `n_splits > n_samples` after NaN drop produces "n_splits=5 greater than the number of samples". `n_perms=0` in `permtest` sets `p_value = 1/1 = 1.0` — a meaningless perfect null.
- Fix — near the top of each function:
  ```python
  require_range(n_splits, "n_splits", kernel="sindylasso", lo=2, hi=float("inf"))
  require_positive(max_iter, "max_iter", kernel="sindylasso", kind="int")
  require_positive(n_estimators, "n_estimators", kernel="pinnscan", kind="int")
  ```
  For `permtest` at [kuant/sindy/permtest.py:52](kuant/sindy/permtest.py#L52):
  ```python
  require_positive(n_perms, "n_perms", kernel="permtest", kind="int")
  ```

### A5. `sindylasso` / `pinnscan` — `n_samples < n_features` not checked

- File: [kuant/sindy/sindylasso.py:141](kuant/sindy/sindylasso.py#L141) (after NaN drop)
- What breaks: passing a library with 50 candidate features against a 30-sample target — after NaN drop, LASSO fits an underdetermined system. CV picks endpoint alpha because everything is degenerate. `require_min_clean` catches `< 30`, but not "clean but too narrow for width". This is the "n < p" trap that classically produces overconfident-looking null results.
- Fix — after `X_clean, y_clean = X[mask], y[mask]`:
  ```python
  n_samp, n_feat = X_clean.shape
  if n_samp < 2 * n_feat:
      raise KuantValueError(
          f"kuant.sindylasso: only {n_samp} clean samples for {n_feat} "
          f"features; LASSO CV is unreliable when n_samples < 2·n_features.  "
          f"[KE-VAL-UNDERDET]\n"
          f"  → Fix: raise n_samples above {2 * n_feat}, or narrow the "
          f"library (drop features you don't have a prior on)."
      )
  ```
  Same shape at [kuant/sindy/pinnscan.py:127](kuant/sindy/pinnscan.py#L127) and [kuant/sindy/symbolicscan.py:154](kuant/sindy/symbolicscan.py#L154). The polynomial expansion in `symbolicscan` makes this bite HARDER — degree=2 on 10 features expands to 65 columns and users don't realize it.

### A6. `rollcoherence` — `fs`, `band`, `nperseg` never validated

- File: [kuant/stats/rollcoherence.py:19](kuant/stats/rollcoherence.py#L19)
- What breaks: `band=(0.3, 0.1)` — inverted range, output is all NaN silently (empty `mask`). `fs=0` — scipy raises a cryptic ValueError inside the loop that is caught + swallowed, so the whole output is NaN. `nperseg > window` is defensively clamped, but `nperseg=0` isn't.
- Fix — insert after `require_positive(w, "window", ...)`:
  ```python
  require_positive(fs, "fs", kernel="rollcoherence")
  lo_band, hi_band = band
  require_range(lo_band, "band[0]", kernel="rollcoherence", lo=0.0, hi=fs / 2)
  require_range(hi_band, "band[1]", kernel="rollcoherence", lo=0.0, hi=fs / 2)
  if lo_band >= hi_band:
      raise KuantValueError(
          f"kuant.rollcoherence: 'band' lo={lo_band} must be strictly less "
          f"than hi={hi_band}.  [KE-VAL-RANGE]\n"
          f"  → Fix: pass (lo, hi) with lo < hi in cycles/sample (Nyquist=fs/2={fs/2})."
      )
  if nperseg is not None:
      require_positive(nperseg, "nperseg", kernel="rollcoherence", kind="int")
  ```

### A7. `dfa` / `hurstrs` — non-1D input silently ravels

- Files: [kuant/stats/dfa.py:74](kuant/stats/dfa.py#L74), [kuant/stats/hurstrs.py:110](kuant/stats/hurstrs.py#L110) (hurstrs already has `require_1d` after `_to_numpy`, but dfa does `np.asarray(x, dtype=np.float64).ravel()` which silently flattens a 2D matrix).
- What breaks: `dfa(returns_matrix)` where `returns_matrix` is `(T, n_names)` ravels row-major and computes a nonsense H over the interleaved values. Silent — user thinks they got a Hurst exponent for one series.
- Fix in dfa.py, replace `arr = np.asarray(x, dtype=np.float64).ravel()` with:
  ```python
  arr = np.asarray(x, dtype=np.float64)
  require_1d(arr, "x", kernel="dfa")
  arr = arr[np.isfinite(arr)]
  ```

### A8. `decoherencescan` — `train_window`, `predict_window` vs `T` never checked

- File: [kuant/qm/decoherencescan.py:61](kuant/qm/decoherencescan.py#L61)
- What breaks: `train_window >= T` makes the `while t + predict_window <= T:` loop never execute; every bucket has `n < 2` so every correlation is 0.0. The result is a formally valid `DecoherenceScanResult` where all buckets read as noise — user gets no error, just a confusing table.
- Fix — after `T = len(y)`:
  ```python
  require_positive(train_window, "train_window", kernel="decoherencescan", kind="int")
  if train_window + predict_window > T:
      raise KuantValueError(
          f"kuant.decoherencescan: need train_window ({train_window}) + "
          f"predict_window ({predict_window}) <= len(y) ({T}), got "
          f"{train_window + predict_window} > {T}.  [KE-VAL-RANGE]\n"
          f"  → Fix: shorten train_window/predict_window, or provide more data."
      )
  ```
  Also `require_2d(X, "X", kernel="decoherencescan")`.

---

## Priority B — Warnings to add

`kuant.errors` has no `KuantWarning` class yet — this section proposes both
the class (a one-liner in `errors.py`) and 5 first call sites for it. See
Section D3.

### B1. Baum-Welch `converged=False` — silent unreliability

- File: [kuant/qm/hmm/baumwelch.py:286](kuant/qm/hmm/baumwelch.py#L286) (return statement), same at [kuant/qm/ghmm/baumwelch.py](kuant/qm/ghmm/baumwelch.py)
- Condition: `converged is False` when we exit the EM loop by hitting `max_iter`.
- Rationale: The kernel returns a `BaumWelchResult` regardless. Currently `.converged` is a bool field that most callers never check. If EM hit the wall without converging, downstream inference (Viterbi, posterior gates, regime classification) inherits noisy parameters. The user has no in-band signal unless they inspect the result manually.
- Fix — right before the `return BaumWelchResult(...)` at the end of the function:
  ```python
  if not converged:
      last_improvement = (log_lik_history[-1] - log_lik_history[-2]
                          if len(log_lik_history) >= 2 else float("nan"))
      warnings.warn(
          _msg(
              "baumwelch",
              "KW-CONV-MAX-ITER",
              f"EM did not converge after {len(log_lik_history)} iterations "
              f"(last ΔlogL {last_improvement:+.2e}, tol {tol:g})",
              f"raise max_iter (currently {max_iter}) or loosen tol; "
              f"result is returned but parameters are still improving",
          ),
          KuantConvergenceWarning,
          stacklevel=2,
      )
  ```
  Distinct from `did_not_converge` (which raises) — this is the "we're returning something, but you should know" case that Baum-Welch was explicitly designed to support.

### B2. `sindylasso` / `symbolicscan` — CV picked grid endpoint

- Files: [kuant/sindy/sindylasso.py:154](kuant/sindy/sindylasso.py#L154), [kuant/sindy/symbolicscan.py:169](kuant/sindy/symbolicscan.py#L169)
- Condition: `abs(alpha_selected - alpha_grid[0]) < 1e-12` (weakest reg, overfit risk) OR `abs(alpha_selected - alpha_grid[-1]) < 1e-12` (strongest reg, null signature).
- Rationale: The endpoint hit means the search range didn't bracket the optimum. The result `summary()` prints a diagnostic string on the strongest-endpoint case only — but programmatic callers reading `.r2` never see it, and the WEAKEST-endpoint case is silently ignored. That is the overfit signature and it's the one you most want a warning on.
- Fix — right before `return SindyLassoResult(...)`:
  ```python
  a_lo, a_hi = float(alpha_grid.min()), float(alpha_grid.max())
  a_sel = float(model.alpha_)
  if abs(a_sel - a_lo) < 1e-12:
      warnings.warn(
          _msg("sindylasso", "KW-CV-ENDPOINT-LOW",
               f"CV picked the weakest regularization α={a_sel:g} at the "
               f"bottom of the grid [{a_lo:g}, {a_hi:g}]",
               "expand alpha_grid downward (e.g. np.logspace(-8, -1, 40)) — "
               "the true optimum may be even smaller, meaning selected "
               "features may be overfit"),
          KuantWarning, stacklevel=2)
  elif abs(a_sel - a_hi) < 1e-12:
      warnings.warn(
          _msg("sindylasso", "KW-CV-ENDPOINT-HIGH",
               f"CV picked the strongest regularization α={a_sel:g} at the "
               f"top of the grid; {len(selected)} features selected",
               "if 0 features selected this is a clean null; otherwise "
               "expand alpha_grid upward to confirm you've found the plateau"),
          KuantWarning, stacklevel=2)
  ```
  Same at `symbolicscan`.

### B3. `tailindex` (Hill) — degenerate result silently returns NaN

- File: [kuant/stats/tailindex.py:20](kuant/stats/tailindex.py#L20)
- Condition: `arr.size < min_k + 2` OR `k >= arr.size` — both currently `return float("nan")` with no explanation. Also: negative Hill estimate (ξ < 0) on financial loss data — that's almost never right physically (returns don't have bounded support) and is a strong signal of insufficient tail samples or wrong sign convention on the input.
- Rationale: Hill is well-known to be biased on non-Pareto tails and unstable at low k. Users who ship a Hill estimate without seeing warnings often ship a nonsense number.
- Fix — replace both `return float("nan")` sites with a warning + return, and add a post-fit sanity check:
  ```python
  if arr.size < min_k + 2:
      warnings.warn(
          _msg("tailindex", "KW-VAL-INSUFFICIENT-TAIL",
               f"only {arr.size} positive finite values, need >= {min_k + 2}",
               "provide more data, lower min_k, or check that the input "
               "contains positive loss magnitudes (not signed returns)"),
          KuantWarning, stacklevel=2)
      return float("nan")
  # … after computing xi_hat …
  if xi_hat < 0:
      warnings.warn(
          _msg("tailindex", "KW-HILL-NEGATIVE",
               f"Hill estimate ξ={xi_hat:.3f} is negative (bounded-support "
               f"regime); rarely correct on financial loss data",
               "check that x is positive loss magnitudes, not returns; "
               "raise k_frac; or use a POT/EVT fit that estimates ξ and σ jointly"),
          KuantWarning, stacklevel=2)
  ```

### B4. `persistenthomology` — fewer than ~20 points

- File: [kuant/topology/persistenthomology.py:193](kuant/topology/persistenthomology.py#L193)
- Condition: `cloud.shape[0] < 20` (already returns empty diagrams for `< 2`, but 2..19 quietly produces a persistence diagram nobody should trust).
- Rationale: Persistent homology's asymptotics kick in around 30–50 points depending on intrinsic dimension. Users passing a 100-bar window with a Takens embedding of dim=5, delay=3 end up with only 88 points — reasonable — but a 30-bar window at dim=5 delay=3 gives 18 points and the diagram is noise. `bettiseries` composes this hundreds of times per call.
- Fix — right after `cloud = _time_delay_embed(...)` or `cloud = arr`:
  ```python
  if 2 <= cloud.shape[0] < 20:
      warnings.warn(
          _msg("persistenthomology", "KW-TOPO-FEW-POINTS",
               f"only {cloud.shape[0]} points in the cloud",
               "results are dominated by boundary effects below ~20 points; "
               "shorten delay/embedding_dim (currently d={emb_d}, τ={d_delay}) "
               "or widen the window"),
          KuantWarning, stacklevel=2)
  ```

### B5. Baum-Welch state re-seeding — silent quality loss

- File: [kuant/qm/hmm/baumwelch.py:269](kuant/qm/hmm/baumwelch.py#L269)
- Condition: `len(reseeded_states) > 0` at the end of training.
- Rationale: A state that had to be re-seeded during EM had zero total responsibility — the model collapsed a state onto ~nothing. The result field records this, but callers seldom check it. A re-seeded state means the final N is effectively N-1 (or worse), and the log-likelihood is not comparable across (N, seed) grids for model selection.
- Fix — near the same return-time block as B1:
  ```python
  if reseeded_states:
      warnings.warn(
          _msg("baumwelch", "KW-HMM-STATE-COLLAPSE",
               f"{len(reseeded_states)} state(s) had zero responsibility "
               f"and were re-seeded during EM: {reseeded_states}",
               "the fit ran to completion but the effective state count is "
               "lower than n_states; refit with different seed, fewer states, "
               "or a longer observation sequence"),
          KuantWarning, stacklevel=2)
  ```

---

## Priority C — Message polish

The audit's 39 remaining `raise Kuant*` sites were reviewed. All follow the
contract (kernel prefix, actual values, `[KE-...]` code, `→ Fix:` line). Two
minor issues:

### C1. `baumwelch` uses `require_probability(tol, ...)` — semantically off

- File: [kuant/qm/hmm/baumwelch.py:177](kuant/qm/hmm/baumwelch.py#L177) and [kuant/qm/ghmm/baumwelch.py:196](kuant/qm/ghmm/baumwelch.py#L196)
- Current: reuses `require_probability` because tol=1e-4 happens to be in `[0, 1]`. Message says "must be in [0, 1]" which is misleading — nothing stops a user from wanting `tol=2` on a very rough fit (converge as soon as ΔlogL < 2).
- Suggested: `require_positive(tol, "tol", kernel="baumwelch")` — the actual constraint. The upper bound in `require_probability` is spurious.

### C2. `KE-VAL-MUTEX` is used but never centralized

- Files raising it: [kuant/stats/rollema.py:93](kuant/stats/rollema.py#L93), [kuant/stats/rollemastd.py:93](kuant/stats/rollemastd.py#L93), [kuant/qm/hmm/baumwelch.py:186](kuant/qm/hmm/baumwelch.py#L186), [kuant/qm/ghmm/baumwelch.py:204](kuant/qm/ghmm/baumwelch.py#L204)
- The code appears four times as an inline literal. Not a message issue per se — every one of them has a good fix line — but this is the strongest candidate for centralizing (see D2).

Everything else in the 39 remaining sites is well-formed. In particular the
GHMM/HMM Baum-Welch monotonicity error and the DFA/Hurst degeneracy errors
already exceed the contract's information bar.

---

## Section D — Proposed additions to `_validation.py`

Three helpers. Bar is: appears at ≥ 3 call sites AND encodes a math
constraint the kernel would otherwise silently accept.

### D1. `require_stochastic` and `require_stochastic_rows`

Priority A1 and A3 are entirely blocked without this — HMM/GHMM shape validation
is already thorough but nothing checks the probability-simplex constraint.

```python
def require_stochastic(vec: Any, name: str, *, kernel: str, atol: float = 1e-6) -> None:
    """Reject a vector that isn't a probability distribution over its axis."""
    arr = _to_ndarray(vec)
    if arr.dtype.kind not in "fc":
        arr = arr.astype(np.float64)
    if np.any(arr < -atol) or np.any(arr > 1.0 + atol):
        bad = int(np.argmin(np.where(arr < 0, arr, np.abs(arr - 0.5))))
        raise KuantValueError(
            _msg(
                kernel,
                "KE-VAL-STOCHASTIC",
                f"'{name}' must lie in [0, 1] (probability distribution); "
                f"index {bad} has value {float(arr.flat[bad])}",
                f"pass a valid probability vector — clip to [0, 1] and "
                f"renormalize with `{name} / {name}.sum()`",
            )
        )
    s = float(arr.sum())
    if abs(s - 1.0) > atol:
        raise KuantValueError(
            _msg(
                kernel,
                "KE-VAL-STOCHASTIC",
                f"'{name}' must sum to 1 (probability distribution), got sum={s:.6g}",
                f"renormalize before calling — `{name} = {name} / {name}.sum()`",
            )
        )


def require_stochastic_rows(mat: Any, name: str, *, kernel: str, atol: float = 1e-6) -> None:
    """Reject a matrix whose rows aren't probability distributions."""
    arr = _to_ndarray(mat)
    if arr.dtype.kind not in "fc":
        arr = arr.astype(np.float64)
    if arr.ndim != 2:
        # Shape errors are the shape helper's job — assume caller ran require_expected_shape first.
        return
    row_sums = arr.sum(axis=1)
    bad = np.where(np.abs(row_sums - 1.0) > atol)[0]
    if bad.size:
        r = int(bad[0])
        raise KuantValueError(
            _msg(
                kernel,
                "KE-VAL-STOCHASTIC-ROWS",
                f"'{name}' row {r} must sum to 1 (transition/emission row), "
                f"got sum={float(row_sums[r]):.6g}",
                f"renormalize each row before calling — "
                f"`{name} / {name}.sum(axis=1, keepdims=True)`",
            )
        )
    if np.any(arr < -atol) or np.any(arr > 1.0 + atol):
        raise KuantValueError(
            _msg(
                kernel,
                "KE-VAL-STOCHASTIC-ROWS",
                f"'{name}' contains values outside [0, 1] "
                f"(min={float(arr.min()):.3g}, max={float(arr.max()):.3g})",
                f"clip and renormalize before calling",
            )
        )
```

**Call sites that would replace ad-hoc/absent checks:**
1. [kuant/qm/hmm/forward.py:64-71](kuant/qm/hmm/forward.py#L64) — after the shape checks, add three lines for `pi`, `A`, `B`.
2. [kuant/qm/ghmm/common.py:41-46](kuant/qm/ghmm/common.py#L41) — same for `pi`, `A`.
3. [kuant/qm/hmm/baumwelch.py:195](kuant/qm/hmm/baumwelch.py#L195) and [kuant/qm/ghmm/baumwelch.py:212](kuant/qm/ghmm/baumwelch.py#L212) — validate warm-start inputs.

### D2. `require_mutex_pair`

Removes the four inline copies of the "exactly one of span/alpha" pattern (span/alpha in ema kernels; pi_init-vs-n_states in Baum-Welch).

```python
def require_mutex_pair(
    a: Any, name_a: str,
    b: Any, name_b: str,
    *, kernel: str,
    a_example: str, b_example: str,
) -> None:
    """Reject when neither or both of a mutually-exclusive arg pair are set.

    Use for OR-XOR constraints: (span XOR alpha), (n_states XOR full init),
    etc. `a_example`/`b_example` are appended to the fix line so users see
    the working form for each branch.
    """
    a_set = a is not None
    b_set = b is not None
    if a_set ^ b_set:
        return
    got = "both" if a_set and b_set else "neither"
    raise KuantValueError(
        _msg(
            kernel,
            "KE-VAL-MUTEX",
            f"provide exactly one of `{name_a}` or `{name_b}`, got {got}",
            f"`{a_example}` OR `{b_example}`",
        )
    )
```

**Call sites that would replace inline `KE-VAL-MUTEX` raises:**
1. [kuant/stats/rollema.py:88-95](kuant/stats/rollema.py#L88) — collapse to `require_mutex_pair(span, "span", alpha, "alpha", kernel="rollema", a_example="span=21", b_example="alpha=0.1")`.
2. [kuant/stats/rollemastd.py:89-96](kuant/stats/rollemastd.py#L89) — same.
3. [kuant/qm/hmm/baumwelch.py:180-189](kuant/qm/hmm/baumwelch.py#L180) — n_states-vs-init pair (needs a small adapter: the "init" side is a triple, so caller passes `pi_init` as `a` and receives the same handling).

### D3. `warn_kuant` + `KuantWarning` (and subclasses)

Section B requires a class hierarchy for warnings that mirrors the error one. This is the ONE addition that touches both `errors.py` and `_validation.py`.

In `errors.py`:

```python
class KuantWarning(UserWarning):
    """Base class for every warning kuant emits directly.

    Subclasses:
      KuantConvergenceWarning — solver returned a result but did not converge.
      KuantNumericWarning     — result is likely unreliable due to input constraints.
    """


class KuantConvergenceWarning(KuantWarning):
    """Iterative solver hit max_iter without meeting tol.

    Distinct from `KuantConvergenceError`: this variant is raised by kernels
    that deliberately return a partial fit rather than failing (e.g. Baum-Welch).
    """
```

In `_validation.py`:

```python
def warn_kuant(*, kernel: str, code: str, what: str, fix: str,
               category: type[Warning] = KuantWarning, stacklevel: int = 3) -> None:
    """Emit a KuantWarning with the standard two-line message shape."""
    import warnings as _warnings
    _warnings.warn(_msg(kernel, code, what, fix), category, stacklevel=stacklevel)
```

**Call sites (all 5 Priority B findings):**
1. [kuant/qm/hmm/baumwelch.py:286](kuant/qm/hmm/baumwelch.py#L286) — B1 non-convergence.
2. [kuant/qm/hmm/baumwelch.py:286](kuant/qm/hmm/baumwelch.py#L286) — B5 re-seed count.
3. [kuant/sindy/sindylasso.py:178](kuant/sindy/sindylasso.py#L178) — B2 CV endpoint (same at symbolicscan).
4. [kuant/stats/tailindex.py:60](kuant/stats/tailindex.py#L60) — B3 Hill sanity.
5. [kuant/topology/persistenthomology.py:207](kuant/topology/persistenthomology.py#L207) — B4 point count.

---

## Section E — Deferred (nice-to-have)

Low-value or context-dependent items surfaced during the audit:

- `bscall`/`bsput`/`bsvega` and all greeks — inputs `T ≤ 0`, `σ ≤ 0`, `S ≤ 0`, `K ≤ 0` are ALREADY handled with intrinsic-value fallbacks in [_bs_common.py](kuant/core/_bs_common.py). No validators needed; behavior is documented and consistent. Do not "fix".
- `rollsharpe` / `rollsortino` — `ann_factor` unvalidated but a negative value gives an obviously wrong sign that users notice immediately.
- `deltabucket` — `targets` outside `[-1, 1]` will still return a nearest index (the min-of-abs-diff wins), which is arguably wrong but harmless (user gets the closest edge of the chain). Not worth an error.
- `permtest` — no warning when `n_perms < 100` even though the p-value's minimum resolution is `1/(n_perms+1)`. Real but the summary already shows `n_perms` prominently.
- `zenoscan` — `retrain_freqs` values not validated positive, but nonsense values (0, negative) throw a stdlib `ZeroDivisionError` naturally within a few lines.
- `hurstrs` catches `ValueError` inside `rollhurst`'s trailing loop — this catches the very `KuantValueError` we spent effort producing. Would be more consistent to catch `KuantError` instead. Cosmetic.
- `impvol` returns NaN for elements that didn't converge and prints nothing about how many. A `KuantWarning` with the count would help — but the user can compute `np.isnan(result).sum()` trivially.
- Dozens of rolling kernels take `x` and quietly int-promote to float64. If the user's memory budget matters they may prefer to keep float32. Not a validator issue; a performance note in docs would be enough.

---

## Summary

- **8 Priority A** items (safety-critical validators to add). The HMM
  stochastic-input gap (A1) alone touches 4 kernels through a shared helper
  and is the single biggest exposure.
- **5 Priority B** warnings, all requiring `KuantWarning` in `errors.py` first
  (proposal D3).
- **2 Priority C** message polish items.
- **3 new helpers** proposed for `_validation.py`
  (`require_stochastic{,_rows}`, `require_mutex_pair`, `warn_kuant`).
- Priority-D3 unlocks all of B; recommend landing D3 alongside B1/B5 first
  (Baum-Welch is highest-usage). A1 next (single change, 4-kernel win).
