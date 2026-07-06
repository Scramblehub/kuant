# Changelog

All notable changes to `kuant` are tracked here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); dates are release dates
on PyPI.

## v0.5.1

- **`kuant.sindy.chaos`** lands: chaos-theory diagnostics for regime
  classification and nonlinear-causality testing. Seven kernels + a
  composer, sharing a single time-delay embedding helper. All CPU-only,
  numpy-only, no heavy deps.
  - `mutualinfo(x, y=None, ...)`: auto-MI curve vs lag (mode 1, returns
    `MutualInfoResult` with `suggested_tau` at first local minimum) or
    cross-MI scalar between two series at a given lag (mode 2). Fraser-
    Swinney 1986 histogram estimator.
  - `falsenearest(x, tau, ...)`: false-nearest-neighbors fraction vs
    embedding dimension (Kennel-Brown-Abarbanel 1992). Returns
    `FalseNearestResult` with `suggested_m` at the first dim below a
    5%-default threshold.
  - `lyapunov(x, tau, m, ...)`: largest Lyapunov exponent via
    Rosenstein-Collins-DeLuca 1993. Configurable Theiler window;
    returns full log-divergence curve so callers can visually check
    the linear fit region.
  - `corrdim(x, tau, m, ...)`: correlation dimension via
    Grassberger-Procaccia 1983. Fits the middle 60% of the log-log
    C(r) curve to avoid noise-floor and finite-size regimes.
  - `rqa(x, tau, m, ...)`: recurrence-quantification analysis
    (Marwan-Romano-Thiel-Kurths 2007). Recurrence rate, determinism,
    laminarity, longest diagonal, entropy of diagonal lengths.
    Auto-picks epsilon to hit a target recurrence rate if not given.
  - `ccm(x, y, tau, m, ...)`: convergent cross-mapping (Sugihara 2012)
    for coupled deterministic systems. Simplex-projection cross-map
    prediction skill vs library size, both directions, with a
    convergence flag.
  - `chaosscan(x, ...)`: composer. Auto-picks (tau, m), runs the full
    battery, classifies into `{chaotic, periodic, stochastic, unknown}`.
- **Tests**: +52 chaos-kernel tests (1933 → 1985 total).
- **No breaking changes.**

## v0.5.0

- **`kuant.qm.quaternion`** lands: Hamilton (w-first) unit-quaternion
  algebra plus two kernels aimed at regime-drift signals.
  - `Quaternion(w, x, y, z)` frozen dataclass with strict unit-norm
    enforcement (auto-normalizes non-unit input, raises on zero norm).
    Methods: `multiply`, `conjugate`, `inverse`, `rotate(v)`,
    `to_axis_angle` / `from_axis_angle`, `to_rotation_matrix` /
    `from_rotation_matrix`, `angle`.
  - Module-level array ops on `(4,)` scalars or `(..., 4)` batches:
    `quat_multiply`, `quat_conjugate`, `quat_normalize`, `quat_angle`,
    `quaternion_distance`, `slerp` (with linear-then-normalize fallback
    at parallel-quaternion pole).
  - `composerotations(quats, return_trajectory=False)`: list-order
    equals application-order via a right-to-left Hamilton product.
    Two input forms (numpy `(T, 4)` or list of `Quaternion`), optional
    running-composition trajectory output.
  - `rollholonomy(quats, window)`: trailing-window composed quaternion
    and its rotation-angle magnitude per bar. First `w - 1` rows NaN;
    windows containing any NaN propagate. Naive O(T*w) in v1; a
    prefix-product O(T) variant is queued.
- **Docs batch**: 20 new markdown files (~4100 lines) closing the docs
  debt accumulated since v0.3.2. Kernel docs for
  `kuant.backtest.liquidity`, `kuant.backtest.fill`,
  `kuant.backtest.position`, `kuant.backtest.warmup`,
  `kuant.backtest.engine`, and `kuant.qm.quaternion`. New central
  `docs/design/errors-and-warnings-index.md` indexing every stable
  `KE-*` and `KW-*` code (38 errors + 64 warnings = 102 total) with
  their exception class, kernel, and one-line trigger.
- **New `kuant.qm` submodule**: `quaternion` joins `hmm`, `ghmm`.
  Version minor-bumped to signal the new surface even though the
  addition is purely additive.
- 56 new quaternion tests. Regression: 1877 -> 1933 pass (+56).
  Coverage 91% held.

## v0.4.7

- **Tier-2B warnings close-out** (Session 5B, part 2): the audit's
  remaining ~13 findings across options, sindy, qm, backtest, and
  edgecases. Wraps up the full-library warnings/errors sweep started
  in v0.4.4.
- **Options:**
  - `impvol` post-loop diagnostics: `KW-CONV-MAX-ITER` names how many
    in-bounds cells failed to converge and their retry knobs;
    `KW-NUM-VEGA-DEGENERATE` flags cells where the Newton step floors
    below `_VEGA_MIN` (deep-OTM / very-short-tenor).
  - `impvolbisection`: `KW-CONV-MAX-ITER` when the bisection bracket
    still exceeds `tol`; `KW-VAL-RANGE` when the passed price is
    outside the `[sigma_lo, sigma_hi]` no-arbitrage bracket.
  - `deltabucket`: `KW-NUM-NO-MATCH` when the nearest available delta
    is more than 0.05 off the target.
- **Sindy / QM:**
  - `grangerscan`: `KW-NUM-SAMPLE-SIZE` per candidate with fewer than
    100 clean rows (F-test asymptotics become anti-conservative).
  - `decoherencescan`: `KW-NUM-BUCKET-SMALL` for day-in-window buckets
    with fewer than 30 samples; per-bucket correlation is dominated by
    sampling noise.
- **Backtest:**
  - `execute_fill_panel`: `KW-FILL-PANEL-EMPTY` and
    `KW-FILL-PANEL-EXTRA-COLS`.
  - `PortfolioState.mark_to_market`: `KW-PORTFOLIO-NAN-MARK` when a
    non-finite price is supplied for an open position; the NaN
    propagates but the diagnostic surfaces.
  - `WarmupCache.tradeable` / `.liquid` / `.universe`:
    `KW-CACHE-TS-NOT-IN-PANEL` (timestamp missing from the cache
    index) and `KW-CACHE-UNIVERSE-UNKNOWN-SYMBOL` (symbol referenced
    but not registered in the membership panel).
- **Edgecases:**
  - `outlierpolicy`: `KW-OUTLIER-EXTREME-RATE` when a threshold flags
    0% or ≥99% of finite values (a strong signal of units mismatch
    between method families).
  - `zero_after_delist` and `hold_last_price` now emit
    `KW-DEPRECATED-USE-LIFECYCLE` (a `KuantDeprecationWarning`)
    pointing at their `kuant.backtest.lifecycle` replacements.
    Scheduled for removal in v0.6.0.
- 14 new Tier-2B audit tests colocated at `tests/test_tier2b_audit.py`.
- Regression: 1863 -> 1877 pass (+14). Coverage held at 91%.

## v0.4.6

- **`kuant.backtest.engine`** lands. Reference orchestrator over the
  correctness-first backtest primitives (`lifecycle`, `liquidity`,
  `fill`, `position`, `warmup`). Intentionally small (~200 lines); users
  who want richer semantics build on top of the primitives directly.
- `run(cache, strategy, liquidity_profiles, fill_model, initial_cash,
  lifecycles=None) -> BacktestResult`. Bar-driven loop:
  - For each timestamp, the user's `strategy(cache, state, timestamp)`
    returns a list of `Order`s.
  - Each order is gated: lifecycle `tradeable_mask`, presence of a
    liquidity profile, symbol presence in the price panel, finite
    positive reference price. Gated orders are recorded (never silently
    dropped) with categorical reasons `GATED_LIFECYCLE`, `NO_PROFILE`,
    `SYMBOL_NOT_IN_PANEL`, `NO_PRICE`.
  - Surviving orders route through `submit_order` -> `execute_fill`.
    The `FillReport` is applied atomically to the `PortfolioState`.
  - Bar closes with a `mark_to_market` snapshot appended to the equity
    curve.
- `BacktestResult`: `equity` (per-bar DataFrame with cash /
  positions_value / total_value / unrealized_pnl / realized_pnl),
  `trades` (per-order DataFrame including gated intents),
  `portfolio_final`, order counters, `summary()`, and `to_parquet(path)`
  emitting `{path}_equity.parquet` and `{path}_trades.parquet`.
- 16 new tests spanning empty runs, buy-and-hold equity growth, gating
  under lifecycle / no-profile / symbol-missing / NaN-price, rejected
  fills below min_size, slippage causing lower final equity than
  zero-slip runs, state visibility across bars, and `.to_parquet`
  round-trip.
- Regression: 1847 -> 1863 pass (+16). Coverage held at 91%.

## v0.4.5

- **Tier-2 warnings sweep** (Session 5B): ~40 new `KuantNumericWarning`
  paths across the library, closing silent-hides-bug patterns identified
  by the audit. Every warning is catchable as `KuantNumericWarning` or
  promotable to a hard error via
  `warnings.filterwarnings("error", category=KuantNumericWarning)`.
- **New shared helpers in `kuant._validation`:**
  - `warn_window_exceeds_data(w, n, kernel)`: reused across ~14 rolling
    kernels that returned silent all-NaN when the window exceeded the
    input length.
  - `warn_ddof_exceeds_window(ddof, w, kernel)`: `rollstd`, `rollcov`.
  - `warn_zero_denominator(name, kernel)`: `sharperatio`, `sortinoratio`,
    `kelly`, `rollsharpe`, `rollsortino`, `rollcalmar`.
- **Portfolio silent-zero warnings:**
  `KW-SHARPE-CONSTANT-RETURNS`, `KW-SORTINO-TINY-DOWNSIDE`,
  `KW-DRAWDOWN-ALL-NAN`, `KW-ULCER-ALL-NAN`, `KW-KELLY-ZERO-VARIANCE`,
  `KW-KELLY-NEGATIVE-EDGE`, `KW-CAPTURE-NO-UP-PERIODS`,
  `KW-CAPTURE-NO-DOWN-PERIODS`, `KW-PSR-INVALID-MOMENTS`,
  `KW-DSR-NO-TRIALS`.
- **Signals silent-degenerate warnings:**
  `KW-WINSORIZE-AGGRESSIVE-LIMITS`, `KW-FIC-SKIPPED-PERIODS`,
  `KW-RANK-CONSTANT-FACTOR`, `KW-QUANTILE-THIN-BUCKETS`,
  `KW-TURNOVER-DEGENERATE-FACTOR`, `KW-NEUTRALIZE-CONSTANT-SIGNAL`,
  `KW-ICDECAY-NO-CLEAN`.
- **Nulltest resolution warnings:**
  `KW-BOOT-BLOCK-TOO-LONG` on `stationary_bootstrap` when the mean block
  length degenerates the resample. `KW-BOOT-LOW-N-BOOT` on `bootstrap_ic`
  below 100 draws.
- **Stats window / ddof / zero-denom warnings** via shared helpers:
  `rollstd`, `rollmean`, `rollsum`, `rollmoments` (rollskew, rollkurt),
  `rollmad`, `rollmdd`, `rollminmax`, `rollargminmax`, `rollquantile`,
  `rollrank`, `rollcov`, `rollcorr`, `rollbeta`, `atr` all warn on
  window > n. `rollstd` and `rollcov` warn on ddof >= window.
  `rollsharpe` (`KW-NUMERIC-ZERO-STD`), `rollsortino`
  (`KW-NUMERIC-ZERO-DOWNSIDE`), and `rollcalmar`
  (`KW-NUMERIC-ZERO-DRAWDOWN`) warn on zero-denominator windows.
- **QM state-order permutation warnings:** both `kuant.qm.hmm.baumwelch`
  and `kuant.qm.ghmm.baumwelch` warn `KW-HMM-STATE-ORDER` on every
  successful fit. Baum-Welch is permutation-invariant over state indices,
  and elementwise comparisons across independent fits are misleading
  without an alignment step.
- **Backtest / data warnings:**
  `KW-ALIGN-EMPTY-INTERSECT` on inner joins with disjoint indices,
  `KW-LIFECYCLE-UNKNOWN-SYMBOL` on `apply_lifecycle_panel` when the map
  references symbols not in the panel, `KW-LIQ-MASK-ALL-FALSE` on
  `liquidity_mask` when `min_adv` excludes every date.
- **Test updates:** 33 new Tier-2 audit tests colocated at
  `tests/test_tier2_audit.py`. One existing kelly test tightened to
  assert the new warning fires alongside the return value.
- Regression: 1814 -> 1847 pass (+33), 83 skip. Coverage 91% held.
- Tier 2B remainders (impvol Newton non-convergence,
  `deltabucket` no-match, `decoherencescan` bucket-small,
  `grangerscan` sample-size, `execute_fill_panel` extra columns,
  `PortfolioState.mark_to_market` NaN prices, `outlierpolicy`
  extreme rate, Warmup timestamp-not-in-panel) queued for the next
  minor.

## v0.4.4

- **Tier-1 warnings/errors sweep across the full library** (39 new error
  paths + 3 new validation helpers). Closes silent-happy-path returns
  identified by a full-library audit:
  - **Backtest accounting.** NaN size or price in `execute_fill`,
    `Position.apply_fill`, and `PortfolioState.apply_fill` now raise
    `KE-FILL-SIZE-NAN`, `KE-POS-PRICE-INVALID`, `KE-POS-SIZE-INVALID`,
    `KE-PORTFOLIO-FILL-PRICE-INVALID`.
  - **Null-test NaN pollution.** `mht_correction` raises on any NaN
    p-value (`KE-VAL-NAN-PVALUES`); `permtest` raises on a NaN
    `real_metric` (`KE-VAL-FINITE`, which was silently reporting
    p=1/(n+1) that looked like a highly-significant result).
  - **Empty-input guards.** `drawdown`, `belltest` features dict,
    `sindylasso` library dict, `deltabucket`, `Warmup` panel.
  - **User-supplied fn contract checks.** `belltest` joint_model_fn and
    `decoherencescan` predict_fn wrong-length return.
  - **Core distribution parameter guards.** `gpdcdf/pdf/ppf` scale <= 0;
    `tcdf/tpdf/logtcdf` df <= 0; `moneynessbucket` S or K <= 0;
    `corpaction` split ratio <= 0.
  - **Rolling-window range errors.** `rollskew` w<3, `rollkurt` w<4,
    `rollcorr` w<2, `rollbeta` w<2, `varianceratiotest` lags<2,
    `accelerationscan` len(x)<3, `bettiseries` window>len(x),
    `dispersioncollapse` window>n_bars.
  - **Realized-vol OHLC ordering.** `parkinson` H<L,
    `garmanklass` / `rogerssatchell` / `yangzhang` OHLC-relationship
    violations.
  - **Categorical enum contracts.** `Order` LIMIT/STOP with 0 or NaN
    limit_price (`KE-ORDER-LIMIT-INVALID`); `submit_order` silent
    fallthrough on unknown FillResult reasons
    (`KE-SUBMIT-UNKNOWN-REASON`); `Warmup` names the failing indicator
    when a registered kernel raises during materialize
    (`KE-WARMUP-INDICATOR-FAILED`).
- New helpers in `kuant._validation`: `require_non_empty`,
  `require_monotone_increasing`, `require_ohlc_ordering`.
- Regression: 1770 -> 1814 pass (+44), 82 skip -> 83 skip (arch dep skip
  in one new test). Coverage 90% -> 91%.

## v0.4.3

- `kuant.backtest.warmup` lands.
  `Warmup(prices, mode)` registers indicators, universe membership,
  lifecycle maps, and liquidity profiles; `materialize()` produces a
  `WarmupCache` with a uniform `.get(name, timestamp, symbol)` interface
  plus `.tradeable`, `.liquid`, `.universe` gate queries.
- Three modes (`WarmupMode.EAGER` / `LAZY` / `OFF`) trade memory vs
  setup cost, with a per-indicator `cache=True|False|None` override for
  mixed slow-moving-cached + fast-moving-live strategies.

## v0.4.2

- `kuant.backtest.fill` and `kuant.backtest.position` land.
- `fill` ships an `Order` dataclass with explicit `OrderSide` /
  `OrderType` / `OrderStatus` enums, `FillReport` for reconciliation,
  and `submit_order` routing MARKET orders through the liquidity layer.
- `position` ships `Position` (signed size, volume-weighted `avg_cost`,
  cumulative `realized_pnl`) with netting semantics, `PortfolioState`
  (cash + per-symbol positions with atomic fill application), and
  `EquitySnapshot` for mark-to-market reporting.

## v0.4.1

- `kuant.backtest.liquidity` lands with `LiquidityProfile` (ADV, spread,
  min_size, max_participation).
- Three fill models: `FlatSlippage`, `LinearImpact`, `SquareRootImpact`
  (Almgren-Chriss).
- `execute_fill` + `execute_fill_panel` with categorical `FillResult`
  reasons (`OK`, `CAPPED_PARTICIPATION`, `BELOW_MIN_SIZE`,
  `NO_LIQUIDITY`, `MISSING_DATE`).
- `liquidity_mask` composes with lifecycle's `tradeable_mask` via `&`.

## v0.4.0

- **Breaking change (with deprecation shim):** lifecycle moves from
  `kuant.lifecycle` to `kuant.backtest.lifecycle` under the new
  `kuant.backtest` umbrella for correctness-first backtest primitives.
- `kuant.lifecycle` remains as a deprecation shim through the 0.4.x line
  and is removed in 0.5.0. Update `from kuant.lifecycle import X` to
  `from kuant.backtest.lifecycle import X`.

## v0.3.2

- `rollemastd` picks up the shifted-cumsum fix that eliminates
  catastrophic cancellation on near-constant inputs.
- Ships alongside the adversarial numerical-stability test suite at
  `tests/stats/test_numerical_stability.py` covering near-constant
  series, large offsets, long slow drifts, alternating signs, higher
  moments, and pandas-parity guardrails.

## v0.3.1

- Lifecycle primitives land: `SecurityLifecycle`, `TerminalAction`,
  `apply_lifecycle`, `tradeable_mask`, `lifecycle_returns`,
  `detect_delistings`.
- Paired identifier scrub in `kuant.text` for tickernorm plus CUSIP
  validation.

## v0.3.0 (yanked)

- `kuant.text.tickernorm`.
- `kuant.signals.factorscoring`.
- Tearsheet parity across `kuant.portfolio`.
- `kuant.stats.realizedvol`.
- `kuant.stats.stationarity`.
- Initial `kuant.nulltest` cluster (bootstrap, MHT correction, SPA test).

**Yanked:** contained non-public identifiers in test-fixture strings;
replaced by v0.3.1.

## v0.2.0

- Portfolio v1: drawdown, sharperatio, sortinoratio, contribution.
- Text v1: occparse, secformparse, cusipvalidate,
  KuantEncodingError/Warning.
- CI publish workflow with PyPI Trusted Publishing on tag push.

## v0.1.0

- Initial release with `kuant.core` (Black-Scholes family, Gaussian and
  Student-t primitives, GPD, logsumexp) and `kuant.options` (Greeks,
  payoffs, chain filters, implied-vol solvers).
