# Changelog

All notable changes to `kuant` are tracked here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); dates are release dates
on PyPI.

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
