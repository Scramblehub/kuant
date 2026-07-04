# Scope drafts for the five empty subpackages

Working scopes for `kuant.text`, `kuant.signals`, `kuant.portfolio`,
`kuant.edgecases`, and `kuant.data`. These are proposals, not
commitments — each one is a candidate list of kernels with a
one-paragraph rationale, dependency notes, and a "first three"
recommendation so we can ship a v1 in a single session per
subpackage without over-scoping.

Each subpackage should be shippable under the same contracts we've
locked in for `kuant.stats` / `kuant.topology` / `kuant.qm`:

- **Kernel-name discipline**: no underscores, one word (`baragg`,
  `occparse`, `sharperatio`). Compound-word run-together names read
  as one thing.
- **Backend-preserving numpy/cupy dispatch** where the math is batched.
- **Full validator suite from the start**: shape, value-range, NaN,
  mutex, deps.
- **Runtime warnings** for known-unreliable regimes.
- **Lazy heavy deps**: text parsing may want `pandas` for output
  scaffolding, but that must not be a hard dep of `kuant.text`.
- **Benchmarks in `benchmarks/suites/bench_<subpkg>.py`**.

---

## `kuant.text`

Financial text parsing and normalization. Not NLP — this is about the
deterministic surface-level parsing you do BEFORE any model touches
the text. Reference implementations exist in a dozen private repos;
kuant ships the canonical ones once.

### Scope

| Kernel | Purpose | Notes |
|---|---|---|
| `occparse(symbol)` | Parse an OCC option symbol → `(underlying, expiry, C/P, strike)` | Pure regex + date math, no deps. Handles both root+expiry ("AAPL240119C150000") and dashed variants. |
| `secformparse(form)` | Parse SEC form types + filing headers into normalized dict | `10-K`, `10-Q`, `8-K/A`, `S-1/A`, plus header lines (CIK, accession, period). Pure string work. |
| `lmdict(text, dictionary='LM2018')` | Loughran-McDonald sentiment tally with configurable dictionary version | Positive/negative/uncertainty/litigious/etc counts + normalized per-1000-word rates. Ships one dictionary in-tree (LM 2018); user can pass their own. |
| `cusipvalidate(cusip)` | CUSIP checksum validation + normalization | Detects malformed CUSIPs from data-vendor exports. Modulo-10 check digit. Also `isinvalidate`. |
| `tickernorm(symbol, exchange=None)` | Normalize a ticker across venue conventions | `BRK.B` ↔ `BRK-B` ↔ `BRK/B`. Deterministic; venue-aware when the exchange is passed. |
| `filingdate(header)` | Extract a canonical filing date + reporting period end from an EDGAR header block | Depends on `secformparse`. |

### Dependencies

Pure Python / stdlib for all six. `re`, `datetime`, `dataclasses`.
No pandas dep. Return types are dataclasses (like
`PersistenceDiagram`) so users can hydrate to their own DataFrame flavor.

### First three (recommended v1)

1. `occparse` — everyone parses OCC symbols, everyone gets it wrong at least once.
2. `secformparse` — high-leverage: unlocks any EDGAR-adjacent kernel.
3. `cusipvalidate` — cheap, common, catches a real class of vendor bugs.

Ship these three + a `TextParseResult` dataclass pattern that the
others will reuse. `lmdict`, `tickernorm`, `filingdate` follow in v1.1.

### Warnings / errors specific to this subpackage

- `KuantValueError` on malformed input for the parsers.
- `KuantNumericWarning` on `lmdict` when the tokenized text has
  fewer than N words (dictionary counts become noisy).

---

## `kuant.signals`

Cross-sectional and time-series signal primitives. **Not full strategies**
— just the mechanical transforms you compose into a signal, with the
subtleties (winsorization, industry-neutralize, IC-decay) handled
correctly.

### Scope

| Kernel | Purpose | Notes |
|---|---|---|
| `breadthscore(returns_panel, window, quantile)` | Cross-sectional breadth: fraction of names above their `window`-day quantile | Boolean threshold with configurable quantile. Panel input. |
| `momentumfamily(prices, family='carhart')` | Standard momentum family (Jegadeesh-Titman, Carhart, Asness) | `family` picks the exact lookback / skip / holding rule. Returns per-name z-score. |
| `winsorize(x, lo=0.01, hi=0.99, per_row=True)` | Winsorize a panel or series to given quantiles | The primitive everyone re-implements; get it right once. Per-row for cross-sectional; per-column axis flip. |
| `neutralize(signal, factors)` | OLS residual after regressing signal on factors | Industry-neutralize, size-neutralize, factor-neutralize. `factors` is a `(T, K)` design matrix or a dict of named series. |
| `icdecay(signal, forward_returns, horizons)` | Information Coefficient decay curve across forecast horizons | Spearman IC at each horizon; also information ratio via t-stat. |
| `signalcombine(signals, weights='equal'\|'ir'\|'ridge', rebalance=21)` | Combine k signals into a composite with a chosen weighting scheme | Rolling-fit weights (IR, ridge on IC covariance). |

### Dependencies

`scipy.stats` for rank/Spearman correlations. Lazy `sklearn` for
`signalcombine(weights='ridge')`.

### First three (recommended v1)

1. `winsorize` — used by literally every other signal kernel.
2. `neutralize` — the OLS-residual primitive; unblocks the family.
3. `icdecay` — critical diagnostic; the tool that catches "you found
   noise, not signal" faster than any backtest.

`breadthscore`, `momentumfamily`, `signalcombine` in v1.1.

### Warnings / errors

- `KuantShapeError` on panel misalignment.
- `KuantNumericWarning` on `neutralize` when the `factors` matrix has
  condition number > 1e10 (near-collinear, regression unstable).
- `KuantNumericWarning` on `icdecay` when the IC standard error at
  the chosen horizon exceeds the IC itself (indistinguishable from
  noise at this sample size).

---

## `kuant.portfolio`

Position-and-P&L primitives. **Not a backtester** — that's
`kuant.backtest`. This subpackage handles the arithmetic you do
AFTER a backtest returns positions and prices.

### Scope

| Kernel | Purpose | Notes |
|---|---|---|
| `sharperatio(returns, ann_factor=252, rf=0.0)` | Full-history Sharpe with an annualization factor | Scalar output. `rollsharpe` exists in `kuant.stats` for rolling. |
| `drawdown(equity)` | Peak-to-trough drawdown series + max drawdown scalar | Returns a `DrawdownResult` dataclass with `series`, `max_dd`, `peak_dates`, `trough_dates`. |
| `calmarratio(returns, ann_factor=252)` | CAGR / \|MaxDD\| | Trivial composition on top of `sharperatio` + `drawdown`. |
| `turnover(positions, prices=None)` | Rolling position turnover (% of capital rotated) | `sum(abs(diff(positions)))/sum(abs(positions))` per rebal. Weight- or notional-based. |
| `contribution(positions, returns, group=None)` | Per-asset (or per-group) P&L attribution | Returns per-name P&L series + aggregated contribution to the total return series. |
| `costadjust(returns, positions, cost_bps)` | Apply linear transaction cost to returns given position changes | Simplest reasonable slippage model; users bring their own for more. |
| `exposure(positions, factor_loadings)` | Portfolio exposure to a set of factor loadings over time | `(T, N) @ (N, K) → (T, K)`. Time-varying loadings supported. |

### Dependencies

Pure numpy. Optional pandas index preservation via a `.attach_index()`
helper on the result dataclasses — no hard dep.

### First three (recommended v1)

1. `drawdown` — universal, always the second call after computing returns.
2. `sharperatio` — classic. Sanity-checking must have this.
3. `contribution` — hardest to get right by hand; highest per-line value.

`calmarratio`, `turnover`, `costadjust`, `exposure` in v1.1.

### Warnings / errors

- `KuantValueError` on non-finite returns.
- `KuantNumericWarning` on `sharperatio` with `len(returns) < 30`
  (Sharpe estimate is dominated by noise below this sample size).
- `KuantNumericWarning` on `contribution` when positions and returns
  don't fully cover the same date range (partial coverage flagged).

---

## `kuant.edgecases`

**Meta-subpackage: strategies as first-class objects.** These are the
"how do you handle NaN" and "how do you handle delisted names" questions
that every backtest re-implements. Ship them as callable strategy
objects so users compose behavior instead of re-writing it.

### Scope

| Kernel | Purpose | Notes |
|---|---|---|
| `nanpolicies` | Callable NaN-handling strategies: `strict`, `skipna`, `forwardfill`, `interpolate`, `dropcolumn` | Each is a callable `policy(x) -> x'`. `strict` re-raises via `require_nonnan`. |
| `delistedhandling` | Utilities for delisted-name handling in historical backtests: `zero_after_delist`, `hold_last_price`, `full_recovery_check` | The set of gotchas that make historical backtests overstate returns. |
| `holidaycalendar(exchange='XNYS')` | Trading-day calendars for a set of major exchanges | `is_trading_day`, `next_trading_day`, `days_between`. Uses `pandas_market_calendars` when installed (lazy), falls back to a bundled subset for XNYS/XLON/XTKS. |
| `stalepricehandling(prices, staleness_threshold)` | Flag stale prints in intraday panels | Zero-diff sequences longer than N ticks → flagged. |
| `outlierpolicy(x, method='mad'\|'iqr'\|'zscore', threshold)` | Universal outlier detection with pluggable method | Returns boolean mask. Companion to `winsorize`. |
| `earlywarning(x, rule)` | Composable early-warning rules on a series ("consecutive N days below Z-score, etc.") | Rule DSL for canned regime-transition detectors. |

### Dependencies

`pandas_market_calendars` (lazy) for exchange calendars. Otherwise
pure numpy.

### First three (recommended v1)

1. `nanpolicies` — the taxonomy paper says this belongs first; it's
   the callable object that everything else can compose with.
2. `delistedhandling` — high-value: a single wrong choice here inflates
   backtested CAGR by 2-5pp.
3. `outlierpolicy` — general utility, downstream of everything else.

`holidaycalendar`, `stalepricehandling`, `earlywarning` in v1.1.

### Warnings / errors

- `KuantValueError` on `nanpolicies.strict` when NaN is present.
- `KuantNumericWarning` on `delistedhandling.hold_last_price` when
  the held span exceeds 20 days (this is the "phantom equity"
  pattern that overstates backtested returns).

---

## `kuant.data`

Data-shape primitives. **Not I/O** — no HTTP, no file readers, no
Sharadar-specific parsers. The kernel scope is: given clean-ish
input, produce the shape you need for the rest of kuant.

### Scope

| Kernel | Purpose | Notes |
|---|---|---|
| `baragg(bars, freq='5min'\|'1D'\|'1W')` | Bar-frequency aggregation with OHLCV semantics | O = first, H = max, L = min, C = last, V = sum. Handles unaligned inputs. Returns dataclass with aligned arrays. |
| `align(*series, method='inner'\|'outer'\|'forward')` | Multi-series alignment on a common calendar/index | Returns tuple of aligned 1D arrays. `inner` drops mismatched dates; `outer` fills with NaN; `forward` fills forward on the shorter series. |
| `corpaction(prices, splits, dividends, mode='total_return'\|'split_only')` | Corporate-action-adjusted price series | Split adjustment always applied; dividends conditionally. Simple total-return calc for split+dividend. |
| `resample(series, from_freq, to_freq, method='last'\|'mean'\|'sum')` | Downsample OR upsample with explicit method | Distinct from `baragg` because this is for scalars/panels, not OHLCV. |
| `panelize(long_form_df, index_col, name_col, value_col)` | Long → wide panel conversion, returning `(T, N) ndarray + name index + date index` | The one pattern everyone re-implements from pandas. |
| `stitch(*panels, method='calendar_union'\|'first_wins')` | Stitch multiple partial-coverage panels into one wider panel | For merging vendor data with different coverage windows. |

### Dependencies

No hard deps. Optional pandas via a `.to_dataframe()` helper on the
result dataclasses.

### First three (recommended v1)

1. `align` — every downstream kernel benefits.
2. `baragg` — universal for any intraday-adjacent work.
3. `corpaction` — the correctness-critical one nobody wants to hand-write.

`resample`, `panelize`, `stitch` in v1.1.

### Warnings / errors

- `KuantShapeError` on `align` when the index dtype differs across
  series (e.g. mixing pandas Timestamps and integer positions).
- `KuantNumericWarning` on `corpaction` when a split ratio > 100 or
  < 0.001 (typical typo signature in dividend/split files).
- `KuantNumericWarning` on `baragg` when the target frequency contains
  fewer than 5 input bars per output bar (aggregation is noisy).

---

## Sequencing recommendation

The subpackages have real dependencies among themselves.
Build order for lowest churn:

```
1. kuant.data        — no dependencies on the others; unblocks everyone.
2. kuant.edgecases   — depends on data alignment; used by portfolio + signals.
3. kuant.signals     — depends on data + edgecases (winsorize uses nanpolicies).
4. kuant.portfolio   — depends on signals (contribution wants factor exposure).
5. kuant.text        — independent; can slot in anywhere.
```

Ship one v1 (3 kernels + tests + benchmarks + `__init__` exports)
per session. Five sessions = five subpackages live.

## Notes on scope discipline

The pattern that has worked for us:
- **First-three cut**: pick the 3 highest-leverage kernels for v1;
  everything else in v1.1. Prevents the "we shipped 8 mediocre
  kernels because we didn't polish any of them" outcome.
- **Result dataclasses beat tuples**: every kernel that returns
  more than one value uses a dataclass with a `.summary()` method.
  Establishes the pattern users can rely on.
- **Warnings BEFORE errors during test writing**: tests that exercise
  a warning branch are as important as tests that exercise errors.
  Coverage of the warn-path is what makes the QOL claim real.
