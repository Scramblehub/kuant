# kuant errors and warnings

This is the canonical index of every stable error and warning code
kuant raises. Codes are stable across releases; message wording may
be tuned per release.

Errors follow the pattern `KE-<CATEGORY>-<DETAIL>`. They inherit from
`KuantValueError`, `KuantShapeError`, `KuantConvergenceError`,
`KuantBackendError`, or `KuantDependencyError` (see
[`kuant/errors.py`](../../kuant/errors.py) for the full hierarchy).

Warnings follow the pattern `KW-<CATEGORY>-<DETAIL>`. They inherit
from `KuantWarning` (base), `KuantConvergenceWarning`,
`KuantNumericWarning`, `KuantOverflowWarning`, `KuantEncodingWarning`,
or `KuantDeprecationWarning`.

To promote any warning to an exception:

    import warnings
    from kuant.errors import KuantNumericWarning
    warnings.filterwarnings("error", category=KuantNumericWarning)

Kernel columns list the module files that raise or emit the code. When
a code is emitted from a shared helper in `kuant/_validation.py`
(`require_1d`, `require_positive`, `require_range`, `did_not_converge`,
etc.), the file is named as the actual raise site; downstream kernels
routing through the helper inherit the same code.

## Errors

| Code | Class | Kernel(s) | Trigger |
|------|-------|-----------|---------|
| KE-CONV-DEGENERATE | KuantValueError | stats/dfa, stats/hurstrs | fewer than 3 windows produced finite log-log points, cannot fit the exponent |
| KE-CONV-MAX-ITER | KuantConvergenceError | `did_not_converge` helper in `_validation.py` (widely used by iterative solvers) | iterative solver hit max_iter without meeting tol |
| KE-CONV-MONOTONE | KuantValueError | qm/hmm/baumwelch, qm/ghmm/baumwelch | Baum-Welch log-likelihood decreased between iterations, violating Baum's inequality |
| KE-CORP-SPLIT-NONPOSITIVE | KuantValueError | data/corpaction | split ratio is zero or negative |
| KE-DEP-MISSING | KuantValueError, and KuantDependencyError via `require_dep` helper | backtest/warmup/warmup, backtest/warmup/cache, backtest/liquidity/profile, `_validation.py` helper | optional third-party dependency (pandas, sklearn, statsmodels, ripser, etc.) is not installed |
| KE-ENCODING-BYTES | KuantEncodingError | text/occparse | `bytes` passed where `str` was expected; text kernels refuse to guess the encoding |
| KE-FILL-SIZE-NAN | KuantValueError | backtest/liquidity/execute | requested fill size is NaN |
| KE-LIQ-MASK-MIN-ADV-NEGATIVE | KuantValueError | backtest/liquidity/execute | `min_adv` threshold is negative |
| KE-ORDER-LIMIT-INVALID | KuantValueError | backtest/fill/order | limit price is non-positive or non-finite |
| KE-PORTFOLIO-FILL-PRICE-INVALID | KuantValueError | backtest/position/portfolio | fill price is non-positive or non-finite when applied to a portfolio update |
| KE-POS-PRICE-INVALID | KuantValueError | backtest/position/position | fill price is non-positive |
| KE-POS-SIZE-INVALID | KuantValueError | backtest/position/position | filled size is non-finite |
| KE-SHAPE-1D | KuantShapeError | `require_1d` helper in `_validation.py` (widely used by 1D kernels) | array is not 1D |
| KE-SHAPE-2D | KuantShapeError | `require_2d` helper in `_validation.py` (widely used by 2D kernels) | array is not 2D |
| KE-SHAPE-EQUAL-LEN | KuantShapeError | data/align, data/resample, backtest/warmup/warmup, backtest/liquidity/profile, nulltest/spa_test + 3 others, plus `require_equal_length` helper | two arrays that must share length do not (OHLC bars, factor and returns, name lists) |
| KE-SHAPE-EXPECTED | KuantShapeError | backtest/engine/engine, backtest/lifecycle/security, backtest/liquidity/execute, backtest/warmup/warmup, data/align + 16 others, plus `require_expected_shape` helper | input array does not match the expected shape or dimensionality |
| KE-SUBMIT-UNKNOWN-REASON | KuantValueError | backtest/fill/submit | reject reason returned by the submit layer is not in the recognized set |
| KE-VAL-CONTRACT | KuantValueError | backtest/engine/engine, backtest/fill/order, backtest/fill/submit, backtest/liquidity/execute, qm/belltest, qm/decoherencescan | caller-supplied object does not satisfy the kernel's protocol or invariant (fill model, order layer, decoherence window) |
| KE-VAL-DUPLICATE | KuantValueError | data/align, data/panelize, backtest/warmup/warmup | duplicate row, index label, or lifecycle name |
| KE-VAL-EMPTY | KuantValueError | portfolio/drawdown, options/deltabucket, qm/belltest, qm/quaternion/composerotations, sindy/sindylasso, plus `require_non_empty` helper | input array or container is empty |
| KE-VAL-FINITE | KuantValueError | portfolio/riskmetrics, portfolio/sharperatio, portfolio/sortinoratio, qm/ghmm/baumwelch, sindy/permtest, plus `require_finite` helper | input contains NaN or infinity where finite values were required |
| KE-VAL-MIN-CLEAN | KuantValueError | nulltest/bootstrap, plus `require_min_clean` helper | too few finite rows remain after NaN drop to fit or estimate |
| KE-VAL-MISSING | KuantValueError | backtest/warmup/cache | expected indicator, symbol, or timestamp is absent from the warmup cache |
| KE-VAL-MUTEX | KuantValueError | data/corpaction, qm/hmm/baumwelch, qm/ghmm/baumwelch, plus `require_mutex_pair` helper | a mutually-exclusive pair of arguments has both set or neither set |
| KE-VAL-NAN | KuantValueError | edgecases/nanpolicies, plus `require_nonnan` helper | input contains NaN where the kernel forbids any NaN |
| KE-VAL-NAN-PVALUES | KuantValueError | nulltest/mht_correction | one or more p-values are NaN, multiple-hypothesis correction is undefined |
| KE-VAL-NONNEGATIVE | KuantValueError | `require_nonnegative` helper in `_validation.py` (widely used) | argument that must be non-negative is negative |
| KE-VAL-POSITIVE | KuantValueError | core/gpdcdf, core/gpdpdf, core/gpdppf, core/logtcdf, core/tcdf + 13 others, plus `require_positive` helper | argument that must be strictly positive is zero or negative |
| KE-VAL-PROBABILITY | KuantValueError | nulltest/mht_correction, plus `require_probability` helper | value that must be in `[0, 1]` is outside |
| KE-VAL-RANGE | KuantValueError | signals/factorscoring, signals/icdecay, options/impvolbisection, stats/rollmoments, stats/hurstrs + 33 others, plus `require_range` helper | value outside the allowed range, or invalid choice string in an enum-like argument |
| KE-VAL-SCHEMA | KuantValueError | backtest/liquidity/execute, backtest/position/portfolio | required column is missing from a DataFrame schema |
| KE-VAL-STOCHASTIC | KuantValueError | `require_stochastic` helper in `_validation.py` (HMM initial pi, probability vectors) | vector is not a probability distribution (entries outside `[0, 1]` or sum not equal to 1) |
| KE-VAL-STOCHASTIC-ROWS | KuantValueError | `require_stochastic_rows` helper in `_validation.py` (HMM transition and emission matrices) | matrix rows are not probability distributions |
| KE-VAL-TYPE | KuantValueError | backtest/warmup/warmup, backtest/lifecycle/security | argument has the wrong Python type (mode is not a string, date is not date-like, etc.) |
| KE-VAL-UNDERDET | KuantValueError | signals/neutralize, sindy/pinnscan, sindy/sindylasso, sindy/symbolicscan | `n_samples < 2 * n_features`, regression is underdetermined |
| KE-VAL-WINDOW | KuantValueError | `require_window` helper in `_validation.py` (all rolling kernels) | rolling window exceeds input length |
| KE-WARMUP-EMPTY-PANEL | KuantValueError | backtest/warmup/warmup | prices panel is empty when constructing Warmup |
| KE-WARMUP-INDICATOR-FAILED | KuantValueError | backtest/warmup/cache | user-supplied indicator callable raised inside WarmupCache |

## Warnings

| Code | Class | Kernel(s) | Trigger |
|------|-------|-----------|---------|
| KW-ALIGN-EMPTY-INTERSECT | KuantNumericWarning | data/align | shared index has zero entries after intersection |
| KW-BOOT-BLOCK-TOO-LONG | KuantNumericWarning | nulltest/bootstrap | stationary bootstrap `mean_block_length` is greater than or equal to `n` |
| KW-BOOT-LOW-N-BOOT | KuantNumericWarning | nulltest/bootstrap | `n_boot` is below the recommended minimum for stable p-value estimation |
| KW-CACHE-TS-NOT-IN-PANEL | KuantNumericWarning | backtest/warmup/cache | cached timestamp is not present in the supplied prices panel |
| KW-CACHE-UNIVERSE-UNKNOWN-SYMBOL | KuantNumericWarning | backtest/warmup/cache | cache entry names a symbol not in the current universe |
| KW-CAPTURE-NO-DOWN-PERIODS | KuantNumericWarning | portfolio/riskmetrics | benchmark has no down-market periods, downside capture is undefined |
| KW-CAPTURE-NO-UP-PERIODS | KuantNumericWarning | portfolio/riskmetrics | benchmark has no up-market periods, upside capture is undefined |
| KW-COLLINEAR-FACTORS | KuantNumericWarning | signals/neutralize | factor matrix is near-collinear, neutralization is unstable |
| KW-CONTRIB-PARTIAL-COVERAGE | KuantNumericWarning | portfolio/contribution | some names in weights are not present in returns (or vice versa) |
| KW-CONV-MAX-ITER | KuantConvergenceWarning in qm/hmm/baumwelch and qm/ghmm/baumwelch; KuantNumericWarning in options/impvol and options/impvolbisection | qm/hmm/baumwelch, qm/ghmm/baumwelch, options/impvol, options/impvolbisection | iterative solver hit `max_iter` without meeting `tol`, kernel returned a partial fit rather than raising |
| KW-CV-ENDPOINT-HIGH | KuantNumericWarning | sindy/sindylasso, sindy/symbolicscan | cross-validation picked the top of the alpha grid, search range likely too narrow |
| KW-CV-ENDPOINT-LOW | KuantNumericWarning | sindy/sindylasso, sindy/symbolicscan | cross-validation picked the bottom of the alpha grid, search range likely too narrow |
| KW-DEPRECATED-USE-LIFECYCLE | KuantDeprecationWarning | edgecases/delistedhandling | legacy kernel superseded by the lifecycle replacement |
| KW-DEPRECATION-MOVE | KuantDeprecationWarning | kuant/lifecycle (re-export shim) | `kuant.lifecycle` import path moved to `kuant.backtest.lifecycle`; old path still works but will be removed |
| KW-DIV-DEGENERATE | KuantNumericWarning | data/corpaction | dividend is greater than or equal to close price, total-return factor would be non-positive; the dividend is skipped |
| KW-DRAWDOWN-ALL-NAN | KuantNumericWarning | portfolio/drawdown | equity series is entirely NaN, drawdown is undefined |
| KW-DSR-NO-TRIALS | KuantNumericWarning | portfolio/riskmetrics | `n_trials <= 0` for Deflated Sharpe |
| KW-ENCODING-REPLACEMENT | KuantEncodingWarning | text/occparse | input contains U+FFFD replacement characters, upstream decode likely broken |
| KW-FIC-SKIPPED-PERIODS | KuantNumericWarning | signals/factorscoring | one or more periods produced no IC and were skipped |
| KW-FILL-PANEL-EMPTY | KuantNumericWarning | backtest/liquidity/execute | fill panel has zero rows |
| KW-FILL-PANEL-EXTRA-COLS | KuantNumericWarning | backtest/liquidity/execute | fill panel carries columns beyond the declared schema |
| KW-HILL-NEGATIVE | KuantNumericWarning | stats/tailindex | Hill tail-index estimate came out negative, tail is not heavy under the model |
| KW-HMM-SIGMA-FLOOR | KuantWarning | qm/ghmm/baumwelch | emission sigma hit its floor during EM |
| KW-HMM-STATE-COLLAPSE | KuantWarning | qm/hmm/baumwelch, qm/ghmm/baumwelch | one or more states received near-zero posterior mass across the sequence |
| KW-HMM-STATE-ORDER | KuantNumericWarning | qm/hmm/baumwelch, qm/ghmm/baumwelch | states were reordered post-fit to make the output deterministic |
| KW-IC-NOISE-FLOOR | KuantNumericWarning | signals/icdecay | cleaned IC value fell below the configured noise-floor threshold |
| KW-ICDECAY-NO-CLEAN | KuantNumericWarning | signals/icdecay | no periods survived the clean-row filter |
| KW-KELLY-NEGATIVE-EDGE | KuantNumericWarning | portfolio/riskmetrics | Kelly fraction is non-positive, no favorable bet |
| KW-KELLY-ZERO-VARIANCE | KuantNumericWarning | portfolio/riskmetrics | return variance collapsed to zero, Kelly is undefined |
| KW-LIFECYCLE-UNKNOWN-SYMBOL | KuantNumericWarning | backtest/lifecycle/security | symbol in prices has no entry in the lifecycle table |
| KW-LIQ-MASK-ALL-FALSE | KuantNumericWarning | backtest/liquidity/execute | liquidity mask excludes every name in the panel |
| KW-NEUTRALIZE-CONSTANT-SIGNAL | KuantNumericWarning | signals/neutralize | signal is constant, nothing to neutralize |
| KW-NUM-BUCKET-SMALL | KuantNumericWarning | qm/decoherencescan | cross-sectional bucket has too few observations for a stable estimate |
| KW-NUM-NO-MATCH | KuantNumericWarning | options/deltabucket | no option matched the requested delta bucket |
| KW-NUM-SAMPLE-SIZE | KuantNumericWarning | sindy/grangerscan | sample size is below the recommended minimum for the Granger test |
| KW-NUM-VEGA-DEGENERATE | KuantNumericWarning | options/impvol | vega collapsed to zero near the boundary of the implied-vol domain |
| KW-NUMERIC-ZERO-DENOMINATOR | KuantNumericWarning | `warn_zero_denominator` helper in `_validation.py` (Sharpe, Sortino, Kelly, Calmar) | risk-adjusted return denominator collapsed to zero, result is 0 or NaN by convention |
| KW-NUMERIC-ZERO-DOWNSIDE | KuantNumericWarning | stats/rollsortino | rolling downside std collapsed to zero |
| KW-NUMERIC-ZERO-DRAWDOWN | KuantNumericWarning | stats/rollcalmar | rolling drawdown is zero (no losses in window) |
| KW-NUMERIC-ZERO-STD | KuantNumericWarning | stats/rollsharpe | rolling std collapsed to zero (constant returns in window) |
| KW-OUTLIER-DEGENERATE | KuantNumericWarning | edgecases/outlierpolicy | winsorization or z-score cannot compute because the column is constant or all-NaN |
| KW-OUTLIER-EXTREME-RATE | KuantNumericWarning | edgecases/outlierpolicy | flagged outlier rate exceeds the configured threshold |
| KW-PHANTOM-EQUITY | KuantNumericWarning | edgecases/delistedhandling | symbol carries prices past its delisting date |
| KW-PORTFOLIO-NAN-MARK | KuantNumericWarning | backtest/position/portfolio | fill price is NaN when marking a position to market |
| KW-PSR-INVALID-MOMENTS | KuantNumericWarning | portfolio/riskmetrics | skew or kurtosis input to Probabilistic Sharpe is not finite |
| KW-PSR-SMALL-SAMPLE | KuantNumericWarning | portfolio/riskmetrics | sample too small for reliable Probabilistic Sharpe |
| KW-QUANTILE-THIN-BUCKETS | KuantNumericWarning | signals/factorscoring | too few names per quantile bucket for a stable bucket return |
| KW-RANK-CONSTANT-FACTOR | KuantNumericWarning | signals/factorscoring | factor is constant, cross-sectional rank is undefined |
| KW-SHARPE-CONSTANT-RETURNS | KuantNumericWarning | portfolio/sharperatio | returns std collapsed to zero, Sharpe is 0 by convention |
| KW-SHARPE-SMALL-SAMPLE | KuantNumericWarning | portfolio/sharperatio | sample too small for reliable Sharpe |
| KW-SORTINO-NO-DOWNSIDE | KuantNumericWarning | portfolio/sortinoratio | no downside observations in the sample |
| KW-SORTINO-SMALL-SAMPLE | KuantNumericWarning | portfolio/sortinoratio | sample too small for reliable Sortino |
| KW-SORTINO-TINY-DOWNSIDE | KuantNumericWarning | portfolio/sortinoratio | downside std is finite but near-zero, Sortino returned as 0 |
| KW-SPLIT-EXTREME | KuantNumericWarning | data/corpaction | split ratio is implausibly large or tiny |
| KW-STITCH-DISAGREE | KuantNumericWarning | data/stitch | finite values disagree between panels being stitched |
| KW-SURVIVOR-BIAS | KuantNumericWarning | edgecases/delistedhandling | filtered universe excludes delisted names, exposing survivorship bias |
| KW-TOPO-FEW-POINTS | KuantNumericWarning | topology/persistenthomology | point cloud has fewer than 20 points, persistence diagram is unreliable |
| KW-TURNOVER-DEGENERATE-FACTOR | KuantNumericWarning | signals/factorscoring | factor produced constant weights, turnover is zero by construction |
| KW-ULCER-ALL-NAN | KuantNumericWarning | portfolio/riskmetrics | equity curve is all-NaN, ulcer index is undefined |
| KW-VAL-DDOF-EXCEEDS-WINDOW | KuantNumericWarning | `warn_ddof_exceeds_window` helper in `_validation.py` (rolling std, rolling cov, all users) | `ddof` leaves zero or negative degrees of freedom for the rolling window, entire output is NaN |
| KW-VAL-INSUFFICIENT-TAIL | KuantNumericWarning | stats/tailindex | too few tail observations for a stable Hill estimator |
| KW-VAL-RANGE | KuantNumericWarning | options/impvolbisection | iterative solver's target lies outside the supplied bracket |
| KW-VAL-WINDOW-EXCEEDS-DATA | KuantNumericWarning | `warn_window_exceeds_data` helper in `_validation.py` (all rolling kernels) | rolling `window` exceeds input length `n`, entire output is NaN |
| KW-WINSORIZE-AGGRESSIVE-LIMITS | KuantNumericWarning | signals/winsorize | winsorize limits clip a large fraction of the data |
