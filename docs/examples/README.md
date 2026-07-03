# kuant examples

End-to-end usage patterns showing how to compose kernels for real tasks.

## Conventions

Each example lives in its own file and:

1. Imports only from kuant (no other quant deps)
2. Solves one specific task (price an option surface, backtest a signal,
   invert an IV curve, decode HMM states, ...)
3. Has a top-of-file comment explaining what it demonstrates
4. Runs on CPU by default; GPU acceleration is optional

## Planned examples

- `bs_price_surface.py` — vectorized BS pricing on a (strike, tenor) grid
- `iv_surface_from_market.py` — invert an option chain to an IV surface
- `rolling_zscore_signal.py` — z-score of returns as a mean-reversion signal
- `hmm_regime_decode.py` — Viterbi-decode market regimes on a return series
- `belltest_your_features.py` — run the classical-bound test on your own data
- `granger_signal_scan.py` — screen a macro library against your target

Currently empty — populate as tasks accumulate.
