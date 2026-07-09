"""kuant.signals: cross-sectional and time-series signal primitives.

Not full strategies: just the mechanical transforms that every quant
signal desk composes:

- `winsorize`: cap values at chosen quantiles. Cross-sectional
  (per-row) or time-series (per-column) modes.
- `neutralize`: OLS residual after regressing a signal on factors.
  Industry/size/factor-neutralize an alpha signal in one call. Warns
  on near-collinear factor sets.
- `icdecay`: Information Coefficient decay curve. Spearman IC at
  multiple forecast horizons, with per-horizon standard errors and
  t-stats. Warns when the IC at any horizon is indistinguishable
  from noise given the sample size.
"""

from kuant.signals.factorscoring import (
    FactorICResult,
    QuantileReturnsResult,
    QuantileSpreadResult,
    QuantileTurnoverResult,
    RankAutocorrResult,
    factor_ic,
    factor_rank_autocorr,
    mean_return_by_quantile,
    quantile_spread,
    quantile_turnover,
)
from kuant.signals.icdecay import ICDecayResult, icdecay
from kuant.signals.neutralize import NeutralizeResult, neutralize
from kuant.signals.winsorize import winsorize

# v0.6.0 batch 7: signal-processing transforms.
from kuant.signals.emd import EmdResult, emd
from kuant.signals.ica import IcaResult, ica
from kuant.signals.kernelpca import KernelPcaResult, kernelpca
from kuant.signals.wavelet import WaveletResult, wavelet
from kuant.signals.whitening import WhiteningResult, whitening

__all__ = [
    "FactorICResult",
    "ICDecayResult",
    "NeutralizeResult",
    "QuantileReturnsResult",
    "QuantileSpreadResult",
    "QuantileTurnoverResult",
    "RankAutocorrResult",
    "factor_ic",
    "factor_rank_autocorr",
    "icdecay",
    "mean_return_by_quantile",
    "neutralize",
    "quantile_spread",
    "quantile_turnover",
    "winsorize",
    # v0.6.0 batch 7: signal-processing transforms.
    "WhiteningResult",
    "whitening",
    "IcaResult",
    "ica",
    "KernelPcaResult",
    "kernelpca",
    "WaveletResult",
    "wavelet",
    "EmdResult",
    "emd",
]
