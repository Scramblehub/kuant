"""kuant.signals — cross-sectional and time-series signal primitives.

Not full strategies — just the mechanical transforms that every quant
signal desk composes:

- `winsorize` — cap values at chosen quantiles. Cross-sectional
  (per-row) or time-series (per-column) modes.
- `neutralize` — OLS residual after regressing a signal on factors.
  Industry/size/factor-neutralize an alpha signal in one call. Warns
  on near-collinear factor sets.
- `icdecay` — Information Coefficient decay curve. Spearman IC at
  multiple forecast horizons, with per-horizon standard errors and
  t-stats. Warns when the IC at any horizon is indistinguishable
  from noise given the sample size.
"""

from kuant.signals.icdecay import ICDecayResult, icdecay
from kuant.signals.neutralize import NeutralizeResult, neutralize
from kuant.signals.winsorize import winsorize

__all__ = [
    "ICDecayResult",
    "NeutralizeResult",
    "icdecay",
    "neutralize",
    "winsorize",
]
