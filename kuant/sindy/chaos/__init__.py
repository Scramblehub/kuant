"""Chaos-theory diagnostics for time-series.

Public surface:

- `mutualinfo(x, ...)` — embedding-delay picker (auto-MI curve or cross-MI).
- `falsenearest(x, tau, ...)` — embedding-dimension picker.
- `lyapunov(x, tau, m, ...)` — largest Lyapunov exponent (Rosenstein 1993).
- `corrdim(x, tau, m, ...)` — correlation dimension (Grassberger-Procaccia 1983).
- `rqa(x, tau, m, ...)` — recurrence-quantification analysis.
- `ccm(x, y, tau, m, ...)` — convergent cross-mapping (Sugihara 2012).
- `chaosscan(x, ...)` — composer: full battery + regime classification.

Result dataclasses (`MutualInfoResult`, `FalseNearestResult`,
`LyapunovResult`, `CorrDimResult`, `RQAResult`, `CCMResult`,
`ChaosScanResult`) are also exported for typing and downstream use.
"""

from __future__ import annotations

from kuant.sindy.chaos.ccm import CCMResult, ccm
from kuant.sindy.chaos.chaosscan import ChaosScanResult, chaosscan
from kuant.sindy.chaos.corrdim import CorrDimResult, corrdim
from kuant.sindy.chaos.embedding import (
    FalseNearestResult,
    MutualInfoResult,
    falsenearest,
    mutualinfo,
)
from kuant.sindy.chaos.lyapunov import LyapunovResult, lyapunov
from kuant.sindy.chaos.rqa import RQAResult, rqa

__all__ = [
    "MutualInfoResult",
    "FalseNearestResult",
    "LyapunovResult",
    "CorrDimResult",
    "RQAResult",
    "CCMResult",
    "ChaosScanResult",
    "mutualinfo",
    "falsenearest",
    "lyapunov",
    "corrdim",
    "rqa",
    "ccm",
    "chaosscan",
]
