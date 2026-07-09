"""Chaos-theory diagnostics for time-series.

Public surface:

- `mutualinfo(x, ...)`: embedding-delay picker (auto-MI curve or cross-MI).
- `falsenearest(x, tau, ...)`: embedding-dimension picker.
- `lyapunov(x, tau, m, ...)`: largest Lyapunov exponent (Rosenstein 1993).
- `corrdim(x, tau, m, ...)`: correlation dimension (Grassberger-Procaccia 1983).
- `rqa(x, tau, m, ...)`: recurrence-quantification analysis.
- `ccm(x, y, tau, m, ...)`: convergent cross-mapping (Sugihara 2012).
- `chaosscan(x, ...)`: composer: full battery + regime classification.

Result dataclasses (`MutualInfoResult`, `FalseNearestResult`,
`LyapunovResult`, `CorrDimResult`, `RQAResult`, `CCMResult`,
`ChaosScanResult`) are also exported for typing and downstream use.
"""

from __future__ import annotations

from kuant.sindy.chaos.ccm import CCMResult, ccm
from kuant.sindy.chaos.chaosscan import ChaosScanResult, chaosscan
from kuant.sindy.chaos.corrdim import CorrDimResult, corrdim
from kuant.sindy.chaos.crossrecurrence import (
    CrossRecurrenceResult,
    JointRecurrenceResult,
    crossrecurrence,
    jointrecurrence,
)
from kuant.sindy.chaos.embedding import (
    FalseNearestResult,
    MutualInfoResult,
    falsenearest,
    mutualinfo,
)
from kuant.sindy.chaos.entropy import (
    ApproximateEntropyResult,
    DispersionEntropyResult,
    PermutationEntropyResult,
    SampleEntropyResult,
    TransferEntropyResult,
    approximateentropy,
    dispersionentropy,
    permutationentropy,
    sampleentropy,
    transferentropy,
)
from kuant.sindy.chaos.lyapunov import (
    KnnLyapunovResult,
    LyapunovResult,
    knnlyapunov,
    lyapunov,
)
from kuant.sindy.chaos.rqa import RQAResult, rqa

__all__ = [
    # embedding
    "MutualInfoResult",
    "FalseNearestResult",
    "mutualinfo",
    "falsenearest",
    # Lyapunov
    "LyapunovResult",
    "KnnLyapunovResult",
    "lyapunov",
    "knnlyapunov",
    # correlation dimension
    "CorrDimResult",
    "corrdim",
    # RQA + cross / joint recurrence
    "RQAResult",
    "CrossRecurrenceResult",
    "JointRecurrenceResult",
    "rqa",
    "crossrecurrence",
    "jointrecurrence",
    # CCM
    "CCMResult",
    "ccm",
    # entropy family
    "PermutationEntropyResult",
    "SampleEntropyResult",
    "ApproximateEntropyResult",
    "DispersionEntropyResult",
    "TransferEntropyResult",
    "permutationentropy",
    "sampleentropy",
    "approximateentropy",
    "dispersionentropy",
    "transferentropy",
    # composer
    "ChaosScanResult",
    "chaosscan",
]
