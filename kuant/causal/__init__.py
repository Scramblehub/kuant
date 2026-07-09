"""kuant.causal: causal-inference kernels.

v0.6.0 batch 10:
  - synthcontrol : Abadie-Diamond-Hainmueller synthetic control
  - iv           : Two-stage least squares IV
  - rdd          : Sharp regression discontinuity (local linear)
  - pcalgo       : PC algorithm skeleton (Fisher-Z CI test)
"""

from .iv import IvResult, iv
from .pcalgo import PcAlgoResult, pcalgo
from .rdd import RddResult, rdd
from .synthcontrol import SynthControlResult, synthcontrol

__all__ = [
    "synthcontrol",
    "SynthControlResult",
    "iv",
    "IvResult",
    "rdd",
    "RddResult",
    "pcalgo",
    "PcAlgoResult",
]
