"""kuant.qm.hmm — hidden Markov model inference (discrete observations).

Log-space implementations of the four standard HMM algorithms:
  forward   — likelihood P(O | model) and forward variable α
  backward  — backward variable β
  viterbi   — most likely state sequence
  posterior — state posteriors γ and joint state posteriors ξ
  baumwelch — EM parameter training (π, A, B) via forward-backward
"""

from .backward import backward
from .baumwelch import BaumWelchResult, baumwelch
from .forward import forward
from .posterior import posterior
from .viterbi import viterbi

__all__ = [
    "BaumWelchResult",
    "backward",
    "baumwelch",
    "forward",
    "posterior",
    "viterbi",
]
