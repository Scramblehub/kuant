'''kuant.qm.hmm — hidden Markov model inference (discrete observations).

Log-space implementations of the four standard HMM algorithms:
  forward   — likelihood P(O | model) and forward variable α
  backward  — backward variable β
  viterbi   — most likely state sequence
  posterior — state posteriors γ and joint state posteriors ξ
'''
from .backward import backward
from .forward import forward
from .posterior import posterior
from .viterbi import viterbi

__all__ = ['backward', 'forward', 'posterior', 'viterbi']
