'''kuant.qm.ghmm — Gaussian-emission hidden Markov model inference.

Same four algorithms as kuant.qm.hmm (forward / backward / viterbi /
posterior), but observations are continuous scalars and per-state
emissions are N(mu, sigma²).

This is what V8 regime work actually uses in production — returns are
continuous, not discrete symbols.
'''
from .backward import backward
from .forward import forward
from .posterior import posterior
from .viterbi import viterbi

__all__ = ['backward', 'forward', 'posterior', 'viterbi']
