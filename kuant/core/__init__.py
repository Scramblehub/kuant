"""kuant.core — mathematical primitives.

Foundational batched kernels that other categories build on. Nothing here
depends on anything else in kuant (except queueing for throttling).

Contains:
  - Base BS pricing: bscall, bsput
  - Gaussian primitives: normcdf, normpdf
  - Shared BS setup (imported by kuant.options Greeks): _bs_common

Greeks (delta, gamma, vega, rho, theta, charm) live in kuant.options
because they are option-specific quantities rather than general math.
"""
from .bscall import bscall
from .bsput import bsput
from .lognormccdf import lognormccdf
from .lognormcdf import lognormcdf
from .logsumexp import logsumexp
from .normcdf import normcdf
from .normpdf import normpdf
from .normppf import normppf
from .tcdf import tcdf
from .tpdf import tpdf
from .tppf import tppf

__all__ = [
    # Black-Scholes primitives
    "bscall", "bsput",
    # Gaussian family
    "normcdf", "normpdf", "normppf",
    "lognormcdf", "lognormccdf",
    # Student-t family (fat-tail)
    "tcdf", "tpdf", "tppf",
    # log-space arithmetic
    "logsumexp",
]
