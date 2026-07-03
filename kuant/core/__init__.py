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
from .normcdf import normcdf
from .normpdf import normpdf

__all__ = [
    "bscall", "bsput",
    "normcdf", "normpdf",
]
