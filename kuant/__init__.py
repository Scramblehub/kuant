"""kuant — GPU-accelerated quantitative research kernels.

kernel × quant.

Batched math primitives, options pricing, regime detection, sparse dynamics
identification, topological data analysis. Built for signal discovery, not
just textbook indicators.

See README for structure, install, design principles.
"""
import logging

# Version is read from installed package metadata when possible so it stays
# in sync with pyproject.toml. Falls back to a hard-coded value for source
# checkouts without installation.
try:
    from importlib.metadata import PackageNotFoundError, version as _dist_version

    try:
        __version__ = _dist_version("kuant")
    except PackageNotFoundError:  # pragma: no cover — source checkout path
        __version__ = "0.1.0"
except ImportError:  # pragma: no cover — Python < 3.8, shouldn't hit
    __version__ = "0.1.0"

__author__ = "Scramblehub"
__license__ = "Apache-2.0"

# Silent-by-default logging pattern for libraries. The application is
# responsible for configuring logging output (e.g. logging.basicConfig()).
# Without a handler, log records are dropped instead of printed to stderr.
#
# Users who WANT to see kuant's log records:
#     import logging
#     logging.basicConfig(level=logging.DEBUG)
#
# Or route specifically to a file / handler:
#     logging.getLogger("kuant").addHandler(logging.FileHandler("kuant.log"))
logging.getLogger(__name__).addHandler(logging.NullHandler())
