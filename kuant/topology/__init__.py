"""kuant.topology — topological data analysis kernels.

Persistent-homology-based tools for time series and point clouds, plus
a cross-sectional dispersion-shape signal. Heavy dependencies (`ripser`,
`persim`) are lazily imported per-kernel so `import kuant.topology` stays
cheap on machines without them.

Install the full topology bundle:

    pip install kuant[topology]
"""

from kuant.topology.bettiseries import bettiseries
from kuant.topology.dispersioncollapse import dispersioncollapse
from kuant.topology.persistenthomology import PersistenceDiagram, persistenthomology
from kuant.topology.wasserstein import wasserstein

__all__ = [
    "PersistenceDiagram",
    "bettiseries",
    "dispersioncollapse",
    "persistenthomology",
    "wasserstein",
]
