"""kuant.stats — rolling and windowed statistical primitives."""

from .hurstrs import HurstResult, hurstrs
from .rollargminmax import rollargmax, rollargmin
from .rollbeta import rollbeta
from .rollcalmar import rollcalmar
from .rollcorr import rollcorr
from .rollcov import rollcov
from .rollema import rollema
from .rollemastd import rollemastd
from .rollhurst import rollhurst
from .rollidio import rollidio
from .rollmad import rollmad
from .rollmdd import rollmdd
from .rollmean import rollmean
from .rollminmax import rollmax, rollmin
from .rollmoments import rollkurt, rollskew
from .rollquantile import rollmedian, rollpercentile, rollquantile
from .rollrange import rollrange
from .rollrank import rollrank
from .rollsharpe import rollsharpe
from .rollsortino import rollsortino
from .rollstd import rollstd
from .rollsum import rollsum
from .zscore import zscore

__all__ = [
    "HurstResult",
    "hurstrs",
    "rollargmax",
    "rollargmin",
    "rollbeta",
    "rollcalmar",
    "rollcorr",
    "rollcov",
    "rollema",
    "rollemastd",
    "rollhurst",
    "rollidio",
    "rollkurt",
    "rollmad",
    "rollmax",
    "rollmdd",
    "rollmean",
    "rollmedian",
    "rollmin",
    "rollpercentile",
    "rollquantile",
    "rollrange",
    "rollrank",
    "rollsharpe",
    "rollsortino",
    "rollskew",
    "rollstd",
    "rollsum",
    "zscore",
]
