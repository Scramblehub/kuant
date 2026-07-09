"""kuant.stats: rolling and windowed statistical primitives."""

from .dfa import DFAResult, dfa
from .hurstrs import HurstResult, hurstrs
from .rollcoherence import rollcoherence
from .rolltailindex import rolltailindex
from .tailindex import tailindex
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

# Realized-volatility estimators (Wave 3).
from .realizedvol import atr, garmanklass, parkinson, rogerssatchell, yangzhang

# Stationarity / unit-root tests (Wave 3). Lazy statsmodels/arch inside.
from .stationarity import (
    StationarityResult,
    adftest,
    kpsstest,
    phillipsperrontest,
    varianceratiotest,
)

# v0.6.0 batch 2: HAC standard errors + autocorr / normality / iid diagnostic tests.
from .autocorrtests import (
    DurbinWatsonResult,
    PortmanteauResult,
    boxpierce,
    durbinwatson,
    ljungbox,
)
from .bdstest import BdsResult, bdstest
from .hac import HacResult, andrewsse, neweywestse
from .normalitytests import (
    NormalityResult,
    andersondarling,
    jarquebera,
    shapirowilk,
)
from .spectralentropy import SpectralEntropyResult, spectralentropy

# v0.6.0 batch 4: Hurst family variants.
from .higuchihurst import HiguchiHurstResult, higuchihurst
from .localwhittle import LocalWhittleResult, localwhittle
from .mfdfa import MfdfaResult, mfdfa
from .wavelethurst import WaveletHurstResult, wavelethurst

# v0.6.0 batch 5: correlation variants.
from .correlations import (
    CorrelationResult,
    chatterjeexi,
    distancecorr,
    downsidecorr,
    kendalltau,
    spearmanrank,
)

__all__ = [
    "DFAResult",
    "dfa",
    "HurstResult",
    "hurstrs",
    "rollcoherence",
    "rolltailindex",
    "tailindex",
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
    # Wave 3 additions.
    "atr",
    "garmanklass",
    "parkinson",
    "rogerssatchell",
    "yangzhang",
    "StationarityResult",
    "adftest",
    "kpsstest",
    "phillipsperrontest",
    "varianceratiotest",
    # v0.6.0 batch 2.
    "HacResult",
    "neweywestse",
    "andrewsse",
    "PortmanteauResult",
    "DurbinWatsonResult",
    "ljungbox",
    "boxpierce",
    "durbinwatson",
    "NormalityResult",
    "jarquebera",
    "andersondarling",
    "shapirowilk",
    "BdsResult",
    "bdstest",
    "SpectralEntropyResult",
    "spectralentropy",
    # v0.6.0 batch 4: Hurst family variants.
    "HiguchiHurstResult",
    "higuchihurst",
    "WaveletHurstResult",
    "wavelethurst",
    "MfdfaResult",
    "mfdfa",
    "LocalWhittleResult",
    "localwhittle",
    # v0.6.0 batch 5: correlation variants.
    "CorrelationResult",
    "kendalltau",
    "spearmanrank",
    "distancecorr",
    "chatterjeexi",
    "downsidecorr",
]
