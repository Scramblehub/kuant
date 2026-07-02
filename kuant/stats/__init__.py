'''kuant.stats — rolling and windowed statistical primitives.'''
from .rollcorr import rollcorr
from .rollmean import rollmean
from .rollquantile import rollmedian, rollpercentile, rollquantile
from .rollstd import rollstd
from .zscore import zscore

__all__ = [
    'rollcorr', 'rollmean', 'rollmedian', 'rollpercentile',
    'rollquantile', 'rollstd', 'zscore',
]
