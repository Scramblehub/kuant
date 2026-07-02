'''kuant.stats — rolling and windowed statistical primitives.'''
from .rollcorr import rollcorr
from .rollmean import rollmean
from .rollminmax import rollmax, rollmin
from .rollquantile import rollmedian, rollpercentile, rollquantile
from .rollrank import rollrank
from .rollstd import rollstd
from .rollsum import rollsum
from .zscore import zscore

__all__ = [
    'rollcorr', 'rollmax', 'rollmean', 'rollmedian', 'rollmin',
    'rollpercentile', 'rollquantile', 'rollrank', 'rollstd',
    'rollsum', 'zscore',
]
