'''kuant.stats — rolling and windowed statistical primitives.'''
from .rollargminmax import rollargmax, rollargmin
from .rollcorr import rollcorr
from .rollcov import rollcov
from .rollema import rollema
from .rollmean import rollmean
from .rollminmax import rollmax, rollmin
from .rollmoments import rollkurt, rollskew
from .rollquantile import rollmedian, rollpercentile, rollquantile
from .rollrange import rollrange
from .rollrank import rollrank
from .rollstd import rollstd
from .rollsum import rollsum
from .zscore import zscore

__all__ = [
    'rollargmax', 'rollargmin', 'rollcorr', 'rollcov', 'rollema',
    'rollkurt', 'rollmax', 'rollmean', 'rollmedian', 'rollmin',
    'rollpercentile', 'rollquantile', 'rollrange', 'rollrank',
    'rollskew', 'rollstd', 'rollsum', 'zscore',
]
