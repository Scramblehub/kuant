'''kuant.stats — rolling and windowed statistical primitives.'''
from .rollargminmax import rollargmax, rollargmin
from .rollbeta import rollbeta
from .rollcorr import rollcorr
from .rollcov import rollcov
from .rollema import rollema
from .rollemastd import rollemastd
from .rollidio import rollidio
from .rollmad import rollmad
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
    'rollargmax', 'rollargmin', 'rollbeta', 'rollcorr', 'rollcov',
    'rollema', 'rollemastd', 'rollidio', 'rollkurt', 'rollmad',
    'rollmax', 'rollmean', 'rollmedian', 'rollmin', 'rollpercentile',
    'rollquantile', 'rollrange', 'rollrank', 'rollskew', 'rollstd',
    'rollsum', 'zscore',
]
