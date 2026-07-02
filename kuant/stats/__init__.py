'''kuant.stats — rolling and windowed statistical primitives.'''
from .rollcorr import rollcorr
from .rollmean import rollmean
from .rollstd import rollstd
from .zscore import zscore

__all__ = ['rollcorr', 'rollmean', 'rollstd', 'zscore']
