'''kuant.stats — rolling and windowed statistical primitives.'''
from .rollmean import rollmean
from .rollstd import rollstd
from .zscore import zscore

__all__ = ['rollmean', 'rollstd', 'zscore']
