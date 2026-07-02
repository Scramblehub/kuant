'''kuant.stats — rolling and windowed statistical primitives.'''
from .rollmean import rollmean
from .rollstd import rollstd

__all__ = ['rollmean', 'rollstd']
