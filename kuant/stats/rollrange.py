'''Rolling window range = rollmax - rollmin.

Trivial composition, kept as a named primitive for readability and
because "range" is often the natural signal (e.g. ATR-style
volatility, breakout detection).

Design: docs/kernels/rollrange.md.
'''
from __future__ import annotations

from .rollminmax import rollmax, rollmin


def rollrange(x, window):
    '''Rolling window range: `rollmax(x, w) - rollmin(x, w)`.

    NaN policy, dtype, backend all inherit from rollminmax.

    Examples
    --------
    >>> import numpy as np
    >>> rollrange(np.array([3.0, 1, 4, 1, 5, 9, 2, 6]), 3)
    array([nan, nan,  3.,  3.,  4.,  8.,  7.,  7.])
    '''
    return rollmax(x, window) - rollmin(x, window)
