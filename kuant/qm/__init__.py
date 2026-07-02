'''kuant.qm — QM-inspired tools for financial time series.

Submodules:
  hmm       — hidden Markov model inference (numpy/cupy)
  bell_test — reusable Bell-inequality-style aggregation test (module)

Direct exports:
  bell_test        — the function (from kuant.qm.bell_test)
  BellTestResult   — the dataclass returned by bell_test()

The bell_test function requires scikit-learn at CALL time. Importing
this module does not require sklearn.
'''
from . import hmm
from .bell_test import BellTestResult
from .bell_test import bell_test as _bell_test_fn

# Expose the function under a clear name, avoiding the module/function
# collision inherent in having a file named bell_test.py.
bell_test = _bell_test_fn

__all__ = ['bell_test', 'BellTestResult', 'hmm']
