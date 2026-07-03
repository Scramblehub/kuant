'''kuant.qm — QM-inspired tools for financial time series.

Submodules:
  hmm       — hidden Markov model inference (numpy/cupy)
  belltest — reusable Bell-inequality-style aggregation test (module)

Direct exports:
  belltest        — the function (from kuant.qm.belltest)
  BellTestResult   — the dataclass returned by belltest()

The belltest function requires scikit-learn at CALL time. Importing
this module does not require sklearn.
'''
from . import hmm
from .belltest import BellTestResult
from .belltest import belltest as _belltest_fn
from .zenoscan import ZenoScanResult, zenoscan

# Expose belltest the function under a clear name (belltest.py the
# module and belltest the function share a name; disambiguate here).
belltest = _belltest_fn

__all__ = ['belltest', 'BellTestResult', 'hmm', 'zenoscan', 'ZenoScanResult']
