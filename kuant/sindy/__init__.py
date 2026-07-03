'''kuant.sindy — SINDy-adjacent reusable research tools.

Tools distilled from real production null-hypothesis testing:

  permtest    — universal null-hypothesis test (numpy only)
  grangerscan — Bonferroni-corrected Granger F-test scan (statsmodels)

Both are safe to import; the statsmodels dep is checked at CALL time,
not import time. Same lazy-dep pattern as kuant.qm.
'''
from .grangerscan import GrangerHit, GrangerScanResult
from .grangerscan import grangerscan as _grangerscan_fn
from .permtest import PermutationTestResult
from .permtest import permtest as _permtest_fn

# Disambiguate module/function name collisions.
permtest = _permtest_fn
grangerscan = _grangerscan_fn

__all__ = [
    'grangerscan', 'GrangerHit', 'GrangerScanResult',
    'permtest', 'PermutationTestResult',
]
