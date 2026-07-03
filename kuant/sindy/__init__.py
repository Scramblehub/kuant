'''kuant.sindy — SINDy-adjacent reusable research tools.

Tools distilled from real production null-hypothesis testing:

  permtest    — universal null-hypothesis test (numpy only)
  grangerscan — Bonferroni-corrected Granger F-test scan (statsmodels)
  sindylasso  — LASSO-with-CV feature-library scan (sklearn)

All heavy deps (statsmodels, sklearn) are lazy-imported at CALL time,
not at module import.
'''
from .grangerscan import GrangerHit, GrangerScanResult
from .grangerscan import grangerscan as _grangerscan_fn
from .permtest import PermutationTestResult
from .permtest import permtest as _permtest_fn
from .sindylasso import SindyLassoResult
from .sindylasso import sindylasso as _sindylasso_fn

# Disambiguate module/function name collisions.
permtest = _permtest_fn
grangerscan = _grangerscan_fn
sindylasso = _sindylasso_fn

__all__ = [
    'grangerscan', 'GrangerHit', 'GrangerScanResult',
    'permtest', 'PermutationTestResult',
    'sindylasso', 'SindyLassoResult',
]
