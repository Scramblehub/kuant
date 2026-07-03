'''kuant.sindy — SINDy-adjacent reusable research tools.

Tools distilled from real production null-hypothesis testing:

  permtest         — universal null-hypothesis test (numpy only)
  grangerscan      — Bonferroni-corrected Granger F-test scan (statsmodels)
  sindylasso       — LASSO-with-CV feature-library scan (sklearn)
  pinnscan         — nonlinear GBR feature-library scan + permutation null (sklearn)
  symbolicscan     — polynomial-symbolic regression scan (sklearn)
  accelerationscan — second-derivative predictive-power scan (numpy)

All heavy deps (statsmodels, sklearn) are lazy-imported at CALL time,
not at module import.
'''
from .accelerationscan import AccelerationScanResult
from .accelerationscan import accelerationscan as _accelerationscan_fn
from .grangerscan import GrangerHit, GrangerScanResult
from .grangerscan import grangerscan as _grangerscan_fn
from .permtest import PermutationTestResult
from .permtest import permtest as _permtest_fn
from .pinnscan import PinnScanResult
from .pinnscan import pinnscan as _pinnscan_fn
from .sindylasso import SindyLassoResult
from .sindylasso import sindylasso as _sindylasso_fn
from .symbolicscan import SymbolicScanResult
from .symbolicscan import symbolicscan as _symbolicscan_fn

# Disambiguate module/function name collisions.
permtest = _permtest_fn
grangerscan = _grangerscan_fn
sindylasso = _sindylasso_fn
pinnscan = _pinnscan_fn
symbolicscan = _symbolicscan_fn
accelerationscan = _accelerationscan_fn

__all__ = [
    'accelerationscan', 'AccelerationScanResult',
    'grangerscan', 'GrangerHit', 'GrangerScanResult',
    'permtest', 'PermutationTestResult',
    'pinnscan', 'PinnScanResult',
    'sindylasso', 'SindyLassoResult',
    'symbolicscan', 'SymbolicScanResult',
]
