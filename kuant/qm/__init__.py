"""kuant.qm — QM-inspired tools for financial time series.

Submodules:
  hmm          — discrete-observation HMM inference
  ghmm         — Gaussian-observation HMM inference (continuous scalars)
  quaternion   — unit-quaternion algebra + rollholonomy + composerotations

Direct exports:
  belltest             — Bell-inequality-style aggregation test
  BellTestResult
  zenoscan             — retrain-frequency scan (Zeno effect)
  ZenoScanResult
  posteriorentropy     — Shannon entropy of an HMM posterior per bar
  PosteriorEntropyResult
  nocloningscan        — multi-seed model variance analysis
  NoCloningScanResult
  decoherencescan      — day-in-window skill decay diagnostic
  DecoherenceScanResult
"""

from . import ghmm, hmm, quaternion
from .belltest import BellTestResult
from .belltest import belltest as _belltest_fn
from .decoherencescan import DecoherenceScanResult
from .decoherencescan import decoherencescan as _decoherencescan_fn
from .nocloningscan import NoCloningScanResult
from .nocloningscan import nocloningscan as _nocloningscan_fn
from .posteriorentropy import PosteriorEntropyResult
from .posteriorentropy import posteriorentropy as _posteriorentropy_fn
from .zenoscan import ZenoScanResult
from .zenoscan import zenoscan as _zenoscan_fn

# Disambiguate module/function name collisions.
belltest = _belltest_fn
zenoscan = _zenoscan_fn
posteriorentropy = _posteriorentropy_fn
nocloningscan = _nocloningscan_fn
decoherencescan = _decoherencescan_fn

__all__ = [
    "hmm",
    "ghmm",
    "quaternion",
    "belltest",
    "BellTestResult",
    "zenoscan",
    "ZenoScanResult",
    "posteriorentropy",
    "PosteriorEntropyResult",
    "nocloningscan",
    "NoCloningScanResult",
    "decoherencescan",
    "DecoherenceScanResult",
]
