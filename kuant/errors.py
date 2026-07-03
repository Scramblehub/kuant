"""kuant.errors — custom exception hierarchy.

All kuant-raised exceptions inherit from `KuantError`, which itself
inherits from `Exception`. This lets users write:

    try:
        result = kuant.some.kernel(...)
    except kuant.errors.KuantError as exc:
        # catch anything raised by kuant, ignore everything else
        ...

Subclasses match the standard-library exception they refine, so existing
code that catches `ValueError` or `RuntimeError` continues to work:

    KuantError            → Exception
    KuantValueError       → KuantError + ValueError
    KuantShapeError       → KuantValueError    (shape / broadcasting bugs)
    KuantConvergenceError → KuantError + RuntimeError  (solver / MC failed)
    KuantBackendError     → KuantError + RuntimeError  (numpy/cupy dispatch)
    KuantDependencyError  → KuantError + ImportError   (optional dep missing)

Design intent: users who care can catch specifically; users who don't
still see natural stdlib types in tracebacks.
"""
from __future__ import annotations


class KuantError(Exception):
    """Base class for every exception kuant raises directly."""


class KuantValueError(KuantError, ValueError):
    """A caller-supplied value violates a kernel's contract.

    Examples: negative volatility, non-positive tenor, empty input array
    to a rolling stat, probability outside [0, 1] to an inverse CDF.
    """


class KuantShapeError(KuantValueError):
    """Input arrays cannot be broadcast to a common shape.

    Distinct subclass of KuantValueError so callers can catch
    shape-specific issues separately from value-range issues.
    """


class KuantConvergenceError(KuantError, RuntimeError):
    """An iterative solver did not converge within tolerance.

    Raised by kernels that iterate (impvol Newton, POT/EVT fits,
    HMM EM re-estimation). Message should include the last iterate
    and the tolerance target so callers can decide whether to accept.
    """


class KuantBackendError(KuantError, RuntimeError):
    """A numpy/cupy backend routing decision failed.

    Rare — usually indicates a bug in `_detect_backend`, or the caller
    mixed numpy and cupy arrays in an unsupported way.
    """


class KuantDependencyError(KuantError, ImportError):
    """An optional dependency required for a specific kernel is missing.

    Kernels that rely on sklearn, statsmodels, ripser etc. import the
    dependency lazily and raise this when it isn't installed. The
    message should include the exact `pip install ...` remedy.
    """


__all__ = [
    "KuantError",
    "KuantValueError",
    "KuantShapeError",
    "KuantConvergenceError",
    "KuantBackendError",
    "KuantDependencyError",
]
