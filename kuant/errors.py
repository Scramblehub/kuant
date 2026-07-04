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


# ---------- warnings --------------------------------------------------------
#
# Errors are for "we can't continue". Warnings are for "we're returning
# something, but you should know it may be unreliable". Distinct hierarchy
# so users can filter with `warnings.filterwarnings('error', category=...)`
# to convert to hard failure without touching the kernel.
#
# Every warning follows the same message shape as errors:
#     kuant.<kernel>: <what happened>.  [<code>]
#       → Fix: <remedy>.
# Codes start with `KW-` to distinguish from error codes (`KE-`).


class KuantWarning(UserWarning):
    """Base class for every warning kuant emits.

    Subclasses:
      KuantConvergenceWarning — solver returned a result but did not converge.
      KuantNumericWarning     — result is likely unreliable due to input
                                constraints (few samples, degenerate estimate,
                                CV endpoint hit, etc.).
    """


class KuantConvergenceWarning(KuantWarning):
    """Iterative solver hit `max_iter` without meeting `tol`.

    Distinct from `KuantConvergenceError`: raised by kernels that
    deliberately return a partial fit rather than failing outright
    (e.g. Baum-Welch, which surfaces `converged=False` on the result).
    """


class KuantNumericWarning(KuantWarning):
    """A numeric result was computed but is likely unreliable.

    Examples: LASSO CV picked the endpoint of the alpha grid (search
    range too narrow), Hill tail estimate went negative, persistent
    homology on <20 points.
    """


class KuantOverflowWarning(KuantWarning):
    """A computation overflowed or underflowed to ±inf.

    Distinct from `KuantNumericWarning` because the failure mode is
    structural (input magnitude), not statistical (input pattern). The
    result is typically `inf`, `-inf`, or `nan`; callers who want to
    hard-fail on this can promote:

        warnings.filterwarnings("error", category=KuantOverflowWarning)

    Examples: Black-Scholes on S=1e20 (exp(...) → inf), log-space
    accumulator hitting float64's finite range.
    """


__all__ = [
    "KuantError",
    "KuantValueError",
    "KuantShapeError",
    "KuantConvergenceError",
    "KuantBackendError",
    "KuantDependencyError",
    "KuantWarning",
    "KuantConvergenceWarning",
    "KuantNumericWarning",
    "KuantOverflowWarning",
]
