"""Test suite for kuant.errors — the custom exception hierarchy."""
from __future__ import annotations

import pytest

from kuant.errors import (
    KuantBackendError,
    KuantConvergenceError,
    KuantDependencyError,
    KuantError,
    KuantShapeError,
    KuantValueError,
)


def test_kuant_error_is_exception():
    assert issubclass(KuantError, Exception)


def test_value_error_dual_inheritance():
    """KuantValueError should be catchable as either KuantError or ValueError."""
    e = KuantValueError("bad value")
    assert isinstance(e, KuantError)
    assert isinstance(e, ValueError)


def test_shape_error_is_subclass_of_value_error():
    assert issubclass(KuantShapeError, KuantValueError)
    assert issubclass(KuantShapeError, ValueError)


def test_convergence_error_dual_inheritance():
    e = KuantConvergenceError("did not converge")
    assert isinstance(e, KuantError)
    assert isinstance(e, RuntimeError)


def test_backend_error_dual_inheritance():
    e = KuantBackendError("bad backend")
    assert isinstance(e, KuantError)
    assert isinstance(e, RuntimeError)


def test_dependency_error_dual_inheritance():
    e = KuantDependencyError("install scikit-learn")
    assert isinstance(e, KuantError)
    assert isinstance(e, ImportError)


def test_catch_all_via_kuant_error():
    """A user should be able to catch any kuant exception with KuantError alone."""
    for exc_cls in (KuantValueError, KuantShapeError, KuantConvergenceError,
                    KuantBackendError, KuantDependencyError):
        with pytest.raises(KuantError):
            raise exc_cls("test")


def test_stdlib_catches_still_work():
    """Existing user code catching ValueError etc. keeps working."""
    with pytest.raises(ValueError):
        raise KuantValueError("range")

    with pytest.raises(ValueError):
        raise KuantShapeError("bad broadcast")

    with pytest.raises(RuntimeError):
        raise KuantConvergenceError("didn't converge")

    with pytest.raises(ImportError):
        raise KuantDependencyError("no scipy")


def test_message_preserved_through_hierarchy():
    e = KuantValueError("volatility must be > 0")
    assert str(e) == "volatility must be > 0"
