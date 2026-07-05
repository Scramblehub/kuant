"""Tests for the `kuant.lifecycle` deprecation shim.

The old import path resolves through 0.4.x for backwards compatibility,
scheduled for removal in 0.5.0. Importing the shim must (a) still yield
the correct symbols and (b) emit a `KuantDeprecationWarning`.
"""

from __future__ import annotations

import importlib
import sys
import warnings

import pytest

from kuant.errors import KuantDeprecationWarning


def _reload_lifecycle_shim():
    """Force a fresh import so the warning fires on THIS test's stack."""
    for mod in list(sys.modules):
        if mod == "kuant.lifecycle" or mod.startswith("kuant.lifecycle."):
            del sys.modules[mod]
    return importlib.import_module("kuant.lifecycle")


def test_shim_emits_deprecation_warning():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _reload_lifecycle_shim()
    kinds = [warn.category for warn in w]
    assert KuantDeprecationWarning in kinds


def test_shim_reexports_same_symbols():
    shim = _reload_lifecycle_shim()
    from kuant.backtest.lifecycle import (
        LifecyclePanelResult,
        SecurityLifecycle,
        TerminalAction,
        apply_lifecycle,
        apply_lifecycle_panel,
        detect_delistings,
        lifecycle_panel_report,
        lifecycle_returns,
        lifecycles_from_panel,
        tradeable_mask,
    )

    assert shim.SecurityLifecycle is SecurityLifecycle
    assert shim.TerminalAction is TerminalAction
    assert shim.LifecyclePanelResult is LifecyclePanelResult
    assert shim.apply_lifecycle is apply_lifecycle
    assert shim.apply_lifecycle_panel is apply_lifecycle_panel
    assert shim.lifecycle_returns is lifecycle_returns
    assert shim.tradeable_mask is tradeable_mask
    assert shim.lifecycle_panel_report is lifecycle_panel_report
    assert shim.detect_delistings is detect_delistings
    assert shim.lifecycles_from_panel is lifecycles_from_panel


def test_shim_submodule_reexports_resolve():
    """`from kuant.lifecycle.security import X` still works."""
    _reload_lifecycle_shim()
    from kuant.backtest.lifecycle.security import SecurityLifecycle as NewLC
    from kuant.lifecycle.security import SecurityLifecycle as ShimLC

    assert ShimLC is NewLC


def test_shim_can_be_promoted_to_error():
    """Users who want strict enforcement can flip the warning to an error."""
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=KuantDeprecationWarning)
        # Force a fresh import so the warning fires under the strict filter.
        for mod in list(sys.modules):
            if mod == "kuant.lifecycle" or mod.startswith("kuant.lifecycle."):
                del sys.modules[mod]
        with pytest.raises(KuantDeprecationWarning):
            importlib.import_module("kuant.lifecycle")
