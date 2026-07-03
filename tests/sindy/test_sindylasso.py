'''Test suite for kuant.sindy.sindylasso.'''
from __future__ import annotations

import numpy as np
import pytest

from kuant.sindy import sindylasso


def test_selects_true_signal():
    '''y = 0.5·x1 + noise, x2 unrelated → x1 selected, x2 not.'''
    rng = np.random.default_rng(0)
    n = 500
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    noise = rng.normal(scale=0.3, size=n)
    y = 0.5 * x1 + noise
    result = sindylasso(y, {'x1': x1, 'x2': x2, 'noise_col': rng.normal(size=n)})
    assert 'x1' in result.selected_features
    # x2 might or might not be selected depending on CV; but if it is, coef should be small
    if 'x2' in result.selected_features:
        assert abs(result.selected_features['x2']) < abs(result.selected_features['x1'])


def test_null_signal_selects_no_features():
    '''y independent of all features → LASSO should select nothing.'''
    rng = np.random.default_rng(0)
    n = 500
    y = rng.normal(size=n)
    library = {f'noise_{i}': rng.normal(size=n) for i in range(10)}
    result = sindylasso(y, library, alpha_grid=np.logspace(-4, -1, 20))
    # A well-behaved null test picks alpha at the top of the range and selects 0 features
    assert len(result.selected_features) == 0 or all(
        abs(c) < 0.05 for c in result.selected_features.values()
    )


def test_nan_rows_dropped():
    '''NaN rows should be dropped, not crash.'''
    rng = np.random.default_rng(0)
    n = 300
    x1 = rng.normal(size=n)
    x1[10] = np.nan  # NaN in x1
    x2 = rng.normal(size=n)
    y = 0.5 * x1 + rng.normal(scale=0.3, size=n)
    y[20] = np.nan  # NaN in y
    result = sindylasso(y, {'x1': x1, 'x2': x2})
    assert isinstance(result.selected_features, dict)


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match='has length'):
        sindylasso(np.arange(100.0), {'x': np.arange(50.0)})


def test_too_few_clean_rows_raises():
    rng = np.random.default_rng(0)
    y = np.full(100, np.nan)  # all NaN
    y[0] = 1.0
    y[1] = 2.0
    x = rng.normal(size=100)
    with pytest.raises(ValueError, match='too few clean rows'):
        sindylasso(y, {'x': x})


def test_summary_readable():
    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(size=n)
    y = 0.5 * x + rng.normal(scale=0.3, size=n)
    text = sindylasso(y, {'x': x}).summary()
    assert 'LASSO' in text
    assert 'Alpha' in text
