"""kuant.nulltest — adversarial 'is this signal noise?' testing.

Consolidates the null-hypothesis tools scattered around kuant.sindy and
kuant.portfolio under one namespace. Every test returns a `NullReport`
dataclass with the same shape so users can compose across tests:

    from kuant.nulltest import bootstrap_ic, spa_test, mht_correction

    b = bootstrap_ic(signal, forward_returns, n_boot=1000)
    s = spa_test(observed=b.point_estimate, alternatives=alt_sharpes)
    corrected = mht_correction(b.p_value, method='bh')

Kernels shipped:

- `bootstrap_ic`: block-bootstrap CI for the IC of a signal.
- `stationary_bootstrap`: Politis-Romano stationary-block resampling.
- `spa_test`: Hansen's Superior Predictive Ability test.
- `mcs_test`: Model Confidence Set (Hansen-Lunde-Nason).
- `mht_correction`: Bonferroni / Holm / Benjamini-Hochberg multiple-hypothesis
  p-value adjustment.
- `deflated_sharpe`: re-exported from kuant.portfolio for convenience.
"""

from kuant.nulltest.bootstrap import (
    BootstrapICResult,
    bootstrap_ic,
    stationary_bootstrap,
)
from kuant.nulltest.mht_correction import mht_correction
from kuant.nulltest.spa_test import SPAResult, mcs_test, spa_test
from kuant.portfolio.riskmetrics import deflated_sharpe

__all__ = [
    "BootstrapICResult",
    "SPAResult",
    "bootstrap_ic",
    "deflated_sharpe",
    "mcs_test",
    "mht_correction",
    "spa_test",
    "stationary_bootstrap",
]
