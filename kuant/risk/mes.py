"""Marginal Expected Shortfall (MES).

MES measures the expected loss of an individual asset conditional on
the SYSTEM being in its worst days (below a system-wide tail quantile).
Introduced by Acharya-Pedersen-Philippon-Richardson 2017 as a
systemic-risk building block and closely related to CoVaR.

Formula:
    MES_i = E[ -r_i | r_system in worst tau tail ]

reported as a POSITIVE loss magnitude.

The empirical implementation identifies the worst `tau` fraction of
system days, then averages the individual return on those same days.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kuant._validation import require_1d, require_range
from kuant.errors import KuantValueError


@dataclass
class MesResult:
    mes: float
    system_var: float
    n_tail_days: int
    n: int
    tau: float

    def summary(self) -> str:
        return (
            "=== MesResult ===\n"
            f"MES (loss):       {self.mes:+.6f}\n"
            f"system VaR:       {self.system_var:+.6f}\n"
            f"tail days:        {self.n_tail_days} / {self.n}\n"
            f"tau:              {self.tau}"
        )


def mes(returns_asset, returns_system, *, tau: float = 0.05) -> MesResult:
    """Marginal Expected Shortfall.

    Parameters
    ----------
    returns_asset : 1D array
    returns_system : 1D array (equal length)
    tau : float, default 0.05
        Fraction of worst system days that define "tail." tau=0.05 uses
        the worst 5% of system days.

    Returns
    -------
    MesResult

    References
    ----------
    Acharya, Pedersen, Philippon & Richardson 2017, "Measuring Systemic
    Risk." Review of Financial Studies.
    """
    arr_a = np.asarray(returns_asset, dtype=np.float64)
    arr_s = np.asarray(returns_system, dtype=np.float64)
    require_1d(arr_a, "returns_asset", kernel="mes")
    require_1d(arr_s, "returns_system", kernel="mes")
    if arr_a.size != arr_s.size:
        raise KuantValueError(
            f"kuant.mes: 'returns_asset' and 'returns_system' must be "
            f"equal length; got {arr_a.size} and {arr_s.size}.  "
            f"[KE-SHAPE-EQUAL-LEN]"
        )
    mask = np.isfinite(arr_a) & np.isfinite(arr_s)
    a = arr_a[mask]
    s = arr_s[mask]
    if a.size < 100:
        raise KuantValueError(
            f"kuant.mes: only {a.size} paired finite values; need at "
            f"least 100.  [KE-VAL-MIN-CLEAN]"
        )
    require_range(tau, "tau", kernel="mes", lo=0.001, hi=0.5)

    threshold = float(np.quantile(s, tau))
    tail_mask = s <= threshold
    n_tail = int(tail_mask.sum())
    if n_tail == 0:
        raise KuantValueError(
            f"kuant.mes: no observations in tail (tau={tau}); increase "
            f"tau or provide more data.  [KE-VAL-MIN-CLEAN]"
        )
    mes_val = -float(a[tail_mask].mean())
    sys_var = -threshold
    return MesResult(
        mes=float(mes_val),
        system_var=float(sys_var),
        n_tail_days=int(n_tail),
        n=int(a.size),
        tau=float(tau),
    )


__all__ = ["MesResult", "mes"]
