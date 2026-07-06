"""Composer: run the full chaos battery and classify the regime.

Given a 1D series, this kernel:

1. Picks `tau` via first minimum of auto-mutual-information.
2. Picks `m` via false-nearest-neighbors threshold crossing.
3. Runs `lyapunov` for the divergence rate.
4. Runs `corrdim` for the correlation dimension.
5. Runs `rqa` for recurrence structure.

It then classifies the regime into one of:

- **chaotic**: positive Lyapunov, finite low correlation dim,
  high determinism.
- **periodic**: near-zero Lyapunov, D_2 close to 1, very high
  determinism.
- **stochastic**: near-zero or negative Lyapunov, D_2 doesn't saturate
  (approximated by D_2 >= embedding dim), low determinism.
- **unknown**: signals are mutually inconsistent.

Design: docs/kernels/sindy/chaos/chaosscan.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kuant._validation import require_1d
from kuant.errors import KuantValueError
from kuant.sindy.chaos.ccm import CCMResult
from kuant.sindy.chaos.corrdim import CorrDimResult, corrdim
from kuant.sindy.chaos.embedding import (
    FalseNearestResult,
    MutualInfoResult,
    falsenearest,
    mutualinfo,
)
from kuant.sindy.chaos.lyapunov import LyapunovResult, lyapunov
from kuant.sindy.chaos.rqa import RQAResult, rqa


@dataclass
class ChaosScanResult:
    """Composite chaos-battery result plus a regime label.

    Attributes
    ----------
    regime : str
        One of {"chaotic", "periodic", "stochastic", "unknown"}.
    mutualinfo : MutualInfoResult
    falsenearest : FalseNearestResult
    lyapunov : LyapunovResult
    corrdim : CorrDimResult
    rqa : RQAResult
    embed_tau : int
    embed_dim : int
    """

    regime: str
    mutualinfo: MutualInfoResult
    falsenearest: FalseNearestResult
    lyapunov: LyapunovResult
    corrdim: CorrDimResult
    rqa: RQAResult
    embed_tau: int
    embed_dim: int

    def summary(self) -> str:
        return (
            "=== ChaosScanResult ===\n"
            f"regime:              {self.regime}\n"
            f"tau (auto-MI):       {self.embed_tau}\n"
            f"m (FNN):             {self.embed_dim}\n"
            f"lambda:              {self.lyapunov.lyapunov:+.5f} nats/sample\n"
            f"D_2:                 {self.corrdim.correlation_dim:.4f}\n"
            f"RR / DET / LAM:      {self.rqa.recurrence_rate:.3f} / "
            f"{self.rqa.determinism:.3f} / {self.rqa.laminarity:.3f}\n"
            f"longest diag:        {self.rqa.longest_diagonal}"
        )


def _classify(
    lyap: float,
    d2: float,
    det: float,
    embed_dim: int,
) -> str:
    """Rule-based regime classifier.

    Thresholds are the "default" ones the literature settles on; users
    can pull the raw kernel results and apply their own thresholds if
    they disagree.
    """
    # Chaotic: positive Lyapunov, finite D_2 well below embed_dim,
    # high determinism.
    if lyap > 0.001 and d2 < embed_dim - 0.5 and det > 0.5:
        return "chaotic"
    # Periodic: Lyapunov ~ 0, D_2 close to 1, very high determinism.
    if abs(lyap) < 0.005 and d2 < 1.5 and det > 0.9:
        return "periodic"
    # Stochastic: D_2 saturates near embed_dim (no attractor), low det.
    if d2 >= embed_dim - 0.5 and det < 0.5:
        return "stochastic"
    return "unknown"


def chaosscan(
    x,
    *,
    tau: int | None = None,
    m: int | None = None,
    max_lag: int = 32,
    max_dim: int = 10,
    n_r: int = 20,
) -> ChaosScanResult:
    """Full chaos-battery scan with regime classification.

    Parameters
    ----------
    x : 1D array
    tau : int, optional
        If None, auto-pick via `mutualinfo`.
    m : int, optional
        If None, auto-pick via `falsenearest`.
    max_lag : int, default 32
    max_dim : int, default 10
    n_r : int, default 20

    Returns
    -------
    ChaosScanResult

    Notes
    -----
    - For CCM causality between two series, call `ccm` directly.
    - Requires at least 300 finite values (bounded by the `corrdim`
      minimum).
    """
    arr = np.asarray(x, dtype=np.float64)
    require_1d(arr, "x", kernel="chaosscan")
    finite = np.isfinite(arr)
    arr = arr[finite]
    if arr.size < 300:
        raise KuantValueError(
            f"kuant.chaosscan: only {arr.size} finite values; need at "
            f"least 300 for the composite battery.  "
            f"[KE-VAL-MIN-CLEAN]"
        )

    mi_res = mutualinfo(arr, max_lag=int(max_lag))
    tau_pick = int(tau) if tau is not None else int(mi_res.suggested_tau)
    fnn_res = falsenearest(arr, tau=tau_pick, max_dim=int(max_dim))
    m_pick = int(m) if m is not None else int(fnn_res.suggested_m)

    lyap_res = lyapunov(arr, tau=tau_pick, m=m_pick)
    corr_res = corrdim(arr, tau=tau_pick, m=m_pick, n_r=int(n_r))
    rqa_res = rqa(arr, tau=tau_pick, m=m_pick)

    regime = _classify(
        lyap=lyap_res.lyapunov,
        d2=corr_res.correlation_dim,
        det=rqa_res.determinism,
        embed_dim=m_pick,
    )
    return ChaosScanResult(
        regime=regime,
        mutualinfo=mi_res,
        falsenearest=fnn_res,
        lyapunov=lyap_res,
        corrdim=corr_res,
        rqa=rqa_res,
        embed_tau=tau_pick,
        embed_dim=m_pick,
    )


# Also re-export CCMResult so `from kuant.sindy.chaos.chaosscan import *`
# doesn't feel incomplete.
__all__ = ["ChaosScanResult", "CCMResult", "chaosscan"]
