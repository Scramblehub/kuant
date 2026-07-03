"""Entropy of an HMM state posterior as a per-bar confidence measure.

For a T×N state-posterior matrix γ where γ[t, i] = P(state = i | O_{1:t}),
Shannon entropy per time step is:

    H[t] = -Σ_i γ[t, i] · log(γ[t, i])

Range: [0, log(N)]. Low → posterior is confident (concentrated on one
state). High → posterior is diffuse (near-uniform over states).

Practical use: gate strategy actions only when the posterior is
confident. Entropy-weighted gating often outperforms threshold gating
when the HMM's posterior collapses cleanly in the target regime and
blurs in the counter-regime.

Optional: pass a categorical regime indicator to get conditional
entropy stats per regime. A collapse in one regime and a blur in
another is the signature this tool is designed to surface.

Design: docs/kernels/qm/posteriorentropy.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from kuant._validation import require_2d
from kuant.errors import KuantShapeError

cp: Any
try:
    import cupy as cp

    _CUPY_NDARRAY = cp.ndarray
except ImportError:
    cp = None
    _CUPY_NDARRAY = type(None)


@dataclass
class PosteriorEntropyResult:
    """Bar-by-bar entropy plus optional per-regime summary stats."""

    entropy: Any  # (T,) numpy or cupy array
    max_entropy: float  # log(N)
    per_regime: Optional[dict[str, dict[str, float]]] = None

    def summary(self) -> str:
        lines = [
            "=== Posterior entropy scan ===",
            f"Max entropy (log N):  {self.max_entropy:.4f}",
            f"Mean entropy:         {float(self.entropy.mean()):.4f}",
            f"Median entropy:       {float(np.median(np.asarray(self.entropy))):.4f}",
            f"Confident bars (<25% of max): {int((np.asarray(self.entropy) < 0.25 * self.max_entropy).sum())} / {len(self.entropy)}",
        ]
        if self.per_regime:
            lines.append("")
            lines.append(f'{"Regime":<20s} {"mean":>10s} {"std":>10s} {"n":>8s}')
            for regime, stats in self.per_regime.items():
                lines.append(
                    f'{regime:<20s} {stats["mean"]:>10.4f} {stats["std"]:>10.4f} {int(stats["n"]):>8d}'
                )
        return "\n".join(lines)


def posteriorentropy(gamma, regime=None):
    """Shannon entropy of an HMM posterior per time step.

    Parameters
    ----------
    gamma : (T, N) array (numpy or cupy)
        State posteriors. Rows should sum to 1 (kuant.qm.hmm.posterior
        guarantees this).
    regime : (T,) array of hashable labels, optional
        Categorical regime indicator (e.g. "high_vol", "low_vol"). If
        supplied, the result includes per-regime entropy statistics.

    Returns
    -------
    PosteriorEntropyResult
        With `entropy` (T,) array of per-bar entropy in nats
        (natural-log-based, range `[0, log(N)]`).

    Examples
    --------
    >>> import numpy as np
    >>> gamma = np.array([[0.99, 0.01], [0.5, 0.5], [0.9, 0.1]])
    >>> r = posteriorentropy(gamma)
    >>> r.entropy[0] < r.entropy[1]  # first bar confident, second bar diffuse
    True
    """
    if isinstance(gamma, _CUPY_NDARRAY):
        xp = cp
    else:
        xp = np
        gamma = np.asarray(gamma)

    require_2d(gamma, "gamma", kernel="posteriorentropy")

    N = gamma.shape[1]
    max_entropy = float(np.log(N))

    # -Σ γ log γ, using where to avoid 0 · log(0)
    log_gamma = xp.where(gamma > 0, xp.log(xp.where(gamma > 0, gamma, 1.0)), 0.0)
    entropy = -(gamma * log_gamma).sum(axis=1)

    per_regime: Optional[dict[str, dict[str, float]]] = None
    if regime is not None:
        entropy_np = entropy if xp is np else cp.asnumpy(entropy)
        regime_np = np.asarray(regime)
        if regime_np.size != entropy_np.size:
            raise KuantShapeError(
                f"kuant.posteriorentropy: 'regime' length {regime_np.size} "
                f"does not match posterior length T={entropy_np.size}.  "
                f"[KE-SHAPE-EQUAL-LEN]\n"
                f"  → Fix: 'regime' should be a categorical label per bar; "
                f"align its index with the posterior before calling"
            )
        per_regime = {}
        for label in np.unique(regime_np):
            mask = regime_np == label
            per_regime[str(label)] = {
                "mean": float(entropy_np[mask].mean()),
                "std": float(entropy_np[mask].std()),
                "n": int(mask.sum()),
            }

    return PosteriorEntropyResult(
        entropy=entropy,
        max_entropy=max_entropy,
        per_regime=per_regime,
    )
