"""Entropy-based complexity measures for time series.

Five kernels that quantify signal complexity from different angles:

- `permutationentropy` (Bandt-Pompe 2002): ordinal-pattern entropy.
  Fast, robust to monotonic transforms and to noise. Best default for
  a first-pass complexity screen.
- `sampleentropy` (Richman-Moorman 2000): improvement over ApEn.
  Removes the self-match bias. Standard for physiologic and financial
  time series.
- `approximateentropy` (Pincus 1991): the original nonlinear
  complexity measure. Includes self-match (weakness vs SampEn) but
  still cited widely for legacy comparison.
- `dispersionentropy` (Rostaghi-Azami 2016): class-based ordinal
  measure. More stable than SampEn for short series (~ 200 points).
- `transferentropy` (Schreiber 2000): directed information flow from
  X to Y. Model-free measure of asymmetric coupling. Naive
  histogram-based estimator in this MVP.

All returns are in nats. Where a small-sample bias is known, the
docstring flags it and points to the correction.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log

import numpy as np

from kuant._validation import require_1d, require_positive, require_range
import warnings

from kuant.errors import KuantNumericWarning, KuantValueError


# ---------- shared helpers --------------------------------------------


def _clean_finite_1d(x, kernel: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    require_1d(arr, "x", kernel=kernel)
    return arr[np.isfinite(arr)]


def _embed(x: np.ndarray, m: int, tau: int) -> np.ndarray:
    """Time-delay embedding: shape (N - (m-1)*tau, m). Shared with the
    embedding module but re-implemented here so this file is standalone."""
    n = x.size - (m - 1) * tau
    out = np.empty((n, m), dtype=np.float64)
    for j in range(m):
        out[:, j] = x[j * tau : j * tau + n]
    return out


# ---------- permutation entropy ---------------------------------------


@dataclass
class PermutationEntropyResult:
    """Bandt-Pompe permutation entropy.

    Attributes
    ----------
    entropy : float
        Shannon entropy (nats) of the ordinal-pattern distribution.
    normalized : float
        `entropy / log(m!)`, in [0, 1]. 1 = fully random ordering, 0 =
        perfectly monotone.
    n_patterns_seen : int
        Distinct ordinal patterns actually observed. Max is `m!`.
    embed_dim : int
    embed_tau : int
    """

    entropy: float
    normalized: float
    n_patterns_seen: int
    embed_dim: int
    embed_tau: int

    def summary(self) -> str:
        return (
            "=== PermutationEntropyResult ===\n"
            f"entropy (nats):   {self.entropy:.4f}\n"
            f"normalized:       {self.normalized:.4f}\n"
            f"patterns seen:    {self.n_patterns_seen}\n"
            f"embed dim / tau:  {self.embed_dim} / {self.embed_tau}"
        )


def permutationentropy(x, *, m: int = 3, tau: int = 1) -> PermutationEntropyResult:
    """Bandt-Pompe permutation entropy.

    Parameters
    ----------
    x : 1D array
    m : int, default 3
        Ordinal-pattern length. Practical range 3-7. Higher m needs
        more data.
    tau : int, default 1
        Embedding delay.

    Returns
    -------
    PermutationEntropyResult

    References
    ----------
    Bandt & Pompe 2002, "Permutation entropy: a natural complexity
    measure for time series."
    """
    arr = _clean_finite_1d(x, kernel="permutationentropy")
    require_range(m, "m", kernel="permutationentropy", lo=2, hi=10)
    require_positive(tau, "tau", kernel="permutationentropy", kind="int")
    if arr.size < int(m) * int(tau) + 1:
        raise KuantValueError(
            f"kuant.permutationentropy: only {arr.size} finite values; "
            f"need at least {int(m) * int(tau) + 1} for m={m}, "
            f"tau={tau}.  [KE-VAL-MIN-CLEAN]"
        )
    emb = _embed(arr, int(m), int(tau))
    # Rank pattern per row.
    order = np.argsort(emb, axis=1, kind="stable")
    # Encode each row's permutation as a tuple (hashable).
    counts: dict = {}
    for row in order:
        key = tuple(int(v) for v in row)
        counts[key] = counts.get(key, 0) + 1
    total = sum(counts.values())
    probs = np.array([c / total for c in counts.values()], dtype=np.float64)
    ent = float(-np.sum(probs * np.log(probs)))
    max_ent = log(_factorial(int(m)))
    normalized = ent / max_ent if max_ent > 0 else 0.0
    return PermutationEntropyResult(
        entropy=ent,
        normalized=normalized,
        n_patterns_seen=len(counts),
        embed_dim=int(m),
        embed_tau=int(tau),
    )


def _factorial(n: int) -> int:
    out = 1
    for i in range(2, n + 1):
        out *= i
    return out


# ---------- sample entropy --------------------------------------------


@dataclass
class SampleEntropyResult:
    """Richman-Moorman sample entropy.

    Attributes
    ----------
    entropy : float
        `-log(A / B)` in nats. Higher = more complex.
    A : int
        Count of (m+1)-length pattern matches under tolerance r.
    B : int
        Count of m-length pattern matches under tolerance r.
    m : int
    r : float
    """

    entropy: float
    A: int
    B: int
    m: int
    r: float

    def summary(self) -> str:
        return (
            "=== SampleEntropyResult ===\n"
            f"SampEn:          {self.entropy:.4f}\n"
            f"A (m+1 match):   {self.A}\n"
            f"B (m match):     {self.B}\n"
            f"m / r:           {self.m} / {self.r:.4g}"
        )


def sampleentropy(x, *, m: int = 2, r: float | None = None) -> SampleEntropyResult:
    """Richman-Moorman sample entropy.

    Parameters
    ----------
    x : 1D array
    m : int, default 2
        Template length. m=2 is the physiologic-signal standard.
    r : float, optional
        Match tolerance. If None, defaults to 0.2 * std(x). Standard
        heuristic from the source paper.

    Returns
    -------
    SampleEntropyResult

    References
    ----------
    Richman & Moorman 2000, "Physiological time-series analysis using
    approximate entropy and sample entropy."
    """
    arr = _clean_finite_1d(x, kernel="sampleentropy")
    require_range(m, "m", kernel="sampleentropy", lo=1, hi=10)
    if arr.size < 50:
        raise KuantValueError(
            f"kuant.sampleentropy: only {arr.size} finite values; need "
            f"at least 50 for a stable estimate.  [KE-VAL-MIN-CLEAN]"
        )
    if r is None:
        r = 0.2 * float(np.std(arr, ddof=1))
    require_positive(r, "r", kernel="sampleentropy")

    A = _count_matches(arr, int(m) + 1, float(r))
    B = _count_matches(arr, int(m), float(r))
    if B == 0 or A == 0:
        # Infinite / undefined SampEn. Return NaN with a soft signal.
        return SampleEntropyResult(entropy=float("nan"), A=int(A), B=int(B), m=int(m), r=float(r))
    ent = -log(A / B)
    return SampleEntropyResult(entropy=float(ent), A=int(A), B=int(B), m=int(m), r=float(r))


def _count_matches(x: np.ndarray, m: int, r: float) -> int:
    """Count template-vector pairs within tolerance r using Chebyshev distance."""
    n = x.size - m + 1
    if n <= 1:
        return 0
    templates = _embed(x, m, 1)
    count = 0
    for i in range(n):
        # Chebyshev distance to all j > i, excluding self.
        diff = np.max(np.abs(templates[i + 1 :] - templates[i]), axis=1)
        count += int(np.sum(diff <= r))
    return count


# ---------- approximate entropy ---------------------------------------


@dataclass
class ApproximateEntropyResult:
    """Pincus approximate entropy.

    Attributes
    ----------
    entropy : float
        ApEn in nats.
    m : int
    r : float

    Notes
    -----
    ApEn includes self-matches, so it is biased toward regularity on
    short series. Use `sampleentropy` for a bias-corrected variant.
    """

    entropy: float
    m: int
    r: float

    def summary(self) -> str:
        return (
            "=== ApproximateEntropyResult ===\n"
            f"ApEn:           {self.entropy:.4f}\n"
            f"m / r:          {self.m} / {self.r:.4g}"
        )


def approximateentropy(x, *, m: int = 2, r: float | None = None) -> ApproximateEntropyResult:
    """Pincus approximate entropy.

    Parameters
    ----------
    x : 1D array
    m : int, default 2
    r : float, optional
        Default 0.2 * std(x).

    Returns
    -------
    ApproximateEntropyResult

    References
    ----------
    Pincus 1991, "Approximate entropy as a measure of system complexity."
    """
    arr = _clean_finite_1d(x, kernel="approximateentropy")
    require_range(m, "m", kernel="approximateentropy", lo=1, hi=10)
    if arr.size < 50:
        raise KuantValueError(
            f"kuant.approximateentropy: only {arr.size} finite values; "
            f"need at least 50 for a stable estimate.  [KE-VAL-MIN-CLEAN]"
        )
    if r is None:
        r = 0.2 * float(np.std(arr, ddof=1))
    require_positive(r, "r", kernel="approximateentropy")

    def phi(m_: int) -> float:
        n = arr.size - m_ + 1
        templates = _embed(arr, m_, 1)
        # Include self-matches (ApEn's defining feature).
        counts = np.empty(n, dtype=np.float64)
        for i in range(n):
            diff = np.max(np.abs(templates - templates[i]), axis=1)
            counts[i] = np.sum(diff <= r) / n
        # log then average, skipping zero counts (shouldn't happen with self-match).
        return float(np.mean(np.log(counts[counts > 0])))

    apen = phi(int(m)) - phi(int(m) + 1)
    return ApproximateEntropyResult(entropy=float(apen), m=int(m), r=float(r))


# ---------- dispersion entropy ----------------------------------------


@dataclass
class DispersionEntropyResult:
    """Rostaghi-Azami dispersion entropy.

    Attributes
    ----------
    entropy : float
        Shannon entropy (nats) of the dispersion-pattern distribution.
    normalized : float
        `entropy / log(c ** m)`, in [0, 1].
    embed_dim : int
    embed_tau : int
    n_classes : int
    """

    entropy: float
    normalized: float
    embed_dim: int
    embed_tau: int
    n_classes: int

    def summary(self) -> str:
        return (
            "=== DispersionEntropyResult ===\n"
            f"entropy (nats):  {self.entropy:.4f}\n"
            f"normalized:      {self.normalized:.4f}\n"
            f"embed dim/tau:   {self.embed_dim} / {self.embed_tau}\n"
            f"n classes:       {self.n_classes}"
        )


def dispersionentropy(x, *, m: int = 3, tau: int = 1, c: int = 6) -> DispersionEntropyResult:
    """Rostaghi-Azami dispersion entropy.

    Parameters
    ----------
    x : 1D array
    m : int, default 3
    tau : int, default 1
    c : int, default 6
        Number of classes for the normal-CDF class assignment.
        Practical range 4-8.

    Returns
    -------
    DispersionEntropyResult

    References
    ----------
    Rostaghi & Azami 2016, "Dispersion entropy: a measure for time-
    series analysis."
    """
    arr = _clean_finite_1d(x, kernel="dispersionentropy")
    require_range(m, "m", kernel="dispersionentropy", lo=2, hi=10)
    require_positive(tau, "tau", kernel="dispersionentropy", kind="int")
    require_range(c, "c", kernel="dispersionentropy", lo=2, hi=32)
    if arr.size < int(m) * int(tau) + 1:
        raise KuantValueError(
            f"kuant.dispersionentropy: only {arr.size} finite values; "
            f"need at least {int(m) * int(tau) + 1} for m={m}, "
            f"tau={tau}.  [KE-VAL-MIN-CLEAN]"
        )
    # Map to classes via normal CDF (z-score then bin).
    std_arr = float(arr.std(ddof=1))
    if std_arr < 1e-15:
        warnings.warn(
            "kuant.dispersionentropy: input is (near-)constant; "
            "dispersion entropy is 0 by construction and provides no "
            "information about complexity.  [KW-DE-CONSTANT-INPUT]",
            KuantNumericWarning,
            stacklevel=2,
        )
        return DispersionEntropyResult(
            entropy=0.0,
            normalized=0.0,
            embed_dim=int(m),
            embed_tau=int(tau),
            n_classes=int(c),
        )
    z = (arr - arr.mean()) / (std_arr + 1e-12)
    # Normal CDF.
    from math import erf, sqrt

    cdf = 0.5 * (1.0 + np.vectorize(lambda v: erf(v / sqrt(2)))(z))
    classes = np.clip((cdf * c).astype(int) + 1, 1, int(c))
    # Embed the class sequence.
    emb = _embed(classes.astype(np.float64), int(m), int(tau)).astype(int)
    # Pattern -> string key for counting.
    counts: dict = {}
    for row in emb:
        key = tuple(int(v) for v in row)
        counts[key] = counts.get(key, 0) + 1
    total = sum(counts.values())
    probs = np.array([v / total for v in counts.values()], dtype=np.float64)
    ent = float(-np.sum(probs * np.log(probs)))
    max_ent = log(int(c) ** int(m))
    normalized = ent / max_ent if max_ent > 0 else 0.0
    return DispersionEntropyResult(
        entropy=ent,
        normalized=normalized,
        embed_dim=int(m),
        embed_tau=int(tau),
        n_classes=int(c),
    )


# ---------- transfer entropy ------------------------------------------


@dataclass
class TransferEntropyResult:
    """Schreiber transfer entropy from X to Y.

    Attributes
    ----------
    te : float
        Transfer entropy X -> Y in nats.
    lag : int
    bins : int
    n_pairs : int
        Number of aligned finite triples used in the estimate.
    """

    te: float
    lag: int
    bins: int
    n_pairs: int

    def summary(self) -> str:
        return (
            "=== TransferEntropyResult ===\n"
            f"TE (X -> Y):     {self.te:.5f} nats\n"
            f"lag:             {self.lag}\n"
            f"bins:            {self.bins}\n"
            f"n pairs:         {self.n_pairs}"
        )


def transferentropy(x, y, *, lag: int = 1, bins: int = 6) -> TransferEntropyResult:
    """Schreiber transfer entropy from X to Y via histogram estimator.

    Estimates `TE(X -> Y) = H(Y_{t+lag} | Y_t) - H(Y_{t+lag} | Y_t, X_t)`
    by binning each variable into `bins` classes and computing empirical
    conditional entropies from the joint histogram.

    Parameters
    ----------
    x, y : 1D arrays of equal length
    lag : int, default 1
        Prediction horizon on Y.
    bins : int, default 6
        Bins per variable. Total joint state = bins**3.

    Returns
    -------
    TransferEntropyResult

    References
    ----------
    Schreiber 2000, "Measuring information transfer."

    Notes
    -----
    Histogram estimators have a positive small-sample bias. For
    inference at short samples, subtract a shuffled-Y baseline as a
    permutation null.
    """
    arr_x = np.asarray(x, dtype=np.float64)
    arr_y = np.asarray(y, dtype=np.float64)
    require_1d(arr_x, "x", kernel="transferentropy")
    require_1d(arr_y, "y", kernel="transferentropy")
    if arr_x.size != arr_y.size:
        raise KuantValueError(
            f"kuant.transferentropy: 'x' and 'y' must be equal length; "
            f"got {arr_x.size} and {arr_y.size}.  [KE-SHAPE-EQUAL-LEN]"
        )
    require_positive(lag, "lag", kernel="transferentropy", kind="int")
    require_range(bins, "bins", kernel="transferentropy", lo=2, hi=64)

    mask = np.isfinite(arr_x) & np.isfinite(arr_y)
    xf = arr_x[mask]
    yf = arr_y[mask]
    if xf.size < 100 + int(lag):
        raise KuantValueError(
            f"kuant.transferentropy: only {xf.size} paired finite values; "
            f"need at least {100 + int(lag)}.  [KE-VAL-MIN-CLEAN]"
        )

    # Aligned triples: (Y_t, X_t, Y_{t+lag}).
    t = xf.size - int(lag)
    y_t = yf[:t]
    x_t = xf[:t]
    y_next = yf[int(lag) :]

    # Bin each into [0, bins).
    def _bin(v: np.ndarray) -> np.ndarray:
        edges = np.quantile(v, np.linspace(0, 1, int(bins) + 1))
        edges[0] = -np.inf
        edges[-1] = np.inf
        return np.clip(np.searchsorted(edges, v, side="right") - 1, 0, int(bins) - 1)

    by = _bin(y_t)
    bx = _bin(x_t)
    bn = _bin(y_next)

    # Joint counts.
    joint = np.zeros((int(bins), int(bins), int(bins)), dtype=np.float64)
    for i in range(t):
        joint[by[i], bx[i], bn[i]] += 1.0

    total = joint.sum()
    if total == 0:
        return TransferEntropyResult(te=float("nan"), lag=int(lag), bins=int(bins), n_pairs=0)

    p_yxn = joint / total
    p_yx = p_yxn.sum(axis=2, keepdims=True)  # marginal over Y_next
    p_yn = p_yxn.sum(axis=1, keepdims=True)  # marginal over X_t
    p_y = p_yn.sum(axis=2, keepdims=True)  # marginal over Y_next AND X_t

    # TE = sum p(yn, xt, yt) log [ p(yn|yt, xt) / p(yn|yt) ]
    #    = sum p(yn, xt, yt) log [ p(yn,xt,yt) * p(yt) / (p(yt,xt) * p(yt,yn)) ]
    with np.errstate(divide="ignore", invalid="ignore"):
        num = p_yxn * p_y
        den = p_yx * p_yn
        ratio = np.where((num > 0) & (den > 0), num / den, 1.0)
        contribs = np.where(p_yxn > 0, p_yxn * np.log(ratio), 0.0)
    te = float(np.sum(contribs))
    return TransferEntropyResult(te=te, lag=int(lag), bins=int(bins), n_pairs=int(t))


__all__ = [
    "PermutationEntropyResult",
    "SampleEntropyResult",
    "ApproximateEntropyResult",
    "DispersionEntropyResult",
    "TransferEntropyResult",
    "permutationentropy",
    "sampleentropy",
    "approximateentropy",
    "dispersionentropy",
    "transferentropy",
]
