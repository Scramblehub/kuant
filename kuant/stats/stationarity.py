"""Unit-root and stationarity tests.

Four standard tests bundled behind `statsmodels`, exposed with the
same result-dataclass and error-contract as the rest of kuant:

- `adftest`: Augmented Dickey-Fuller. Null = unit root (non-stationary).
- `kpsstest`: KPSS. Null = trend-stationary (opposite of ADF).
- `phillipsperrontest`: Phillips-Perron. Robust to serial correlation.
- `varianceratiotest`: Lo-MacKinlay variance ratio. Null = random walk.

All four take a 1D return or price series and report a test statistic,
a p-value, and a boolean `is_stationary` derived from a user-specified
significance level.

Design: docs/kernels/stats/stationarity.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kuant._validation import require_1d, require_dep, require_probability
from kuant.errors import KuantValueError


@dataclass
class StationarityResult:
    """Result common to the four stationarity tests.

    Attributes
    ----------
    statistic : float
        The test statistic.
    p_value : float
        Approximate p-value under the test's null hypothesis.
    is_stationary : bool
        Interpretation of the test at the requested `alpha`:
        `True` means the series is judged stationary under that test.
    test : str
        Which test produced this result.
    null_hypothesis : str
        Human-readable statement of the null.
    n : int
        Number of finite observations used.
    """

    statistic: float
    p_value: float
    is_stationary: bool
    test: str
    null_hypothesis: str
    n: int

    def summary(self) -> str:
        return (
            f"=== StationarityResult ({self.test}) ===\n"
            f"statistic:         {self.statistic:+.6f}\n"
            f"p-value:           {self.p_value:.4g}\n"
            f"is_stationary:     {self.is_stationary}\n"
            f"null:              {self.null_hypothesis}\n"
            f"n observations:    {self.n}"
        )


def _prep(series, kernel: str) -> np.ndarray:
    arr = np.asarray(series, dtype=np.float64)
    require_1d(arr, "series", kernel=kernel)
    finite = arr[np.isfinite(arr)]
    if finite.size < 20:
        raise KuantValueError(
            f"kuant.{kernel}: need at least 20 finite observations, got "
            f"{finite.size}.  [KE-VAL-RANGE]\n"
            f"  → Fix: provide more data or clean the input"
        )
    return finite


def adftest(series, alpha: float = 0.05, regression: str = "c") -> StationarityResult:
    """Augmented Dickey-Fuller test. Null = unit root (non-stationary).

    Small p means REJECT unit root → series IS stationary.

    Parameters
    ----------
    series : 1D array
    alpha : float, default 0.05
    regression : {'c', 'ct', 'ctt', 'n'}, default 'c'
        Deterministic terms in the ADF regression:
        `'c'` constant only, `'ct'` constant + trend, `'ctt'`
        constant + trend + quadratic trend, `'n'` no deterministic terms.
    """
    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError as e:
        require_dep("statsmodels", kernel="adftest", install="pip install statsmodels", cause=e)
    require_probability(alpha, "alpha", kernel="adftest")
    finite = _prep(series, "adftest")
    stat, pval, *_ = adfuller(finite, regression=regression, autolag="AIC")
    return StationarityResult(
        statistic=float(stat),
        p_value=float(pval),
        is_stationary=pval < alpha,
        test="adftest",
        null_hypothesis="series has a unit root (non-stationary)",
        n=int(finite.size),
    )


def kpsstest(series, alpha: float = 0.05, regression: str = "c") -> StationarityResult:
    """KPSS test. Null = trend-stationary.

    Small p means REJECT stationarity → series is NON-stationary.
    Opposite convention to ADF.

    Parameters
    ----------
    series : 1D array
    alpha : float, default 0.05
    regression : {'c', 'ct'}, default 'c'
    """
    try:
        from statsmodels.tsa.stattools import kpss
    except ImportError as e:
        require_dep("statsmodels", kernel="kpsstest", install="pip install statsmodels", cause=e)
    require_probability(alpha, "alpha", kernel="kpsstest")
    finite = _prep(series, "kpsstest")
    # kpss emits a InterpolationWarning that's expected; ignore.
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, pval, *_ = kpss(finite, regression=regression, nlags="auto")
    return StationarityResult(
        statistic=float(stat),
        p_value=float(pval),
        # Under KPSS the null is stationarity: a small p rejects it.
        is_stationary=pval >= alpha,
        test="kpsstest",
        null_hypothesis="series is trend-stationary",
        n=int(finite.size),
    )


def phillipsperrontest(series, alpha: float = 0.05, regression: str = "c") -> StationarityResult:
    """Phillips-Perron test. Null = unit root; robust to serial correlation.

    Uses `arch.unitroot.PhillipsPerron` if `arch` is installed, otherwise
    the `statsmodels` implementation.
    """
    try:
        from arch.unitroot import PhillipsPerron

        arch_available = True
    except ImportError:
        arch_available = False
    if not arch_available:
        # Fallback: statsmodels does not ship PP; raise an informative dep error.
        require_dep(
            "arch",
            kernel="phillipsperrontest",
            install="pip install arch",
            cause=ImportError("arch is not installed"),
        )
    require_probability(alpha, "alpha", kernel="phillipsperrontest")
    finite = _prep(series, "phillipsperrontest")
    trend_map = {"c": "c", "ct": "ct", "n": "n"}
    if regression not in trend_map:
        raise KuantValueError(
            f"kuant.phillipsperrontest: 'regression' must be one of "
            f"{tuple(trend_map)}, got {regression!r}.  [KE-VAL-RANGE]\n"
            f"  → Fix: pick one of {tuple(trend_map)}"
        )
    pp = PhillipsPerron(finite, trend=trend_map[regression])
    return StationarityResult(
        statistic=float(pp.stat),
        p_value=float(pp.pvalue),
        is_stationary=pp.pvalue < alpha,
        test="phillipsperrontest",
        null_hypothesis="series has a unit root",
        n=int(finite.size),
    )


def varianceratiotest(series, lags: int = 2, alpha: float = 0.05) -> StationarityResult:
    """Lo-MacKinlay variance-ratio test.

    Null: `series` is a random walk (returns are IID).
    Small p → returns are NOT IID → mean-reversion or momentum present.

    Parameters
    ----------
    series : 1D array
        Levels (prices or log-prices). Internal diff computes returns.
    lags : int, default 2
        Aggregation horizon. VR at lag k = Var(k-period returns) / (k * Var(1-period returns)).
    alpha : float
    """
    try:
        from arch.unitroot import VarianceRatio
    except ImportError as e:
        require_dep(
            "arch",
            kernel="varianceratiotest",
            install="pip install arch",
            cause=e,
        )
    require_probability(alpha, "alpha", kernel="varianceratiotest")
    finite = _prep(series, "varianceratiotest")
    vr = VarianceRatio(finite, lags=int(lags))
    return StationarityResult(
        statistic=float(vr.stat),
        p_value=float(vr.pvalue),
        # For VR the null is "random walk"; is_stationary=True means we
        # REJECT random walk, i.e. mean-reversion/momentum detected.
        is_stationary=vr.pvalue < alpha,
        test="varianceratiotest",
        null_hypothesis="returns are a random walk (IID)",
        n=int(finite.size),
    )


__all__ = [
    "StationarityResult",
    "adftest",
    "kpsstest",
    "phillipsperrontest",
    "varianceratiotest",
]
