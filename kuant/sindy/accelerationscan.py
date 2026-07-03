'''Second-derivative (acceleration) predictive-power scan.

Motivation. Physics analog: knowing an object's position AND velocity
AND acceleration should predict its trajectory better than position
alone. Do financial time series show a similar pattern?

For a series `x` with target `y` (typically `y = x.shift(-h)`), compute
smoothed second derivatives at multiple bandwidths and report the
correlation of each acceleration variant with the target. If a
smoothing produces a meaningful correlation, acceleration is
predictive.

V8 SINDy #4 (acceleration null): tested three variants:

    accel_5_z  (5d MA of d²V8),   corr fwd 5d = -0.023,   fwd 21d = -0.004
    accel_21_z (21d MA of d²V8),  corr fwd 5d = +0.001,   fwd 21d = +0.020
    d2_z       (raw d²V8, 5d MA), corr fwd 5d = -0.013,   fwd 21d = -0.004

All |corr| < 0.025. Clean null: V8 returns are martingale-like at daily
frequency, no self-referential acceleration structure. This tool is
the automation of that test.

Design: docs/kernels/sindy/accelerationscan.md.
'''
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AccelerationScanResult:
    smoothings: list[int]
    correlations: dict[int, float]
    ns: dict[int, int]
    peak_smoothing: int
    peak_corr: float
    noise_floor: float

    def summary(self) -> str:
        lines = [
            '=== Acceleration predictive-power scan ===',
            f'Peak smoothing:           {self.peak_smoothing}',
            f'Peak |correlation|:       {self.peak_corr:+.4f}',
            f'Noise-floor threshold:    {self.noise_floor:.4f}',
            '',
            f'{"Smoothing":<12s} {"corr":>10s} {"n":>10s}',
        ]
        for w in self.smoothings:
            c = self.correlations[w]
            n = self.ns[w]
            marker = ' <- peak' if w == self.peak_smoothing else ''
            lines.append(f'{w:>10d}   {c:>+10.4f} {n:>10d}{marker}')

        if abs(self.peak_corr) < self.noise_floor:
            lines.append('')
            lines.append('DIAGNOSTIC: peak |correlation| below the noise floor. The series')
            lines.append('shows no exploitable acceleration structure. Returns are')
            lines.append('martingale-like at this frequency.')
        return '\n'.join(lines)


def _second_derivative(x: np.ndarray, smoothing: int) -> np.ndarray:
    '''Second derivative smoothed by an `smoothing`-length moving average.

    Approach: compute the discrete second difference `x[t] - 2x[t-1] +
    x[t-2]`, then apply a centered moving-average of length `smoothing`.
    NaN-filled at the boundaries where the smoothing window is not
    completely populated.
    '''
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    d2 = np.full(n, np.nan)
    d2[2:] = x[2:] - 2 * x[1:-1] + x[:-2]

    if smoothing <= 1:
        return d2

    out = np.full(n, np.nan)
    # Centered moving average
    half = smoothing // 2
    for t in range(half, n - half):
        window = d2[t - half:t + half + 1]
        finite = window[np.isfinite(window)]
        if finite.size == smoothing:
            out[t] = finite.mean()
    return out


def accelerationscan(
    x: np.ndarray,
    target: np.ndarray,
    smoothings: list[int] | None = None,
    noise_floor: float = 0.025,
) -> AccelerationScanResult:
    '''Test whether smoothed second derivatives of x predict target.

    Parameters
    ----------
    x : 1D np.ndarray
        Series whose acceleration will be measured (e.g. cumulative returns).
    target : 1D np.ndarray
        Same-length target (e.g. next-day return, next-week return).
    smoothings : list of int, optional
        MA lengths for the acceleration. Default `[1, 5, 21, 63]`.
    noise_floor : float, default 0.025
        Correlation magnitude below which the summary tags a null result.
        Rule of thumb: |corr| < 0.025 is indistinguishable from noise on
        typical daily-return time series.

    Returns
    -------
    AccelerationScanResult

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> # Series with a genuine acceleration signal: y_{t+1} = 0.5·(d²x)_t + noise
    >>> n = 500
    >>> x = np.cumsum(rng.normal(size=n))
    >>> d2 = np.zeros(n)
    >>> d2[2:] = x[2:] - 2 * x[1:-1] + x[:-2]
    >>> y = np.roll(0.5 * d2, -1) + rng.normal(scale=0.5, size=n)
    >>> result = accelerationscan(x, y, smoothings=[1, 5])
    >>> abs(result.correlations[1]) > result.noise_floor
    True
    '''
    if smoothings is None:
        smoothings = [1, 5, 21, 63]

    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(target, dtype=np.float64)

    if x_arr.size != y_arr.size:
        raise ValueError(f'x and target must have equal length, got {x_arr.size} vs {y_arr.size}')

    correlations: dict[int, float] = {}
    ns: dict[int, int] = {}

    for w in smoothings:
        if w < 1:
            raise ValueError(f'smoothing must be >= 1, got {w}')
        accel = _second_derivative(x_arr, w)
        mask = np.isfinite(accel) & np.isfinite(y_arr)
        n = int(mask.sum())
        if n < 30:
            correlations[w] = 0.0
            ns[w] = n
            continue
        c = np.corrcoef(accel[mask], y_arr[mask])[0, 1]
        correlations[w] = float(c) if np.isfinite(c) else 0.0
        ns[w] = n

    peak_w = max(smoothings, key=lambda w: abs(correlations[w]))
    peak_corr = correlations[peak_w]

    return AccelerationScanResult(
        smoothings=list(smoothings),
        correlations=correlations,
        ns=ns,
        peak_smoothing=peak_w,
        peak_corr=peak_corr,
        noise_floor=noise_floor,
    )
