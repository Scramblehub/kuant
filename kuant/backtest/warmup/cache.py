"""WarmupCache: uniform query surface over precomputed and live values.

The engine's inner loop looks the same regardless of whether the value
came from a precomputed cache, was computed on first access and then
cached, or is being recomputed every call:

    val = cache.get("rsi14", timestamp=t, symbol="XYZ")
    if cache.tradeable(t, "XYZ") and cache.liquid(t, "XYZ"):
        submit_order(...)

`Warmup` builds a cache; the cache handles the three modes internally.
Strategy code doesn't know or care which mode the operator picked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from kuant._validation import warn_kuant
from kuant.backtest.lifecycle.security import (
    _as_date,
)
from kuant.errors import KuantNumericWarning, KuantValueError


class WarmupMode(str, Enum):
    """How the cache handles indicator materialization.

    - `EAGER`: materialize all cached items once, upfront. Highest
      memory, fastest per-bar access. Default for offline backtests.
    - `LAZY`: cache-on-first-access. Any indicator not queried never
      gets computed. Useful for exploratory notebooks and partial
      backtests.
    - `OFF`: never cache. Every `.get()` calls the kernel. Right for
      live-trading loops where data is fresh each bar, small-window
      backtests where setup cost > savings, and iterative debugging.
    """

    EAGER = "eager"
    LAZY = "lazy"
    OFF = "off"


@dataclass
class _IndicatorRecord:
    """Internal: per-indicator state.

    `cached_result` is None until materialized (EAGER) or first
    accessed (LAZY). OFF mode never fills this.
    """

    name: str
    kernel: Callable
    kwargs: dict
    per_symbol: bool
    cache_flag: bool  # True = cached in this mode; False = always live
    cached_result: Any = None


@dataclass
class WarmupCache:
    """Query surface for indicators, universe, lifecycle, and liquidity.

    Users query via:
      - `get(name, timestamp, symbol=None)`: named indicator value
      - `tradeable(timestamp, symbol)`: lifecycle-window boolean
      - `liquid(timestamp, symbol)`: liquidity-mask boolean
      - `universe(timestamp, symbol)`: PIT-membership boolean

    Attributes
    ----------
    mode : WarmupMode
    prices : pandas.DataFrame
        Reference kept for LAZY / OFF modes (kernels are re-run against it).
    indicators : dict[str, _IndicatorRecord]
    membership : pandas.DataFrame or None
        PIT universe panel (date x symbol boolean). None = "no membership
        gating" (all symbols in `prices` are considered universe members).
    lifecycles : dict[str, SecurityLifecycle]
    liquidity_profiles : dict[str, LiquidityProfile]
    tradeable_panel : pandas.DataFrame or None
        Precomputed tradeable_mask per symbol; None if no lifecycles
        were registered.
    liquid_panel : pandas.DataFrame or None
        Precomputed liquidity_mask per symbol; None if no profiles
        were registered.
    materialization_time_s : float
        Wall-clock seconds spent in `Warmup.materialize()`.
    """

    mode: WarmupMode
    prices: object  # pandas.DataFrame
    indicators: dict = field(default_factory=dict)
    membership: object = None  # pandas.DataFrame or None
    lifecycles: dict = field(default_factory=dict)
    liquidity_profiles: dict = field(default_factory=dict)
    tradeable_panel: object = None  # pandas.DataFrame or None
    liquid_panel: object = None  # pandas.DataFrame or None
    materialization_time_s: float = 0.0

    # ---------- indicator get ------------------------------------------

    def get(self, name: str, timestamp, symbol: str | None = None):
        """Return the indicator value at `timestamp` (and optionally symbol).

        The return type depends on the underlying kernel:
        - Kernel that returns a Series (per-symbol call): scalar per
          timestamp. If `symbol` is None and the kernel produced a
          DataFrame, returns a Series across all symbols at that
          timestamp.
        - Kernel that returns a DataFrame (panel call): a Series across
          symbols at that timestamp, or a scalar if `symbol` is set.
        """
        if name not in self.indicators:
            raise KuantValueError(
                f"kuant.WarmupCache.get: no indicator named {name!r}. "
                f"Registered: {sorted(self.indicators.keys())}.  "
                f"[KE-VAL-MISSING]\n"
                f"  → Fix: call Warmup.add_indicator({name!r}, ...) "
                f"before materialize()"
            )
        rec = self.indicators[name]
        if rec.cache_flag and rec.cached_result is None and self.mode is WarmupMode.LAZY:
            rec.cached_result = self._compute_indicator(rec)
        if rec.cache_flag and rec.cached_result is not None:
            return self._slice(rec.cached_result, timestamp, symbol)
        # OFF mode or cache_flag == False: recompute every call.
        result = self._compute_indicator(rec)
        return self._slice(result, timestamp, symbol)

    def _compute_indicator(self, rec: _IndicatorRecord):
        """Run a single indicator kernel against the price panel."""
        try:
            import pandas as pd
        except ImportError as e:  # pragma: no cover
            raise KuantValueError("kuant.WarmupCache requires pandas.  [KE-DEP-MISSING]") from e
        prices = self.prices
        try:
            if rec.per_symbol:
                out = {}
                for col in prices.columns:
                    out[col] = rec.kernel(prices[col].to_numpy(), **rec.kwargs)
                return pd.DataFrame(out, index=prices.index)
            return rec.kernel(prices, **rec.kwargs)
        except Exception as exc:
            raise KuantValueError(
                f"kuant.Warmup: indicator {rec.name!r} failed during "
                f"materialization: {type(exc).__name__}: {exc}.  "
                f"[KE-WARMUP-INDICATOR-FAILED]\n"
                f"  → Fix: run the kernel standalone against the panel "
                f"to reproduce, or register {rec.name} with cache=False "
                f"to defer failure to first-access"
            ) from exc

    def _slice(self, result, timestamp, symbol):
        """Uniform slicing of the kernel's output."""
        try:
            import pandas as pd
        except ImportError as e:  # pragma: no cover
            raise KuantValueError("kuant.WarmupCache requires pandas.  [KE-DEP-MISSING]") from e
        ts = _as_date(timestamp)
        if isinstance(result, pd.DataFrame):
            # Locate the row.
            row_dates = [_as_date(x) for x in result.index]
            if ts not in row_dates:
                raise KuantValueError(
                    f"kuant.WarmupCache: timestamp {ts} not in "
                    f"indicator index.  [KE-VAL-MISSING]\n"
                    f"  → Fix: query a date within the price panel's index"
                )
            row = result.iloc[row_dates.index(ts)]
            if symbol is None:
                return row
            if symbol not in result.columns:
                raise KuantValueError(
                    f"kuant.WarmupCache: symbol {symbol!r} not in "
                    f"indicator columns.  [KE-VAL-MISSING]\n"
                    f"  → Fix: pass a symbol present in the price panel"
                )
            return row[symbol]
        if isinstance(result, pd.Series):
            row_dates = [_as_date(x) for x in result.index]
            if ts not in row_dates:
                raise KuantValueError(
                    f"kuant.WarmupCache: timestamp {ts} not in "
                    f"indicator index.  [KE-VAL-MISSING]"
                )
            return result.iloc[row_dates.index(ts)]
        # Scalar or array — return as-is; caller knows the shape they asked for.
        return result

    def is_cached(self, name: str) -> bool:
        """Whether an indicator's result is currently materialized."""
        if name not in self.indicators:
            raise KuantValueError(
                f"kuant.WarmupCache.is_cached: no indicator named " f"{name!r}.  [KE-VAL-MISSING]"
            )
        rec = self.indicators[name]
        return rec.cache_flag and rec.cached_result is not None

    # ---------- lifecycle / liquidity / universe gates ----------------

    def tradeable(self, timestamp, symbol: str) -> bool:
        """Was this symbol tradeable on this date?

        Returns True when no lifecycle was registered for the symbol
        (fall-through: assume tradeable if we know nothing).
        """
        if self.tradeable_panel is None or symbol not in self.tradeable_panel.columns:
            return True
        return bool(self._panel_get(self.tradeable_panel, timestamp, symbol))

    def liquid(self, timestamp, symbol: str) -> bool:
        """Did this symbol meet the liquidity gate on this date?

        Returns True when no liquidity profile was registered
        (fall-through: assume liquid if we know nothing).
        """
        if self.liquid_panel is None or symbol not in self.liquid_panel.columns:
            return True
        return bool(self._panel_get(self.liquid_panel, timestamp, symbol))

    def universe(self, timestamp, symbol: str) -> bool:
        """Was this symbol in the PIT universe on this date?

        Returns True when no membership panel was registered.
        """
        if self.membership is None:
            return True
        if symbol not in self.membership.columns:
            warn_kuant(
                kernel="WarmupCache.universe",
                code="KW-CACHE-UNIVERSE-UNKNOWN-SYMBOL",
                what=(
                    f"symbol {symbol!r} not in registered membership "
                    f"columns; treated as out-of-universe on every date"
                ),
                fix=(
                    "confirm the symbol was intended to be in the "
                    "universe; extending the membership panel is "
                    "preferable to silently gating out symbols the "
                    "strategy still trades"
                ),
                category=KuantNumericWarning,
            )
            return False
        return bool(self._panel_get(self.membership, timestamp, symbol))

    def _panel_get(self, panel, timestamp, symbol):
        """Row-slice a boolean panel by timestamp and symbol."""
        ts = _as_date(timestamp)
        row_dates = [_as_date(x) for x in panel.index]
        if ts not in row_dates:
            warn_kuant(
                kernel="WarmupCache._panel_get",
                code="KW-CACHE-TS-NOT-IN-PANEL",
                what=(
                    f"timestamp {ts} not in cache panel index; gate "
                    f"returned False by fallthrough, which will silently "
                    f"suppress any dependent orders"
                ),
                fix=(
                    "align the query timestamps to the panel's index "
                    "before iterating; the indicator-side .get() raises "
                    "on the same input"
                ),
                category=KuantNumericWarning,
            )
            return False
        return panel.iloc[row_dates.index(ts)][symbol]

    # ---------- reporting ---------------------------------------------

    def summary(self) -> str:
        n_ind = len(self.indicators)
        n_cached = sum(1 for r in self.indicators.values() if r.cached_result is not None)
        n_lc = len(self.lifecycles)
        n_liq = len(self.liquidity_profiles)
        return (
            "=== WarmupCache ===\n"
            f"mode:                    {self.mode.value}\n"
            f"panel shape:             {self.prices.shape}\n"
            f"indicators registered:   {n_ind}\n"
            f"indicators materialized: {n_cached}\n"
            f"lifecycles:              {n_lc}\n"
            f"liquidity profiles:      {n_liq}\n"
            f"membership panel:        "
            f"{'yes' if self.membership is not None else 'no'}\n"
            f"materialization time:    {self.materialization_time_s:.4f} s"
        )


def _resolve_cache_flag(mode: WarmupMode, override) -> bool:
    """Whether an indicator should be cached under this mode + override."""
    if override is True:
        return True
    if override is False:
        return False
    return mode is not WarmupMode.OFF


__all__ = ["WarmupCache", "WarmupMode", "_IndicatorRecord", "_resolve_cache_flag"]
