"""Warmup: register indicators + gates upfront, materialize into a cache.

The vectorizable part of a backtest lives here. Indicators that don't
change per bar are computed once against the full price panel;
universe membership, lifecycle tradeable-masks, and liquidity-masks are
also precomputed. The engine loop then only handles the path-dependent
work (positions, cash, order tracking).

Three modes let the caller pick their tradeoff:

- `EAGER` (default): materialize everything upfront. Highest memory,
  fastest per-bar access. Right for offline backtests.
- `LAZY`: cache on first access. Right for exploratory notebooks where
  you don't need every indicator every run.
- `OFF`: never cache. Right for live-trading loops (fresh data each
  bar), small-window backtests (setup cost > savings), or iterative
  debugging.

Per-indicator `cache=True|False|None` override lets a caller mix
cached slow-moving indicators with live fast-moving ones.
"""

from __future__ import annotations

import time as _time
from typing import Callable

from kuant.backtest.lifecycle.security import (
    SecurityLifecycle,
    tradeable_mask,
)
from kuant.backtest.liquidity import LiquidityProfile, liquidity_mask
from kuant.backtest.warmup.cache import (
    WarmupCache,
    WarmupMode,
    _IndicatorRecord,
    _resolve_cache_flag,
)
from kuant.errors import KuantShapeError, KuantValueError


class Warmup:
    """Registration surface for indicators + gates before materializing.

    Parameters
    ----------
    prices : pandas.DataFrame
        Rows are dates, columns are symbols. Index must be date-like.
    mode : WarmupMode or str, default WarmupMode.EAGER
        Materialization strategy. Accepts either an enum value or a
        string (`"eager"`, `"lazy"`, `"off"`) for convenience.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> idx = pd.date_range('2020-01-01', periods=100, freq='D')
    >>> prices = pd.DataFrame(
    ...     {'XYZ': np.linspace(50, 100, 100)},
    ...     index=idx,
    ... )
    >>> from kuant.stats import rollmean
    >>> w = Warmup(prices, mode='eager')
    >>> w.add_indicator('sma20', rollmean, per_symbol=True, window=20)
    >>> cache = w.materialize()
    >>> cache.is_cached('sma20')
    True
    """

    def __init__(self, prices, mode=WarmupMode.EAGER) -> None:
        try:
            import pandas as pd
        except ImportError as e:  # pragma: no cover
            raise KuantValueError(
                "kuant.Warmup requires pandas.  [KE-DEP-MISSING]\n" "  → Fix: `pip install pandas`"
            ) from e
        if not isinstance(prices, pd.DataFrame):
            raise KuantShapeError(
                f"kuant.Warmup: 'prices' must be a pandas.DataFrame, "
                f"got {type(prices).__name__}.  [KE-SHAPE-EXPECTED]\n"
                f"  → Fix: wrap the price panel in "
                f"`pd.DataFrame(values, index=dates, columns=symbols)`"
            )
        if isinstance(mode, str):
            try:
                mode = WarmupMode(mode)
            except ValueError as e:
                raise KuantValueError(
                    f"kuant.Warmup: mode must be one of "
                    f"{[m.value for m in WarmupMode]}, got {mode!r}.  "
                    f"[KE-VAL-RANGE]\n"
                    f"  → Fix: pass 'eager', 'lazy', or 'off'"
                ) from e
        if not isinstance(mode, WarmupMode):
            raise KuantValueError(
                f"kuant.Warmup: mode must be WarmupMode or string, "
                f"got {type(mode).__name__}.  [KE-VAL-TYPE]"
            )
        self._prices = prices
        self._mode = mode
        self._indicators: dict[str, _IndicatorRecord] = {}
        self._membership = None
        self._lifecycles: dict[str, SecurityLifecycle] = {}
        self._liquidity_profiles: dict[str, LiquidityProfile] = {}

    # ---------- registration API -------------------------------------

    def add_indicator(
        self,
        name: str,
        kernel: Callable,
        *,
        per_symbol: bool = False,
        cache: bool | None = None,
        **kwargs,
    ) -> None:
        """Register an indicator.

        Parameters
        ----------
        name : str
            Handle used to query the value later via `cache.get(name, ...)`.
            Must be unique.
        kernel : callable
            The compute function. If `per_symbol=True`, called as
            `kernel(prices[col].to_numpy(), **kwargs)` per column and
            the results are stacked into a DataFrame. If False, called
            once as `kernel(prices, **kwargs)`; kernel is responsible
            for panel handling.
        per_symbol : bool, default False
        cache : bool or None, default None
            True forces caching regardless of mode; False forces live;
            None follows the Warmup's mode.
        **kwargs
            Passed straight to `kernel`.
        """
        if name in self._indicators:
            raise KuantValueError(
                f"kuant.Warmup.add_indicator: name {name!r} is already "
                f"registered.  [KE-VAL-DUPLICATE]\n"
                f"  → Fix: use a distinct name or remove the earlier "
                f"registration"
            )
        cache_flag = _resolve_cache_flag(self._mode, cache)
        self._indicators[name] = _IndicatorRecord(
            name=name,
            kernel=kernel,
            kwargs=dict(kwargs),
            per_symbol=per_symbol,
            cache_flag=cache_flag,
        )

    def add_universe_membership(self, membership) -> None:
        """Register a PIT membership panel.

        Parameters
        ----------
        membership : pandas.DataFrame
            Boolean (or 0/1) panel with the same date index as `prices`
            and columns matching a subset of symbols. Values are
            interpreted as `bool`; True = in-universe on that date.
        """
        try:
            import pandas as pd
        except ImportError as e:  # pragma: no cover
            raise KuantValueError("kuant.Warmup requires pandas.  [KE-DEP-MISSING]") from e
        if not isinstance(membership, pd.DataFrame):
            raise KuantShapeError(
                f"kuant.Warmup.add_universe_membership: 'membership' "
                f"must be a pandas.DataFrame, got "
                f"{type(membership).__name__}.  [KE-SHAPE-EXPECTED]"
            )
        if len(membership) != len(self._prices):
            raise KuantShapeError(
                f"kuant.Warmup.add_universe_membership: length "
                f"{len(membership)} does not match prices length "
                f"{len(self._prices)}.  [KE-SHAPE-EQUAL-LEN]\n"
                f"  → Fix: reindex membership onto the price panel's "
                f"date index before passing"
            )
        self._membership = membership.astype(bool)

    def add_lifecycles(self, lifecycles) -> None:
        """Register a symbol -> SecurityLifecycle mapping.

        Symbols not in the mapping are treated as always-tradeable.
        Symbols in the mapping but missing from `prices.columns` are
        silently retained (may be added later); tradeable_mask on such
        symbols has no effect on the panel.
        """
        for sym, lc in lifecycles.items():
            if not isinstance(lc, SecurityLifecycle):
                raise KuantValueError(
                    f"kuant.Warmup.add_lifecycles: value for {sym!r} "
                    f"must be a SecurityLifecycle, got "
                    f"{type(lc).__name__}.  [KE-VAL-TYPE]"
                )
        self._lifecycles.update(lifecycles)

    def add_liquidity_profiles(self, profiles) -> None:
        """Register a symbol -> LiquidityProfile mapping."""
        for sym, prof in profiles.items():
            if not isinstance(prof, LiquidityProfile):
                raise KuantValueError(
                    f"kuant.Warmup.add_liquidity_profiles: value for "
                    f"{sym!r} must be a LiquidityProfile, got "
                    f"{type(prof).__name__}.  [KE-VAL-TYPE]"
                )
        self._liquidity_profiles.update(profiles)

    # ---------- materialize ------------------------------------------

    def materialize(self) -> WarmupCache:
        """Compute cached items and return a queryable WarmupCache."""
        try:
            import pandas as pd
        except ImportError as e:  # pragma: no cover
            raise KuantValueError("kuant.Warmup requires pandas.  [KE-DEP-MISSING]") from e
        start = _time.perf_counter()

        cache = WarmupCache(
            mode=self._mode,
            prices=self._prices,
            indicators=self._indicators,
            membership=self._membership,
            lifecycles=dict(self._lifecycles),
            liquidity_profiles=dict(self._liquidity_profiles),
        )

        # Precompute tradeable_panel from lifecycles.
        if self._lifecycles:
            trade = {}
            for sym, lc in self._lifecycles.items():
                trade[sym] = tradeable_mask(self._prices.index, lc)
            cache.tradeable_panel = pd.DataFrame(trade, index=self._prices.index)

        # Precompute liquid_panel from liquidity profiles.
        if self._liquidity_profiles:
            liq = {}
            for sym, prof in self._liquidity_profiles.items():
                liq[sym] = liquidity_mask(self._prices.index, prof)
            cache.liquid_panel = pd.DataFrame(liq, index=self._prices.index)

        # Materialize EAGER-mode indicators (or those with cache=True override).
        if self._mode is WarmupMode.EAGER or any(
            rec.cache_flag for rec in self._indicators.values()
        ):
            for rec in self._indicators.values():
                if rec.cache_flag and self._mode is WarmupMode.EAGER:
                    rec.cached_result = cache._compute_indicator(rec)
                # LAZY + cache_flag=True is left to compute on first access.
                # OFF + cache_flag=True (explicit override) is materialized upfront.
                elif rec.cache_flag and self._mode is WarmupMode.OFF:
                    rec.cached_result = cache._compute_indicator(rec)

        cache.materialization_time_s = _time.perf_counter() - start
        return cache


__all__ = ["Warmup"]
