# kuant.backtest.liquidity

Volume-aware fill semantics for point-in-time equity panels.

## The companion to lifecycle

`kuant.backtest.lifecycle` answers "does this security exist right
now?" `kuant.backtest.liquidity` answers the next question in the
chain: "can this order fill at the stated price, and how much
slippage?" A tradeable name at a live price still fails to fill
cleanly if the requested size dwarfs the day's volume, undershoots the
broker's minimum lot, or lands on a zero-volume bar. This subpackage
encodes those three frictions as a typed primitive so a simulator
cannot silently drift into fills that a real venue would refuse.

Typical backtest engines model slippage as a flat basis-point haircut
and ignore volume entirely. That approximation is fine for small
orders in liquid names and wildly wrong at scale; a strategy that
looks profitable in-sample can evaporate the moment the notional it
prints in the report would have moved the market it traded against.
`kuant.backtest.liquidity` treats participation, minimum size, and
size-dependent impact as first-class inputs.

## Files

- [`profile.md`](profile.md): `LiquidityProfile` dataclass bundling
  ADV, spread, min_size, and max_participation for a single symbol.
- [`models.md`](models.md): the three shipped fill models,
  `FlatSlippage`, `LinearImpact`, and `SquareRootImpact`, plus the
  `compute_slippage(size, adv, side)` protocol for user-defined models.
- [`execute.md`](execute.md): `execute_fill`, `execute_fill_panel`,
  `liquidity_mask`, and the `FillResult` dataclass, with the full
  reason-code table.

## Public API

```python
from kuant.backtest.liquidity import (
    LiquidityProfile,
    FlatSlippage,
    LinearImpact,
    SquareRootImpact,
    FillResult,
    execute_fill,
    execute_fill_panel,
    liquidity_mask,
)
```

Four things live here:

1. `LiquidityProfile`: per-security metadata (`adv_series`,
   `spread_series`, `min_size`, `max_participation`).
2. The `FillModel` trio: `FlatSlippage(bps)`, `LinearImpact(k)`,
   `SquareRootImpact(k)`.
3. `execute_fill` and its batch cousin `execute_fill_panel`, returning
   `FillResult` records with a categorical `reason`.
4. `liquidity_mask`, a boolean per-date gate on ADV threshold.

## Compose with lifecycle

The "can this order fill today?" gate is the AND of the lifecycle
mask and the liquidity mask:

```python
from kuant.backtest.lifecycle import tradeable_mask
from kuant.backtest.liquidity import liquidity_mask

can_trade = tradeable_mask(idx, lc) & liquidity_mask(idx, profile)
```

`tradeable_mask` rejects rows outside the security's listing window;
`liquidity_mask` rejects rows where ADV is NaN, zero, or below the
configured floor. The two are orthogonal by construction: a name can
be listed but untradeable (halted, zero volume), and vice versa
(pre-listing rows never have ADV).

## Shared kernel contract

Follows the standard [kuant kernel
contract](../README.md#shared-kernel-contract):

- Backend preserved for the mask kernels (numpy in, numpy out).
- Errors are `KuantValueError`, `KuantShapeError`. Every message names
  the kernel, the offending value, a stable code like
  `KE-VAL-POSITIVE`, and a one-line fix.
- Warnings surface unreliable-but-computed cases via
  `KuantNumericWarning` with stable codes like `KW-LIQ-MASK-ALL-FALSE`.

## Related subpackages

- [`lifecycle/`](../lifecycle/README.md): the tradeable-window
  primitive that gates upstream of liquidity.
- [`fill/`](../fill/README.md): the `Order` and `submit_order` layer
  that routes strategy intents through `execute_fill`.
