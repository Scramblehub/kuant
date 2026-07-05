# position - Position dataclass

## Purpose

Track one symbol's signed size, volume-weighted average entry cost,
and cumulative realized P&L under netting semantics. Every fill
routes through `apply_fill`, which is the sole state machine on the
type. See [README](README.md) for how netting compares against
per-lot alternatives.

## Public API

```python
from kuant.backtest.position import Position
```

### `Position`

Mutable dataclass. One instance per traded symbol.

```python
Position(
    symbol: str,
    size: float = 0.0,
    avg_cost: float = 0.0,
    realized_pnl: float = 0.0,
)
```

- `symbol`: opaque identifier used by the caller. Not validated
  against any universe.
- `size`: signed quantity currently held. `> 0` long, `< 0` short,
  `== 0` flat.
- `avg_cost`: volume-weighted average entry price for the OPEN
  portion. Undefined when `size == 0`; the invariant is that
  `avg_cost` becomes meaningful again as soon as `size` returns to
  nonzero.
- `realized_pnl`: cumulative P&L across every close and reversal on
  this symbol. Never reset.

### Methods

- `apply_fill(size_filled: float, price: float) -> None`. In-place
  update. Signed convention: positive `size_filled` is a buy,
  negative is a sell.
- `market_value(price: float) -> float`. `size * price`; signed.
- `unrealized_pnl(price: float) -> float`. P&L if the open size were
  closed at `price` right now.
- `total_pnl(price: float) -> float`. `realized_pnl +
  unrealized_pnl(price)`.
- `summary() -> str`. Human-readable snapshot for logs.

## Design decisions

### 1. Four-case state machine on `apply_fill`

Every fill lands in exactly one of four disjoint branches. The
kernel validates first, then routes.

1. **Zero fill.** `size_filled == 0.0` short-circuits and returns
   without mutating state. Rejected fills already have
   `size_filled == 0`, so `PortfolioState.apply_fill` can hand them
   straight in without pre-filtering.
2. **Start from flat.** `self.size == 0.0`. Adopt the fill: set
   `size = size_filled`, `avg_cost = price`, leave `realized_pnl`
   untouched.
3. **Same direction.** Both signs positive or both negative.
   Weighted-average the entry cost against the new size:
   `avg_cost = (avg_cost * size + price * size_filled) / (size +
   size_filled)`. `realized_pnl` is untouched; no position was
   closed.
4. **Opposite direction.** Realize P&L on the closed quantity
   `min(|size|, |size_filled|)` at the difference between `price`
   and `avg_cost`. If the fill exactly closes the position, reset
   to flat. If the fill is smaller than the open size, keep
   `avg_cost` on the remainder. If the fill exceeds the open size,
   flip past zero: the residual `|size_filled| - |size|` opens a
   new position on the opposite side, with `avg_cost = price`.

The four cases are mutually exclusive and cover every valid input.
Non-finite `size_filled` or non-finite / non-positive `price` raises
`KuantValueError` before any branch runs.

### 2. Realized P&L uses the CURRENT side, not the fill's side

When a position is closed or flipped, `realized_pnl` accumulates
`side_sign * closed_qty * (price - avg_cost)`, where `side_sign` is
`+1` if the current position is long and `-1` if it is short. The
fill's sign only decides how much is being closed; it does not enter
the P&L formula. A short position closed by a buy is profitable when
`price < avg_cost`, and `side_sign = -1` produces the correct sign
without a separate short-side branch.

### 3. `avg_cost` is preserved on a partial close

If a fill closes only part of the open position, the remaining size
carries the same `avg_cost` as before the fill. Rationale: the
average entry cost is a property of the still-held shares, not of
the ones that just left. This matches how a broker reports remaining
basis after a partial sell.

On a full flip, the remaining size is entered fresh at the fill
`price`; the old `avg_cost` has no relevance to the new opposite
side.

### 4. Netting collapses long-short segregation

There is one `Position` per symbol. A caller who buys 100 and then
sells 150 ends the sequence short 50, not long 100 and short 150.
See [README](README.md) for the rationale. Callers who need
per-lot detail consume the `FillReport` stream and reconstruct lots
outside this type.

### 5. Mutable dataclass, no `__post_init__` validation

`Position` is a plain mutable dataclass. Direct construction with
`Position(symbol='XYZ')` is the intended entry point; the state
machine lives on `apply_fill`. Callers who want to seed a Position
from a broker snapshot are expected to know what they are doing.
The invariants that matter (`size` finite, `avg_cost` finite,
`price` finite and strictly positive) are enforced at the point of
mutation, not at construction.

## Edge cases

| Condition | Behavior |
| --- | --- |
| `size_filled == 0.0` | no-op; return immediately |
| `size_filled` non-finite | raises `KuantValueError [KE-POS-SIZE-INVALID]` |
| `price` non-finite or `<= 0.0` | raises `KuantValueError [KE-POS-PRICE-INVALID]` |
| Fill exactly closes the position | `size = 0.0`, `avg_cost = 0.0`, `realized_pnl` accumulated |
| Fill flips past zero | remainder opens on opposite side at `avg_cost = price` |
| `market_value` called on flat position | returns `0.0` (any finite `price`) |
| `unrealized_pnl` on flat position | returns `0.0` regardless of `price` |

## Examples

### Four-fill worked example

Long open, add same direction, partial close, flip past zero.

```python
>>> from kuant.backtest.position import Position
>>> p = Position(symbol='XYZ')
>>> p.apply_fill(size_filled=100.0, price=50.0)
>>> p.size, round(p.avg_cost, 2), round(p.realized_pnl, 2)
(100.0, 50.0, 0.0)
>>> p.apply_fill(size_filled=100.0, price=60.0)
>>> p.size, round(p.avg_cost, 2), round(p.realized_pnl, 2)
(200.0, 55.0, 0.0)
>>> p.apply_fill(size_filled=-40.0, price=70.0)
>>> p.size, round(p.avg_cost, 2), round(p.realized_pnl, 2)
(160.0, 55.0, 600.0)
>>> p.apply_fill(size_filled=-200.0, price=80.0)
>>> p.size, round(p.avg_cost, 2), round(p.realized_pnl, 2)
(-40.0, 80.0, 4600.0)
```

Trace:

- Fill 1 (start from flat): buy 100 at 50. `size = 100`, `avg_cost
  = 50`, no realized P&L.
- Fill 2 (same direction): buy 100 at 60. New size `200`,
  weighted-average cost `(50 * 100 + 60 * 100) / 200 = 55`.
- Fill 3 (partial close): sell 40 at 70. Closed quantity `40`,
  realized `1 * 40 * (70 - 55) = 600`. Remaining size `160`,
  `avg_cost` preserved at `55`.
- Fill 4 (flip past zero): sell 200 at 80. Closes the full open 160
  at realized `1 * 160 * (80 - 55) = 4000`; cumulative realized
  `600 + 4000 = 4600`. Residual `200 - 160 = 40` opens on the
  opposite side: `size = -40`, `avg_cost = 80`.

### Mark-to-market helpers

```python
>>> round(p.market_value(75.0), 2)
-3000.0
>>> round(p.unrealized_pnl(75.0), 2)
200.0
>>> round(p.total_pnl(75.0), 2)
4800.0
```

At a mark price of 75 the short 40 carries a signed market value of
`40 * 75 * -1 = -3000`. The short is profitable because
`avg_cost = 80 > price = 75`: `unrealized = -1 * 40 * (75 - 80) =
200`. Total P&L is `realized 4600 + unrealized 200 = 4800`.

### Short-side arithmetic

Opening a short and closing it at a lower price accumulates positive
realized P&L. `side_sign = -1` handles the sign inside `apply_fill`
without a separate branch.

```python
>>> s = Position(symbol='ABC')
>>> s.apply_fill(size_filled=-100.0, price=50.0)
>>> s.size, round(s.avg_cost, 2)
(-100.0, 50.0)
>>> s.apply_fill(size_filled=100.0, price=40.0)
>>> s.size, round(s.avg_cost, 2), round(s.realized_pnl, 2)
(0.0, 0.0, 1000.0)
```

Short 100 at 50, cover at 40, realize `-1 * 100 * (40 - 50) =
1000`. Position returns to flat.

### Validation

Non-finite size or non-positive price raise before any state
mutation.

```python
>>> from kuant.errors import KuantValueError
>>> p = Position(symbol='XYZ')
>>> p.apply_fill(size_filled=100.0, price=50.0)
>>> try:
...     p.apply_fill(size_filled=float('nan'), price=50.0)
... except KuantValueError as e:
...     'KE-POS-SIZE-INVALID' in str(e)
True
>>> try:
...     p.apply_fill(size_filled=10.0, price=0.0)
... except KuantValueError as e:
...     'KE-POS-PRICE-INVALID' in str(e)
True
>>> p.size, round(p.avg_cost, 2)
(100.0, 50.0)
```

## Cross-references

- [`portfolio.md`](portfolio.md): `PortfolioState` owns the
  `dict[symbol, Position]` and drives `Position.apply_fill` from
  incoming `FillReport` objects.
- `kuant.backtest.fill.submit_order`: produces the `FillReport`
  objects that the portfolio layer consumes.
- `kuant.backtest.lifecycle`: on delisting, the caller is expected
  to apply the terminal action against the equity curve rather than
  through `Position` directly; `apply_fill` refuses non-positive
  prices, so a bankruptcy mark-to-zero does not route through here.
