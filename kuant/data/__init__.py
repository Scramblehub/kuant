"""kuant.data — data-shape primitives.

Handles the alignment, aggregation, and adjustment operations that
every downstream kernel implicitly assumes have already been done
correctly. Not I/O — no HTTP, no file readers, no vendor-specific
parsers. The kernel scope is: given clean-ish input, produce the
shape you need for the rest of kuant.

Serialized outputs default to parquet (columnar, typed, compressed).
Any dataclass returned by a kernel here that represents tabular
output ships with a `.to_parquet(path)` convenience via lazy `pyarrow`.
"""

from kuant.data.align import AlignResult, align

__all__ = ["AlignResult", "align"]
