# kuant benchmarks

Performance comparisons: CPU numpy fallback vs GPU cupy path per kernel.

Each kernel gets a benchmark script that:

1. Generates representative-sized inputs
2. Runs CPU path N times, records median
3. Runs GPU path N times, records median
4. Reports speedup ratio + throughput (ops/sec)

Benchmarks run under `pytest -m benchmark`. Results are informational, not
correctness — see `tests/` for correctness validation.
