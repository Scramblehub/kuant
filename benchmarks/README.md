# kuant benchmarks

Micro-benchmarks per subpackage, driven by a file-based queue. You can
add new benchmark files while a run is in progress; they get picked up
on the next scan.

## Layout

```
benchmarks/
├── suites/            One file per subpackage. pytest-benchmark tests.
│   ├── bench_core.py
│   ├── bench_options.py
│   ├── bench_stats.py
│   ├── bench_qm.py
│   ├── bench_sindy.py
│   ├── bench_topology.py
│   ├── bench_data.py
│   ├── bench_edgecases.py
│   └── bench_signals.py
├── results/           Append-only JSONL logs (one line per benchmark).
├── queue.py           The runner. Scans suites, executes one at a time.
└── run.sh             Shell wrapper (activates venv, runs queue.py).
```

## Queue semantics

Each suite is a standalone pytest-benchmark file. The runner:

1. Finds `benchmarks/suites/bench_*.py`.
2. For each, computes a content hash and checks a manifest of completed hashes.
3. Runs any that are new or changed. `--force` overrides.
4. Streams results to `results/latest.jsonl` as each completes.
5. Holds a lockfile so two concurrent invocations do not collide.

## Adding a new benchmark

1. Edit or create a file in `suites/`. pytest-benchmark style:

   ```python
   def test_bench_bscall_scalar(benchmark):
       benchmark(bscall, 100.0, 100.0, 1.0, 0.05, 0.20)
   ```

2. Run the queue: `./benchmarks/run.sh` (foreground), or
   `./benchmarks/run.sh --detach` for background (logs to
   `benchmarks/queue.log`).

3. You can add more suites while the queue is running. They get picked
   up on the next scan. The currently-running suite is not interrupted.

4. Watch results live:

   ```bash
   tail -f benchmarks/results/latest.jsonl
   ```

## Result format

Each JSON line:

```json
{
  "timestamp": "2026-07-04T02:33:11Z",
  "git_sha": "abc1234",
  "suite": "bench_core",
  "benchmark": "test_bench_bscall_scalar",
  "mean_ns": 12500,
  "stddev_ns": 300,
  "median_ns": 12400,
  "min_ns": 12100,
  "max_ns": 13800,
  "rounds": 1000,
  "hostname": "...",
  "python": "3.12.1"
}
```

## GPU benchmarks

Tests that call cupy need a CUDA device. They skip gracefully via
`pytest.importorskip("cupy")` at the top of the file. GPU numbers are
recorded alongside CPU numbers so you can compute speedup ratios from
`results/latest.jsonl`.

## Forcing a rerun

```bash
./benchmarks/run.sh --force            # rerun everything
./benchmarks/run.sh --only bench_core  # rerun one suite
```
