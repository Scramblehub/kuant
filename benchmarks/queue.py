#!/usr/bin/env python3
"""Benchmark queue runner.

Scans `benchmarks/suites/bench_*.py`, runs any that have changed since
the last completed run, and streams results to `results/latest.jsonl`.

Design goals:
- Non-blocking: streams each benchmark result as it completes; don't
  buffer everything until the end.
- Restart-safe: content-hash manifest of completed suites lets us skip
  work already done.
- Extensible: drop a new file in suites/, next `run.sh` picks it up.
- Concurrent-safe: a lockfile prevents two runs stepping on each other.
- Interruptible: Ctrl-C cleanly abandons the current suite; prior
  results in latest.jsonl remain intact.

Usage:
    python benchmarks/queue.py               # incremental run
    python benchmarks/queue.py --force       # rerun everything
    python benchmarks/queue.py --only NAME   # single suite (basename)
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).parent
SUITES = HERE / "suites"
RESULTS = HERE / "results"
MANIFEST = RESULTS / "manifest.json"
LATEST = RESULTS / "latest.jsonl"
LOCKFILE = HERE / ".queue.lock"


def _log(msg: str) -> None:
    """Progress log for the operator; not written to the results stream."""
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{stamp}] queue: {msg}", flush=True)


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "-C", str(HERE.parent), "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except subprocess.CalledProcessError:
        return "unknown"


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _load_manifest() -> dict:
    if not MANIFEST.exists():
        return {"completed": {}}
    return json.loads(MANIFEST.read_text())


def _save_manifest(m: dict) -> None:
    tmp = MANIFEST.with_suffix(".tmp")
    tmp.write_text(json.dumps(m, indent=2))
    tmp.replace(MANIFEST)


def _acquire_lock():
    """Non-blocking lock. Raise if another queue.py is already running."""
    LOCKFILE.parent.mkdir(parents=True, exist_ok=True)
    f = open(LOCKFILE, "w")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        raise RuntimeError(
            f"another queue.py is running (lockfile {LOCKFILE}). "
            "Wait for it, or delete the lockfile if stale."
        ) from exc
    return f


def _run_suite(suite_path: Path) -> list[dict]:
    """Run one suite via pytest-benchmark, parse machine-readable output.

    Writes a temporary JSON file (pytest-benchmark --benchmark-json), reads it,
    returns a list of per-benchmark records.
    """
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as tmp:
        json_path = tmp.name

    try:
        # Invoke pytest with the benchmark plugin. --no-header keeps the
        # streaming line-by-line output readable in run.sh's tee target.
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "--no-header",
            "-q",
            f"--benchmark-json={json_path}",
            "--benchmark-columns=median,min,max,stddev,rounds",
            "--benchmark-warmup=off",
            "-p",
            "no:cacheprovider",
            str(suite_path),
        ]
        subprocess.run(cmd, check=False, cwd=str(HERE.parent))

        raw = json.loads(Path(json_path).read_text())
    finally:
        try:
            os.unlink(json_path)
        except FileNotFoundError:
            pass

    # Normalize each benchmark run into our JSONL record schema.
    records = []
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for entry in raw.get("benchmarks", []):
        stats = entry["stats"]
        records.append(
            {
                "timestamp": now,
                "git_sha": _git_sha(),
                "suite": suite_path.stem,
                "benchmark": entry["name"],
                "mean_ns": int(stats["mean"] * 1e9),
                "median_ns": int(stats["median"] * 1e9),
                "stddev_ns": int(stats["stddev"] * 1e9),
                "min_ns": int(stats["min"] * 1e9),
                "max_ns": int(stats["max"] * 1e9),
                "rounds": stats["rounds"],
                "hostname": platform.node(),
                "python": platform.python_version(),
            }
        )
    return records


def _append_records(records: list[dict]) -> None:
    LATEST.parent.mkdir(parents=True, exist_ok=True)
    with LATEST.open("a") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
            f.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="kuant benchmark queue")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun all suites even if their hash is in the manifest.",
    )
    parser.add_argument(
        "--only",
        metavar="NAME",
        help="Run only one suite by basename (e.g. bench_core).",
    )
    args = parser.parse_args(argv)

    if not SUITES.exists():
        _log(f"no suites/ dir at {SUITES}; nothing to do")
        return 0

    lock = _acquire_lock()
    try:
        manifest = _load_manifest()
        completed = manifest.get("completed", {})

        # Discover suites; sort for stable ordering.
        candidates = sorted(SUITES.glob("bench_*.py"))
        if args.only:
            candidates = [c for c in candidates if c.stem == args.only]
            if not candidates:
                _log(f"no suite matches --only {args.only!r}")
                return 1

        n_ran = 0
        for suite_path in candidates:
            h = _hash_file(suite_path)
            skip = (not args.force) and completed.get(suite_path.stem) == h
            if skip:
                _log(f"skip {suite_path.name} (hash unchanged)")
                continue

            _log(f"run  {suite_path.name} (hash {h})")
            try:
                records = _run_suite(suite_path)
            except KeyboardInterrupt:
                _log(f"interrupted while running {suite_path.name}; keeping prior results")
                return 130
            except Exception as exc:
                _log(f"ERROR running {suite_path.name}: {exc}")
                continue

            _append_records(records)
            completed[suite_path.stem] = h
            _save_manifest({"completed": completed})
            n_ran += 1
            _log(f"done {suite_path.name} — {len(records)} benchmarks recorded")

        _log(f"queue complete — {n_ran} suite(s) ran, results in {LATEST}")
        return 0
    finally:
        try:
            lock.close()
            LOCKFILE.unlink(missing_ok=True)
        except OSError:
            pass


if __name__ == "__main__":
    sys.exit(main())
