#!/usr/bin/env bash
# run.sh — activate venv and kick off the benchmark queue.
#
# Foreground (default):    ./benchmarks/run.sh [--force|--only NAME]
# Background:              ./benchmarks/run.sh --detach [other args]
#     ↳ logs to benchmarks/queue.log; results still stream to results/latest.jsonl

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
VENV="$REPO/.venv"
PY="$VENV/bin/python"

if [[ ! -x "$PY" ]]; then
    echo "error: expected venv at $VENV but $PY is missing" >&2
    echo "hint: create it via  python3 -m venv $VENV && $VENV/bin/pip install -e ." >&2
    exit 1
fi

# Split --detach out of the arg list; everything else passes through to queue.py.
DETACH=0
FORWARD=()
for arg in "$@"; do
    if [[ "$arg" == "--detach" ]]; then
        DETACH=1
    else
        FORWARD+=("$arg")
    fi
done

LOG="$HERE/queue.log"

if [[ $DETACH -eq 1 ]]; then
    echo "queue running in background; tail -f $LOG"
    echo "results stream to $HERE/results/latest.jsonl"
    nohup "$PY" "$HERE/queue.py" "${FORWARD[@]}" >>"$LOG" 2>&1 &
    echo "PID: $!"
    exit 0
fi

exec "$PY" "$HERE/queue.py" "${FORWARD[@]}"
