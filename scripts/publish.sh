#!/usr/bin/env bash
# publish — upload dist/* to TestPyPI or PyPI using tokens from secret/*.env
#
# Usage:
#   ./scripts/publish.sh test          # uploads to TestPyPI
#   ./scripts/publish.sh prod          # uploads to real PyPI
#
# Reads tokens from:
#   secret/testPyPI.env  →  Kuant_key=pypi-xxx
#   secret/PyPI.env      →  Kuant_key_main=pypi-yyy
#
# SAFETY:
#   - Refuses to run if secret/ is not gitignored.
#   - Does not echo the token to stdout.
#   - Uses TWINE_USERNAME=__token__ + TWINE_PASSWORD (twine env-var API).

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

TARGET="${1:-}"

case "$TARGET" in
    test|testpypi)
        SECRET_FILE="secret/testPyPI.env"
        VAR_NAME="Kuant_key"
        TWINE_REPO_ARG=(--repository testpypi)
        NAME="TestPyPI"
        ;;
    prod|pypi)
        SECRET_FILE="secret/PyPI.env"
        VAR_NAME="Kuant_key_main"
        TWINE_REPO_ARG=()
        NAME="PyPI (production)"
        ;;
    *)
        echo "Usage: $0 [test|prod]"
        echo "  test → TestPyPI"
        echo "  prod → real PyPI"
        exit 2
        ;;
esac

# Safety: never run if secret file isn't ignored.
if ! git check-ignore "$SECRET_FILE" >/dev/null 2>&1; then
    echo "REFUSING: $SECRET_FILE is not gitignored." >&2
    echo "Add it to .gitignore before running this script." >&2
    exit 1
fi

if [[ ! -f "$SECRET_FILE" ]]; then
    echo "ERROR: $SECRET_FILE not found." >&2
    exit 1
fi

if [[ ! -d "dist" || -z "$(ls dist/ 2>/dev/null)" ]]; then
    echo "ERROR: dist/ is empty. Run 'python -m build' first." >&2
    exit 1
fi

# Source the env file (KEY=VALUE format). Only exports lines matching NAME=VALUE.
set -a
# shellcheck disable=SC1090
source "$SECRET_FILE"
set +a

# Pull the token value into TWINE_PASSWORD. Indirect expansion (${!VAR}).
TOKEN="${!VAR_NAME:-}"
if [[ -z "$TOKEN" ]]; then
    echo "ERROR: variable '$VAR_NAME' not set in $SECRET_FILE." >&2
    exit 1
fi

export TWINE_USERNAME="__token__"
export TWINE_PASSWORD="$TOKEN"

# Confirm which artifacts will be uploaded and where.
echo "Uploading to $NAME:"
ls -1 dist/ | sed 's/^/  /'
echo

# Run twine. --non-interactive prevents any surprise prompt.
python -m twine upload --non-interactive "${TWINE_REPO_ARG[@]}" dist/*

# Clear token from environment (belt + braces; process is about to exit anyway).
unset TWINE_PASSWORD
