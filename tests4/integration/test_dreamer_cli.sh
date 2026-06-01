#!/usr/bin/env bash
# dreamer CLI integration test (option B).
#
# Seeds a rich workspace via _dreamer_fixture.py, starts `reme start`
# bound to that vault, reindexes so Phase 2 recall can hit the pre-
# seeded digest nodes, then calls `reme dream`.
#
# Usage (from anywhere):
#   VAULT_PATH=/tmp/reme-dreamer-test bash tests4/integration/test_dreamer_cli.sh
#   VAULT_PATH=/tmp/reme-dreamer-test bash tests4/integration/test_dreamer_cli.sh daily/2026-05-28/auth-refactor/notes.md
#
# Defaults:
#   VAULT_PATH unset → /tmp/reme-dreamer-test
#   Workspace seeded on first run (idempotent).
#
# Required env (from .env or shell):
#   LLM_API_KEY, LLM_BASE_URL, LLM_MODEL_NAME
set -euo pipefail

VAULT="${VAULT_PATH:-/tmp/reme-dreamer-test}"
INTEGRATION_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$INTEGRATION_DIR/../.." && pwd)"
LOG="/tmp/test_dreamer_cli_server.log"

# Resolve input from arg, else default to fixture's input path.
DEFAULT_INPUT="$(python -c "import sys; sys.path.insert(0, '$INTEGRATION_DIR'); from _dreamer_fixture import INPUT_PATH; print(INPUT_PATH)")"
INPUT="${1:-$DEFAULT_INPUT}"

mkdir -p "$VAULT"
echo "--- seeding fixture under $VAULT"
python "$INTEGRATION_DIR/_dreamer_fixture.py" "$VAULT"

cd "$REPO"

echo ""
echo "--- starting reme server (log: $LOG)"
reme start "vault_dir=$VAULT" >"$LOG" 2>&1 &
SERVER_PID=$!
trap 'echo "--- stopping reme server (pid $SERVER_PID)"; kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true' EXIT

echo "--- waiting for server"
for _ in $(seq 1 30); do
  if curl -s -o /dev/null http://localhost:8000/docs 2>/dev/null; then
    echo "--- server up"
    break
  fi
  sleep 0.5
done

echo ""
echo "--- reindexing vault so Phase 2 recall has something to hit"
reme reindex

echo ""
echo "=== reme dream path=$INPUT ==="
reme dream "path=$INPUT"
echo ""

echo "=== digest/ tree after dream ==="
if [ -d "$VAULT/digest" ]; then
  find "$VAULT/digest" -name "*.md" | sort | while read -r f; do
    echo ""
    echo "--- ${f#$VAULT/} ---"
    cat "$f"
  done
else
  echo "  (no digest/ created)"
fi
