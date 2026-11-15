#!/bin/bash -l
set -euo pipefail

REME_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$REME_ROOT"

# Finder may not inherit the Node environment from an interactive terminal.
if ! command -v npm >/dev/null 2>&1 && [[ -s "${NVM_DIR:-$HOME/.nvm}/nvm.sh" ]]; then
  set +u
  source "${NVM_DIR:-$HOME/.nvm}/nvm.sh"
  set -u
fi

check_python() {
  [[ -x "$1" ]] && "$1" "$REME_ROOT/reme_desktop/scripts/restart.py" --check
}

if [[ -n "${REME_PYTHON:-}" ]]; then
  REME_PYTHON_BIN="$REME_PYTHON"
  check_python "$REME_PYTHON_BIN" || exit 1
else
  REME_PYTHON_BIN=""
  for candidate in \
    "$REME_ROOT/.venv/bin/python" \
    "${CONDA_PREFIX:-/nonexistent}/bin/python" \
    "$HOME/miniconda3/bin/python" \
    "$(command -v python3 || true)"; do
    if check_python "$candidate"; then
      REME_PYTHON_BIN="$candidate"
      break
    fi
  done
fi

if [[ -z "$REME_PYTHON_BIN" ]]; then
  printf '%s\n' "找不到 ReMe Python 环境。请先激活开发环境，或设置 REME_PYTHON=/path/to/python。" >&2
  exit 1
fi

printf '使用 Python：%s\n' "$REME_PYTHON_BIN"
exec "$REME_PYTHON_BIN" "$REME_ROOT/reme_desktop/scripts/restart.py"
