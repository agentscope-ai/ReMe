#!/usr/bin/env bash
# Sequential BEAM evaluation: 100K eval-only → 1M full build
# Usage: bash benchmark/beam/run_beam_100k_eval_1m_build.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CONFIG="$PROJECT_ROOT/benchmark/beam/config.yaml"
LOG_DIR="$PROJECT_ROOT/logs"
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/beam_sequential_${TS}.log"
mkdir -p "$LOG_DIR"

echo "=== BEAM Sequential Evaluation ===" | tee "$LOG_FILE"
echo "Started at: $(date)" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"

# ── Phase 1: BEAM 100K eval-only (20 workers) ──
echo "" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
echo "  Phase 1: BEAM 100K eval-only" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"

# Write 100K eval-only config
cat > "$CONFIG" <<'EOF'
# BEAM evaluation configuration
# This file controls what/how to evaluate.

dataset:
  beam_root: "benchmark/beam/dataset/BEAM"   # BEAM dataset root
  chat_size: "100K"                      # 100K | 500K | 1M | 10M (dataset variant)
  case_ids: []                          # empty = all cases; or ["1", "2", "3"]
  start_index: 0                        # first case index (for pagination)
  num_items: 0                          # 0 = all cases; >0 = limit
  workspace_root: "benchmark/beam/workspaces/beam"  # workspace root for case workspaces

evaluation:
  num_workers: 20                       # 0 = auto; 1 = sequential; >1 = parallel (per-case)
  compress_session: false               # true = compress session chunks in search_v2 (query-aware); false = no compression

reme:
  config: "beam.yaml"                   # reme config (in reme/config/)

output:
  dir: "benchmark/beam/results"
  log_dir: "logs"                       # log directory (relative to project root)
  log_prefix: "beam"                    # benchmark name used in log filenames
  log_to_console: true
  log_to_file: true
EOF

echo "Config written for 100K eval-only" | tee -a "$LOG_FILE"
cd "$PROJECT_ROOT"
conda run -n reme --no-capture-output python benchmark/beam/run.py --eval_only 2>&1 | tee -a "$LOG_FILE"
PHASE1_EXIT=${PIPESTATUS[0]}

if [ $PHASE1_EXIT -ne 0 ]; then
    echo "Phase 1 (100K eval-only) FAILED with exit code $PHASE1_EXIT" | tee -a "$LOG_FILE"
    exit $PHASE1_EXIT
fi

echo "" | tee -a "$LOG_FILE"
echo "Phase 1 (100K eval-only) completed at $(date)" | tee -a "$LOG_FILE"

# ── Phase 2: BEAM 1M full build (20 workers) ──
echo "" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
echo "  Phase 2: BEAM 1M full build" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"

# Write 1M full build config
cat > "$CONFIG" <<'EOF'
# BEAM evaluation configuration
# This file controls what/how to evaluate.

dataset:
  beam_root: "benchmark/beam/dataset/BEAM"   # BEAM dataset root
  chat_size: "1M"                        # 100K | 500K | 1M | 10M (dataset variant)
  case_ids: []                          # empty = all cases; or ["1", "2", "3"]
  start_index: 0                        # first case index (for pagination)
  num_items: 0                          # 0 = all cases; >0 = limit
  workspace_root: "benchmark/beam/workspaces/beam"  # workspace root for case workspaces

evaluation:
  num_workers: 20                       # 0 = auto; 1 = sequential; >1 = parallel (per-case)
  compress_session: false               # true = compress session chunks in search_v2 (query-aware); false = no compression

reme:
  config: "beam.yaml"                   # reme config (in reme/config/)

output:
  dir: "benchmark/beam/results"
  log_dir: "logs"                       # log directory (relative to project root)
  log_prefix: "beam"                    # benchmark name used in log filenames
  log_to_console: true
  log_to_file: true
EOF

echo "Config written for 1M full build" | tee -a "$LOG_FILE"
conda run -n reme --no-capture-output python benchmark/beam/run.py 2>&1 | tee -a "$LOG_FILE"
PHASE2_EXIT=${PIPESTATUS[0]}

if [ $PHASE2_EXIT -ne 0 ]; then
    echo "Phase 2 (1M full build) FAILED with exit code $PHASE2_EXIT" | tee -a "$LOG_FILE"
    exit $PHASE2_EXIT
fi

echo "" | tee -a "$LOG_FILE"
echo "Phase 2 (1M full build) completed at $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "=== ALL DONE at $(date) ===" | tee -a "$LOG_FILE"
