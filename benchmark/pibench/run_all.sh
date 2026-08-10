#!/bin/bash
# Run all 5 personas with the ReMe agent, PARALLEL at a time (default 2).
# Each persona's tasks follow data/{persona}/episode.yaml order.
#
# Usage:
#   bash run_all.sh                  # FRESH official run: wipes ALL personas'
#                                    # ReMe memory/outputs/trace logs first,
#                                    # then runs everything from scratch.
#   bash run_all.sh --resume         # Checkpoint continuation: no wipe; every
#                                    # persona skips already-completed tasks.
#   bash run_all.sh --parallel 1     # sequential (original behavior)
#   bash run_all.sh --skip-eval      # run phase only
#
# Memory-wipe vs resume conflict resolution:
#   The full ReMe memory wipe happens ONLY here, ONLY in fresh mode (the
#   default), and ONLY before any service/bridge starts. --resume never
#   wipes; run_persona.sh then additionally performs a surgical cleanup of
#   residual memory belonging to interrupted (to-be-re-run) tasks, so a
#   resumed run keeps all completed-task memory but never inherits a partial
#   task's own answer. The two modes are mutually exclusive.
set -uo pipefail

SUITE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PERSONAS=(researcher marketer law_trainee pharmacist Financier)
TRACE_ROOT="${HOME}/.nanobot/trace_logs"

PARALLEL=2
MODE="fresh"
PASS_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --parallel)
            PARALLEL="${2:-}"; shift 2 || true
            case "$PARALLEL" in (""|*[!0-9]*) echo "--parallel needs a positive integer"; exit 2 ;; esac
            [ "$PARALLEL" -lt 1 ] && PARALLEL=1
            [ "$PARALLEL" -gt ${#PERSONAS[@]} ] && PARALLEL=${#PERSONAS[@]}
            ;;
        --resume)
            if [ "$MODE" = "fresh_set" ]; then echo "--fresh and --resume are mutually exclusive"; exit 2; fi
            MODE="resume"; shift ;;
        --fresh)
            if [ "$MODE" = "resume" ]; then echo "--fresh and --resume are mutually exclusive"; exit 2; fi
            MODE="fresh_set"; shift ;;
        --skip-eval) PASS_ARGS+=(--skip-eval); shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done
[ "$MODE" = "fresh_set" ] && MODE="fresh"

START_TS=$(date +%Y%m%d_%H%M%S)
SUMMARY_LOG="${SUITE_DIR}/logs/run_all_${START_TS}.summary"
mkdir -p "${SUITE_DIR}/logs"

echo "############################################################"
echo "# reme_eval suite | mode=${MODE} parallel=${PARALLEL} | ${START_TS}"
echo "############################################################"

# ─── Fresh mode: suite-level wipe BEFORE anything starts ──────────────
if [ "$MODE" = "fresh" ]; then
    echo "[fresh] wiping ALL personas' memory workspaces, outputs and trace logs..."
    for persona in "${PERSONAS[@]}"; do
        rm -rf "${SUITE_DIR}/reme_workspace/${persona}"
        rm -rf "${SUITE_DIR}/outputs/reme/${persona}"
        rm -rf "${TRACE_ROOT}/reme/${persona}"
        rm -rf "${SUITE_DIR}/nanobot_workspace/${persona}"
    done
    echo "[fresh] wipe done."
else
    echo "[resume] no memory wipe; personas resume after their last completed task."
fi

# ─── Run personas in batches of PARALLEL ──────────────────────────────
STATUS_LIST=()
OVERALL_START=$(date +%s)
TOTAL=${#PERSONAS[@]}

for ((i = 0; i < TOTAL; i += PARALLEL)); do
    BATCH=("${PERSONAS[@]:i:PARALLEL}")
    BATCH_PIDS=()
    BATCH_NAMES=()
    echo ""
    echo "============================================================"
    echo "# BATCH $(( i / PARALLEL + 1 )): ${BATCH[*]}   started $(date '+%F %T')"
    echo "============================================================"
    for persona in "${BATCH[@]}"; do
        bash "${SUITE_DIR}/run_persona.sh" "${persona}" --resume ${PASS_ARGS[@]+"${PASS_ARGS[@]}"} \
            > "${SUITE_DIR}/logs/suite_${persona}.log" 2>&1 &
        BATCH_PIDS+=($!)
        BATCH_NAMES+=("$persona")
    done
    for j in $(seq 0 $(( ${#BATCH[@]} - 1 ))); do
        pid=${BATCH_PIDS[$j]}
        persona=${BATCH_NAMES[$j]}
        if wait "$pid"; then
            STATUS_LIST+=("${persona}: OK")
        else
            rc=$?
            STATUS_LIST+=("${persona}: FAILED rc=${rc}")
            echo "[run_all] ${persona} FAILED (rc=${rc}); see logs/suite_${persona}.log"
        fi
    done
done

total=$(( $(date +%s) - OVERALL_START ))
echo ""
echo "================ FINAL SUMMARY (${total}s total) ================" | tee -a "${SUMMARY_LOG}"
for line in "${STATUS_LIST[@]}"; do
    echo "  ${line}" | tee -a "${SUMMARY_LOG}"
done
echo "Summary: ${SUMMARY_LOG}"
