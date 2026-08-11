#!/bin/bash
# Run the full pi-bench evaluation for ONE persona with the ReMe agent.
# Tasks follow data/{persona}/episode.yaml order (runner-native).
#
# Usage: bash run_persona.sh <persona> [--fresh|--resume] [--skip-eval]
#
# Modes (default: --resume):
#   --resume  Checkpoint continuation. Never wipes memory. Tasks already
#             finished (SUCCESS/MAX_TURNS/TIMEOUT in the task history logs)
#             are skipped via repeated --task-id flags. Before starting, any
#             residual memory of tasks that are about to be RE-RUN (partial
#             sessions from an interrupted run) is surgically removed by
#             resume.py cleanup, so re-runs don't inherit leaked answers.
#   --fresh   Wipes THIS persona's ReMe memory, outputs and trace logs first,
#             then runs all tasks from scratch.
# The two flags are mutually exclusive. A full multi-persona memory wipe is a
# suite-level action of `run_all.sh` (fresh mode), never done here implicitly.
set -uo pipefail

SUITE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRACE_ROOT="${HOME}/.nanobot/trace_logs"

# ─── External dependencies (pi-bench / ReMe are NOT bundled; see README) ──
if [ ! -f "${SUITE_DIR}/env.sh" ]; then
    echo "env.sh not found. Run: cp env.sh.example env.sh  (then fill in the TODO items)"
    exit 1
fi
source "${SUITE_DIR}/env.sh"

PIBENCH_DIR="${PI_BENCH_ROOT:-}"
if [ -z "${PIBENCH_DIR}" ] || [ ! -f "${PIBENCH_DIR}/src/main.py" ]; then
    echo "PI_BENCH_ROOT is unset or invalid (src/main.py not found). Set it in env.sh."
    exit 1
fi
if [ ! -x "${PIBENCH_DIR}/.venv/bin/python" ] || [ ! -x "${PIBENCH_DIR}/.venv/bin/appworld" ]; then
    echo "pi-bench venv incomplete: ${PIBENCH_DIR}/.venv must provide python + appworld (see README setup)."
    exit 1
fi
if [ ! -x "${REME_DIR}/.venv/bin/python" ]; then
    echo "ReMe venv not found: ${REME_DIR}/.venv/bin/python (check REME_DIR in env.sh)"
    exit 1
fi
if [ ! -e "${SUITE_DIR}/data" ]; then
    echo 'Benchmark data not linked. Run: ln -s "$PI_BENCH_ROOT/data" data'
    exit 1
fi

# ─── Pre-flight: files the runner needs before any service starts ─────
MODEL_CONFIG="${SUITE_DIR}/config/models/reme.yaml"
HISTORY_CONFIG="${SUITE_DIR}/config/bench/evaluation/trace_history.yaml"
if [ ! -f "${MODEL_CONFIG}" ]; then
    echo "Model config not found: ${MODEL_CONFIG} (see README directory layout)."
    exit 1
fi
if [ ! -f "${HISTORY_CONFIG}" ]; then
    echo "Trace history config not found: ${HISTORY_CONFIG}"
    echo "pi-bench requires config/bench/evaluation/trace_history.yaml; see README."
    exit 1
fi

APPWORLD_DIR="${PIBENCH_DIR}/third_party/appworld"
PI_PYTHON="${PIBENCH_DIR}/.venv/bin/python"
APPWORLD_BIN="${PIBENCH_DIR}/.venv/bin/appworld"
# resume.py runs on the ReMe venv so it can reuse ReMe's daily-index rebuild.
REME_PYTHON="${REME_DIR}/.venv/bin/python"

PERSONA="${1:-}"
if [ -z "$PERSONA" ]; then
    echo "Usage: $0 <persona> [--fresh|--resume] [--skip-eval]"
    exit 1
fi
shift

MODE="resume"
SKIP_EVAL=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --fresh)
            if [ "$MODE" = "resume_set" ]; then echo "--fresh and --resume are mutually exclusive"; exit 2; fi
            MODE="fresh"; shift ;;
        --resume)
            if [ "$MODE" = "fresh" ]; then echo "--fresh and --resume are mutually exclusive"; exit 2; fi
            MODE="resume_set"; shift ;;
        --skip-eval) SKIP_EVAL=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done
[ "$MODE" = "resume_set" ] && MODE="resume"

# ─── Per-persona ports (pi-bench AGENTS.md convention) ────────────────
# REME_PORT: ReMe's internal HTTP service; must be unique per concurrent bridge.
case "$PERSONA" in
    marketer)     API_PORT=9001; MCP_PORT=10001; TEST_PORT=9998; REME_PORT=18766 ;;
    law_trainee)  API_PORT=9002; MCP_PORT=10002; TEST_PORT=9997; REME_PORT=18767 ;;
    pharmacist)   API_PORT=9003; MCP_PORT=10003; TEST_PORT=9996; REME_PORT=18768 ;;
    researcher)   API_PORT=9004; MCP_PORT=10004; TEST_PORT=9995; REME_PORT=18765 ;;
    Financier)    API_PORT=9005; MCP_PORT=10005; TEST_PORT=9994; REME_PORT=18769 ;;
    *) echo "Unknown persona: $PERSONA"; exit 1 ;;
esac

API_URL="http://127.0.0.1:${API_PORT}"
MCP_URL="http://127.0.0.1:${MCP_PORT}/mcp"
TEST_URL="http://127.0.0.1:${TEST_PORT}"
LOG_DIR="${SUITE_DIR}/logs"
mkdir -p "${LOG_DIR}"

# ─── Environment (env.sh already sourced at the top) ──────────────────
WORKSPACE_DIR="${REME_WORKSPACE_ROOT}/${PERSONA}"
NANOBOT_WORKSPACE_DIR="${SUITE_DIR}/nanobot_workspace/${PERSONA}"
mkdir -p "${WORKSPACE_DIR}" "${NANOBOT_WORKSPACE_DIR}"

echo "========================================="
echo "ReMe x Pi-Bench | persona=${PERSONA} | mode=${MODE}"
echo "  api=${API_PORT} mcp=${MCP_PORT} test=${TEST_PORT} reme=${REME_PORT}"
echo "  model=${REME_MODEL_NAME}"
echo "  memory workspace=${WORKSPACE_DIR}  (persistent)"
echo "========================================="

# ─── Fresh mode: wipe this persona's state ────────────────────────────
if [ "$MODE" = "fresh" ]; then
    echo "[fresh] wiping persona state: memory workspace, outputs, trace logs"
    rm -rf "${WORKSPACE_DIR}"
    rm -rf "${SUITE_DIR}/outputs/reme/${PERSONA}"
    rm -rf "${TRACE_ROOT}/reme/${PERSONA}"
    rm -rf "${NANOBOT_WORKSPACE_DIR}"
    mkdir -p "${WORKSPACE_DIR}" "${NANOBOT_WORKSPACE_DIR}"
fi

# ─── Resume: determine remaining tasks + clean partial memories ───────
TASK_ARGS=()
RUN_PHASE_NEEDED=true
if [ "$MODE" = "resume" ]; then
    REMAINING_JSON="$("${REME_PYTHON}" "${SUITE_DIR}/resume.py" remaining "${PERSONA}" --json)"
    if [ -z "$REMAINING_JSON" ]; then
        echo "Failed to compute remaining tasks"; exit 1
    fi
    echo "[resume] ${REMAINING_JSON}"
    REMAINING_TASKS=()
    while IFS= read -r tid_line; do
        [ -n "$tid_line" ] && REMAINING_TASKS+=("$tid_line")
    done < <("${REME_PYTHON}" "${SUITE_DIR}/resume.py" remaining "${PERSONA}" 2>/dev/null)
    if [ ${#REMAINING_TASKS[@]} -eq 0 ]; then
        RUN_PHASE_NEEDED=false
        echo "[resume] all tasks already completed; skipping run phase"
    else
        # Remove residual memory of interrupted (to-be-re-run) tasks so
        # re-runs don't get their own partial answers injected.
        "${REME_PYTHON}" "${SUITE_DIR}/resume.py" cleanup "${PERSONA}"
        for tid in "${REMAINING_TASKS[@]}"; do
            TASK_ARGS+=(--task-id "$tid")
        done
        echo "[resume] running ${#REMAINING_TASKS[@]} remaining task(s): ${REMAINING_TASKS[*]}"
    fi
fi

# ─── Port cleanup from previous runs ──────────────────────────────────
for port in ${API_PORT} ${MCP_PORT} ${TEST_PORT} ${REME_PORT}; do
    pids=$(lsof -ti :${port} 2>/dev/null || true)
    if [ -n "$pids" ]; then
        echo "Killing stale processes on port ${port}: ${pids}"
        kill -9 $pids 2>/dev/null || true
    fi
done
sleep 2

PIDS=()
cleanup() {
    echo "[${PERSONA}] cleaning up services..."
    for pid in "${PIDS[@]:-}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

wait_for_service() {
    local url="$1" name="$2" port="$3" timeout="${4:-180}"
    echo -n "  waiting for ${name}..."
    local start=$(date +%s)
    while true; do
        if curl -sf --max-time 5 "${url}" > /dev/null 2>&1; then
            echo " ready"; return 0
        fi
        if [ -n "$port" ] && lsof -ti :${port} > /dev/null 2>&1; then
            local elapsed=$(( $(date +%s) - start ))
            if [ "$elapsed" -ge 10 ]; then echo " ready (port)"; return 0; fi
        fi
        if [ $(( $(date +%s) - start )) -ge "$timeout" ]; then
            echo " TIMEOUT"; return 1
        fi
        sleep 2
    done
}

# ─── [1/5] AppWorld API ────────────────────────────────────────────────
echo "[1/5] AppWorld API (:${API_PORT})"
(cd "${APPWORLD_DIR}" && exec "${APPWORLD_BIN}" serve apis --root . \
    --port ${API_PORT}) > "${LOG_DIR}/appworld_api_${PERSONA}.log" 2>&1 &
PIDS+=($!)
if ! wait_for_service "${API_URL}/docs" "AppWorld API" "${API_PORT}" 180; then
    tail -20 "${LOG_DIR}/appworld_api_${PERSONA}.log"; exit 1
fi

# ─── [2/5] AppWorld MCP ────────────────────────────────────────────────
echo "[2/5] AppWorld MCP (:${MCP_PORT})"
TOOLS_CONFIG="${SUITE_DIR}/data/${PERSONA}/tools.yaml"
(cd "${APPWORLD_DIR}" && exec "${APPWORLD_BIN}" serve mcp http --root . \
    --remote-apis-url "${API_URL}" --port ${MCP_PORT} \
    --tools-config-file "${TOOLS_CONFIG}") > "${LOG_DIR}/appworld_mcp_${PERSONA}.log" 2>&1 &
PIDS+=($!)
if ! wait_for_service "${MCP_URL}" "AppWorld MCP" "${MCP_PORT}" 180; then
    tail -20 "${LOG_DIR}/appworld_mcp_${PERSONA}.log"; exit 1
fi

# ─── [3/5] Test Server ─────────────────────────────────────────────────
echo "[3/5] Test Server (:${TEST_PORT})"
PORT=${TEST_PORT} "${PI_PYTHON}" "${PIBENCH_DIR}/scripts/test_server.py" \
    > "${LOG_DIR}/test_server_${PERSONA}.log" 2>&1 &
PIDS+=($!)
if ! wait_for_service "${TEST_URL}/sent?after=-1" "Test Server" "${TEST_PORT}" 30; then
    tail -20 "${LOG_DIR}/test_server_${PERSONA}.log"; exit 1
fi

# ─── [4/5] ReMe Bridge (ReMe venv) ─────────────────────────────────────
echo "[4/5] ReMe Bridge (reme service port ${REME_PORT})"
"${REME_DIR}/.venv/bin/python" "${SUITE_DIR}/bridge_reme.py" \
    --test-server-url "${TEST_URL}" \
    --appworld-mcp-url "${MCP_URL}" \
    --reme-dir "${REME_DIR}" \
    --data-root "${SUITE_DIR}/data" \
    --user-id "${PERSONA}" \
    --workspace-dir "${WORKSPACE_DIR}" \
    --reme-port "${REME_PORT}" \
    --model-name "${REME_MODEL_NAME}" \
    --model-base-url "${REME_LLM_BASE_URL}" \
    --model-api-key "${REME_LLM_API_KEY}" \
    > "${LOG_DIR}/bridge_${PERSONA}.log" 2>&1 &
BRIDGE_PID=$!
PIDS+=(${BRIDGE_PID})
sleep 5
if ! kill -0 "${BRIDGE_PID}" 2>/dev/null; then
    echo "Bridge failed to start:"; tail -30 "${LOG_DIR}/bridge_${PERSONA}.log"; exit 1
fi
for i in $(seq 1 12); do
    if grep -q "Bridge started:" "${LOG_DIR}/bridge_${PERSONA}.log" 2>/dev/null; then
        echo "  bridge initialized"; break
    fi
    sleep 5
done
grep -q "Bridge started:" "${LOG_DIR}/bridge_${PERSONA}.log" 2>/dev/null || {
    echo "WARNING: bridge may not be ready:"; tail -20 "${LOG_DIR}/bridge_${PERSONA}.log"; }

# ─── [5/5] Runner (run phase) ──────────────────────────────────────────
if [ "$RUN_PHASE_NEEDED" = true ]; then
    echo "[5/5] Runner: run phase (episode order from data/${PERSONA}/episode.yaml)"
    cd "${SUITE_DIR}"
    BENCH_TEST_SERVER_URL="${TEST_URL}" PYTHONPATH="${PIBENCH_DIR}" \
        "${PI_PYTHON}" -m src.main \
        --model-config "${MODEL_CONFIG}" \
        --history-config-path "${HISTORY_CONFIG}" \
        --mode run --user-id "${PERSONA}" \
        --workspace-dir "${NANOBOT_WORKSPACE_DIR}" \
        ${TASK_ARGS[@]+"${TASK_ARGS[@]}"} \
        2>&1 | tee "${LOG_DIR}/runner_run_${PERSONA}.log"
    RUN_EXIT=${PIPESTATUS[0]}
    if [ ${RUN_EXIT} -ne 0 ]; then
        echo "Run phase failed (exit ${RUN_EXIT}). Logs: ${LOG_DIR}/"
        exit ${RUN_EXIT}
    fi
else
    echo "[5/5] Runner: run phase skipped (all tasks completed)"
fi

if [ "$SKIP_EVAL" = true ]; then
    echo "Skipping eval (--skip-eval)"
    exit 0
fi

# ─── Trace conversion + eval phase (always over all available traces) ──
echo "Converting trace logs..."
"${PI_PYTHON}" "${SUITE_DIR}/fix_trace_logs.py" "${PERSONA}"

echo "Runner: eval phase"
cd "${SUITE_DIR}"
BENCH_TEST_SERVER_URL="${TEST_URL}" PYTHONPATH="${PIBENCH_DIR}" \
    "${PI_PYTHON}" -m src.main \
    --model-config "${MODEL_CONFIG}" \
    --history-config-path "${HISTORY_CONFIG}" \
    --mode eval --user-id "${PERSONA}" \
    --workspace-dir "${NANOBOT_WORKSPACE_DIR}" \
    2>&1 | tee "${LOG_DIR}/runner_eval_${PERSONA}.log"
EVAL_EXIT=${PIPESTATUS[0]}

echo ""
echo "========================================="
echo "persona=${PERSONA} finished (eval exit=${EVAL_EXIT})"
echo "  results : ${SUITE_DIR}/outputs/reme/${PERSONA}/"
echo "  memory  : ${WORKSPACE_DIR}/"
echo "  logs    : ${LOG_DIR}/"
echo "========================================="
exit ${EVAL_EXIT}
