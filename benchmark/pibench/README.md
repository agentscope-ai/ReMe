[中文版 / Chinese version](./README_ZH.md)

# π-Bench Evaluation Suite

A glue layer that connects the **ReMe agent (with persistent memory)** to
**π-Bench** (Proactive Personal Assistant Benchmark). This directory contains
only the minimal code and configuration needed for the integration: the
π-Bench framework (`src/`), evaluation data (`data/`), the AppWorld tool
environment, and ReMe itself are all **external third-party dependencies**,
referenced in place via symlink and environment variables and never bundled
with this suite.

- π-Bench: https://github.com/Simplified-Reasoning/Pi-Bench (arXiv: 2605.14678)
- ReMe: the root of the ReMe repository this suite lives in (recommended
  location: `ReMe/benchmark/pibench/`)

## 1. Architecture

```
π-Bench runner (src.main --mode run)
  │  user_agent (simulated-user LLM) walks data/{persona}/episode.yaml
  │  task by task, chatting with the agent over multiple turns and judging
  │  hidden intents (PROC) during the run phase
  ▼
test server (π-Bench scripts/test_server.py, HTTP long-polling)
  ▲ /send            │ /poll
  │                  ▼
bridge_reme.py ──────────────► ReMe Application (embedded as a library)
  │                             ├─ agent_wrapper: agent under test (AgentScope)
  │                             ├─ jobs: search / auto_memory / daily_write
  │                             └─ workspace: reme_workspace/{persona}/
  │                                (isolated persistent memory per persona)
  └──── MCP ────► AppWorld MCP ────► AppWorld APIs (tool/app environment)

π-Bench runner (src.main --mode eval)
  judger (judge LLM) reads the traces and scores each checklist item (COMP)
```

Key points:
- The bridge runs on **ReMe's own venv python** and uses ReMe as a library
  (`resolve_app_config` + `Application`); **no ReMe source modification** is
  required.
- Every incoming user message automatically triggers a ReMe memory `search`
  and injects the matched memories (tuning knobs in §8); on task end (reset)
  the session is distilled into daily notes by `auto_memory`.
- Tool calls executed by the agent (AppWorld MCP + ReMe job tools) are
  captured per turn into the trace as `tool_steps`, so π-Bench
  `tools_evaluation_path` scripts can score tool behavior (§7).
- π-Bench's `data/`, `src/` and AppWorld are not part of this suite; install
  π-Bench first (§3.1).

## 2. Directory layout

```
pibench/
├── README.md / README_ZH.md  # this document (English / Chinese)
├── env.sh.example            # environment template (copy to env.sh, fill TODOs)
├── bridge_reme.py            # ReMe ↔ test server bridge (memory inject/save,
│                             #   profile injection, tool-trace capture)
├── run_persona.sh            # full pipeline for ONE persona (5 services + run + eval)
├── run_all.sh                # batch over 5 personas (fresh/resume, default parallel=2)
├── resume.py                 # checkpoint resume: completion detection + surgical
│                             #   cleanup of interrupted tasks' residual memory
├── fix_trace_logs.py         # run outputs → ~/.nanobot/trace_logs conversion,
│                             #   merging tool sidecars into turn files (pre-eval)
├── .gitignore                # excludes env.sh and all runtime artifacts
└── config/
    ├── models/reme.yaml      # runner model config (model_id=reme)
    └── bench/evaluation/trace_history.yaml   # trace render policy (shipped with
                                              #   the suite; passed via --history-config-path)
```

Generated at runtime (all git-ignored): `data` (symlink), `logs/`, `outputs/`,
`reme_workspace/`, `nanobot_workspace/`.

## 3. Prerequisites (third-party, install first)

### 3.1 π-Bench repository (with AppWorld)

```bash
git clone https://github.com/Simplified-Reasoning/Pi-Bench.git <pi-bench-dir>
cd <pi-bench-dir>
python3.11 -m venv .venv            # scripts expect exactly this venv name
source .venv/bin/activate
pip install -e .                    # pibench runner (src.main)
bash scripts/setup_appworld.sh      # install AppWorld and download its data (large)
```

Post-install sanity checks:
```bash
ls data/                    # should contain researcher marketer pharmacist law_trainee Financier
.venv/bin/python -c "import src" && echo OK
.venv/bin/appworld --help >/dev/null && echo OK
```

### 3.2 ReMe repository

```bash
cd <reme-dir>               # ReMe repository root (contains the reme/ package)
python3.11 -m venv .venv    # scripts expect exactly this venv name
source .venv/bin/activate
pip install -e .            # or ReMe's own install flow; `import reme` must work
```

Sanity check: `.venv/bin/python -c "import reme; print('ok')"`

## 4. Install this suite (step by step)

1. **Place the suite** (recommended inside the ReMe repo so `REME_DIR` is
   inferred automatically):
   ```bash
   cp -r pibench <reme-dir>/benchmark/pibench
   cd <reme-dir>/benchmark/pibench
   ```
   If placed elsewhere, set `REME_DIR` explicitly in env.sh later.

2. **Create the environment file and fill in the custom parameters**:
   ```bash
   cp env.sh.example env.sh
   ```
   Open `env.sh`; required items (marked TODO):
   | Variable | Description |
   |---|---|
   | `PI_BENCH_ROOT` | π-Bench repo root (contains `src/` `data/` `.venv` `third_party/appworld`) |
   | `USER_API_KEY` | API key of the simulated-user LLM (run phase, hidden-intent judging) |
   | `JUDGER_API_KEY` | API key of the judger LLM (eval phase, checklist scoring) |
   | `BRAVE_SEARCH_API_KEY` | optional; for the agent's web_search tool, `dummy` when unused |

   Optional tuning: `REME_MODEL_NAME` (base model of the agent under test),
   `REME_DIR`, `REME_LLM_BASE_URL` (default: DashScope OpenAI-compatible
   endpoint).

3. **Link the evaluation data** (referenced in place, never copied):
   ```bash
   ln -s "$PI_BENCH_ROOT/data" data
   ```

4. **(Optional) adjust model config** `config/models/reme.yaml`:
   - `user_agent.model` / `judger.model`: model names for the simulated user
     and the judger (literal values; π-Bench only expands `${ENV}` in
     base_url/api_key).
   - `run.turn_timeout`, `max_tool_iterations`, etc. as needed.

5. **Smoke check** (does not start the evaluation):
   ```bash
   bash -n run_all.sh && bash -n run_persona.sh
   source env.sh && "$REME_DIR/.venv/bin/python" -c "import reme; print('reme ok')"
   ```

## 5. Run the evaluation

> ⚠️ For long runs use `screen`, **not nohup** (nohup loses the permission
> context in sandboxed/restricted environments and breaks child processes).

```bash
# Full official run: wipe ALL personas' memory/outputs/traces first (default
# fresh mode, parallel=2)
mkdir -p logs   # on a fresh deployment logs/ does not exist yet
screen -dmS pibench_suite bash -c "cd $(pwd) && bash run_all.sh > logs/run_all_master.log 2>&1"

# Checkpoint continuation (after an interruption; no wipe, completed tasks skipped)
bash run_all.sh --resume

# Other usages
bash run_all.sh --parallel 1          # sequential
bash run_all.sh --resume --skip-eval  # run phase only
bash run_persona.sh researcher        # single persona (default --resume semantics)
bash run_persona.sh researcher --fresh
```

Time reference: 5 personas × 20 tasks, parallel=2, fresh full run ≈ 12–14 hours.

`run_all.sh` exits non-zero when any persona fails, so upstream automation
cannot mistake a partially failed suite run for a success.

## 6. Port allocation (parallel personas never collide)

| persona     | AppWorld API | AppWorld MCP | Test Server | ReMe internal service |
|-------------|------|-------|------|-------|
| marketer    | 9001 | 10001 | 9998 | 18766 |
| law_trainee | 9002 | 10002 | 9997 | 18767 |
| pharmacist  | 9003 | 10003 | 9996 | 18768 |
| researcher  | 9004 | 10004 | 9995 | 18765 |
| Financier   | 9005 | 10005 | 9994 | 18769 |

## 7. Outputs and scores

- **Results**: `outputs/reme/{persona}/{task}/eval/results/*_result.json`
  - `overall_average_score`: checklist completeness (COMP; the judger scores
    each criterion YES/NO, weighted across dependency groups)
  - `overall_proactiveness_average_score`: proactiveness (PROC; the
    user_agent judges hidden-intent coverage during the run phase; each task
    file also carries the global average)
- **Traces**: `~/.nanobot/trace_logs/reme/{persona}/{task}/...` (the scoring
  input of the eval phase)
- **Logs**: `logs/` (`suite_<persona>.log` per persona; `bridge_*`,
  `runner_run/eval_*`, `appworld_*`, `test_server_*` per service)
- **Memory store**: `reme_workspace/{persona}/` (daily/digest notes, raw
  session dialogs, BM25 index, etc.; persistent across runs, wiped only in
  fresh mode)

Score summary:
```bash
grep -h "overall_average_score\|overall_proactiveness" \
  outputs/reme/*/*/eval/results/*_result.json | head
```

### Tool-trace capture (tools_evaluation support)

Some tasks define `objectives.tools_evaluation_path`: Python scripts that
score tool behavior (e.g. "the temporary Todoist board was created and
removed"). They need the executed tool calls in the trace. The pipeline:

1. During `reply()`, the bridge reads the persisted AgentScope session state
   after each turn and extracts the new `tool_call` / `tool_result` blocks
   (tool name, arguments, result).
2. Records are appended to
   `outputs/reme/{persona}/{task}/history/{ts}-tools.jsonl`, tagged with the
   turn number; AgentScope MCP names (`mcp__AppWorld__<tool>`) are normalized
   to the π-Bench convention (`mcp_appworld_<tool>`).
3. `fix_trace_logs.py` pairs each `{ts}-messages.jsonl` run with the
   temporally closest tools sidecar and merges the records into the generated
   `turn_N.json` files under the `tool_steps` key — one of the two
   tool-history formats understood by π-Bench's `collect_tool_history()`.
4. The eval phase then feeds `tool_steps` to both the tools_evaluation
   scripts and the rendered `<tool_trace_extracts>` seen by the judger.

## 8. Memory mechanism (core design of this suite)

- **Persona isolation**: each persona has its own workspace
  (`reme_workspace/{persona}/`); the bridge takes an exclusive
  `.bridge.lock` on it at startup, so two bridges can never share one memory
  store, and one persona's memory search can never reach another's memories.
- **Writes**: on task end (runner sends reset), the session is distilled by
  the `auto_memory` job into daily notes and indexed by the background
  watcher (BM25). Saves are non-blocking background tasks; the first message
  of a new session waits for in-flight writes before searching.
- **Reads**: on every incoming user message the bridge runs one `search` and
  injects matched memories (`[Relevant memories from previous sessions]`
  prefix); without matches the message passes through unchanged. Retrieval
  tuning (bridge CLI flags, adjustable in run_persona.sh):
  - `--search-limit 3`: at most 3 memory chunks injected per message;
  - `--search-min-score 2.0`: weak BM25 hits are filtered out;
  - `tool_context_id` rotates per task: chunks already injected within the
    same task are not re-injected (ReMe's seen-chunk dedup, 24h TTL); normal
    recall resumes after task boundaries.
- **No self-leakage**: the in-progress session is not in the store yet
  (saves happen on reset), so a task can never retrieve its own unfinished
  content.
- The agent also holds `search`/`daily_write` tools and can retrieve/record
  proactively.
- **System prompt**: `bridge_reme.py:build_system_prompt()` embeds the
  HIDDEN-NEEDS protocol (proactiveness-oriented) and injects the persona
  profile from `data/{persona}/profile.yaml` into every turn's system prompt.

## 9. Checkpoint resume and memory-cleanup semantics

- **Completion detection** (resume.py): scans
  `outputs/reme/{persona}/**/history/*-log.jsonl` and
  `outputs/reme/{persona}/run/*-log.jsonl` for
  `Task finished task_id=X status=Y`. The status with the **newest event
  timestamp** wins per task (record `timestamp`, falling back to
  `timestamp_iso`, then to the timestamp embedded in the log file name) —
  file category and read order alone can never override a newer record, so an
  old run-level SUCCESS cannot mask a newer per-task ERROR. `SUCCESS /
  MAX_TURNS / TIMEOUT` count as completed; `ERROR` and never-started tasks
  are re-run (passed to the runner as repeated `--task-id` flags in episode
  order).
- **Answer-leak prevention**: an interrupted task may already have been
  distilled into daily notes during graceful shutdown; re-running it with
  that memory injected would inflate scores. Before resuming,
  `resume.py cleanup` therefore removes residual memory **only for tasks
  about to be re-run** (daily/digest notes, session/dialog, mem_session;
  matched via `session_id = pibench_{task}_*`). Completed tasks' memories are
  never touched. Daily index files are refreshed **only for the dates that
  lost notes**, by full workspace-relative wikilink path — and when the ReMe
  package is importable, the refresh reuses ReMe's own daily-index rebuild
  logic (`refresh_day_index`), so same-named notes on other dates are never
  modified.
- **fresh vs resume are mutually exclusive**: a full memory wipe belongs to
  fresh mode only (`run_all.sh` default, executed before any service starts);
  resume never wipes.

## 10. Customization entry points

| Goal | Location |
|---|---|
| Base model of the agent under test | `REME_MODEL_NAME` in `env.sh` |
| user_agent / judger models | `config/models/reme.yaml` |
| Agent system prompt | `bridge_reme.py` `build_system_prompt()` |
| Memory retrieval limit/threshold | `--search-limit/--search-min-score` on the bridge command in `run_persona.sh` |
| ReMe internal parameters | **Do not modify ReMe source**; write a dedicated config modeled on `plugins/beam/src/reme_beam/configs/beam.yaml` and override via `resolve_app_config(config=...)` (see bridge `_init_reme_app`) |
| Turn timeout / tool iteration cap | `config/models/reme.yaml` `run.turn_timeout`, `model.max_tool_iterations` |

## 11. Troubleshooting

- **Port already in use**: the scripts auto-kill residual processes on the
  four port groups above; if another suite (e.g. a different π-Bench
  experiment) holds them, stop it first or change the port table in
  run_persona.sh.
- **Bridge exits immediately with workspace locked**: another bridge already
  holds the same workspace; make sure each persona uses its own
  `--workspace-dir` (the scripts allocate one per persona).
- **Runner reports `${USER_API_KEY} ... empty`**: env.sh is unfilled or not
  sourced; run_persona.sh sources env.sh automatically — when running the
  runner manually, `source env.sh` first.
- **`Cannot import 'reme'`**: the bridge must run with
  `${REME_DIR}/.venv/bin/python` (run_persona.sh already does); otherwise
  check that `REME_DIR` points at the ReMe repository root.
- **AppWorld fails to start**: run `bash scripts/setup_appworld.sh` in the
  π-Bench repo first (downloads data); inspect
  `logs/appworld_*_<persona>.log`.
- **trace_history.yaml not found**: the runner needs
  `config/bench/evaluation/trace_history.yaml`; this suite ships the file and
  passes it explicitly via `--history-config-path`, and run_persona.sh fails
  fast with a clear error if it is missing. Always launch run_persona.sh /
  run_all.sh from the suite directory.

## 12. Privacy and security

- The suite code and config templates contain **no real API keys, user names
  or absolute paths**; real keys live only in your local `env.sh`
  (git-ignored).
- `logs/`, `outputs/`, `reme_workspace/` and `nanobot_workspace/` contain
  full conversations and model outputs; never commit or share them.
- The `data` symlink points at the official π-Bench evaluation data; respect
  its data license terms.
