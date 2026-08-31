[中文版 / Chinese version](./README_ZH.md)

# LongMemEval Benchmark

LongMemEval is a benchmark for **long-term memory over multi-session chat
histories**. Each item provides a chronologically ordered set of chat sessions
between a user and an assistant, followed by a probing question whose answer is
only recoverable by reasoning over the user-owned memory. ReMe ingests the
sessions into an isolated per-item workspace, answers the question via an
agentic (ReAct) mode, and scores the answer with an LLM-as-judge.

Question types include single-session (user / assistant / preference),
multi-session reasoning, knowledge update, and temporal reasoning.

Install ReMe and its runtime dependencies from the repository root; no `reme plugins install` is needed:

```bash
python -m pip install -e ".[as]"
```

Each worker directly loads the checkout's [plugin source](../../plugins/lme/README.md) and `lme.yaml`
application configuration without installing the plugin or mutating the global registry.
Custom application config paths still work through `reme.config`. Without installation,
use the plugin configuration file path for `extends`, rather than the installed `lme` alias.
This directory continues to own the runner, evaluation settings, dataset and outputs.
Model credentials use the environment variables in the plugin's application configuration.

## 1. Get the Dataset

ReMe uses only the **cleaned-S** split, hosted on HuggingFace:
[agentscope-ai/ReMe_longmemeval_clean_s_v2](https://huggingface.co/datasets/agentscope-ai/ReMe_longmemeval_clean_s_v2).
The download script fetches it via the hf-mirror.com mirror; to use a different
mirror, modify `BASE_URL` in [`download.py`](./download.py).

```bash
cd benchmark/longmemeval
python download.py            # saves dataset/longmemeval_s_reme_cleaned.json; skips if already present
```

Ground truth is embedded in the data file.

## 2. Run

From the repository root:

```bash
python benchmark/longmemeval/run.py
python benchmark/longmemeval/run.py --config benchmark/longmemeval/config.yaml
python benchmark/longmemeval/run.py -q                        # quiet: only eval-level logs
python benchmark/longmemeval/run.py --log-level WARNING       # reduce eval runner logs
python benchmark/longmemeval/run.py --reme-log-level WARNING  # reduce reme internal logs
python benchmark/longmemeval/run.py --eval_only               # reuse existing workspaces, query + judge only
```

## 3. Pipeline

1. Load the dataset (ground truth is embedded in the data file).
2. For each item, create an isolated workspace and ingest sessions in chronological order.
3. If a custom application configuration enables `auto_dream`, trigger it when sessions cross the configured hour
   (default 23:00). The packaged preset leaves it disabled.
4. Answer each question via agentic (ReAct) mode.
5. Judge the answer (binary yes/no) with the `answer_judge` job and print per-type accuracy.

## 4. Key config — `benchmark/longmemeval/config.yaml`

| Key | Meaning |
| --- | --- |
| `dataset.path` | Dataset file to evaluate (e.g. `longmemeval_s_reme_cleaned.json`); ground truth is included. |
| `dataset.start_index` / `num_items` | Slice of items to evaluate. |
| `dataset.question_types` | Filter by question type; empty = all. |
| `dataset.workspace_root` | Per-item workspace root (`benchmark/longmemeval/workspaces/longmemeval-s`). |
| `evaluation.num_workers` | `0` = auto (cpu-2), `1` = sequential, `>1` = parallel. |
| `evaluation.filter_future_sessions` | Only ingest sessions with timestamp ≤ `question_date`. |
| `reme.config` | ReMe config used (`lme.yaml`). |
| `reme.dream_trigger_hour` / `dream_scan_days` / `dream_max_units` | Dream triggering behavior. |
| `output.dir` | Results directory (`benchmark/longmemeval/results`). |

## 5. Outputs

Results are JSON files written to `output.dir` as `results_<timestamp>.json`,
with a per-type accuracy summary also printed to the console. Logging
conventions are shared across benchmarks — see the
[top-level README](../README.md#outputs--logs).

## 6. Reference Results

### cleaned-s

**Basic settings**

1. Modified auto-memory prompt, auto-dream disabled.
2. All sessions in reme-memory are strictly earlier than the question time.

**Results**

agentscope==2.0.4.post1, conda reme env, 32 workers, eval-only (reusing prebuilt memory)
(2026-08-06, 500 items, total 10.0 min)

| Type | Agentic | input tok/q | output tok/q | total tok/q | tool calls/q |
|---|---|---|---|---|---|
| knowledge-update | 0.910 | 31,581 | 589 | 32,169 | 2.90 |
| multi-session | 0.842 | 52,837 | 1,474 | 54,311 | 4.21 |
| single-session-assistant | 1.000 | 15,596 | 279 | 15,875 | 1.89 |
| single-session-preference | 0.633 | 36,802 | 818 | 37,620 | 3.60 |
| single-session-user | 0.986 | 27,433 | 359 | 27,792 | 2.60 |
| temporal-reasoning | 0.902 | 62,674 | 985 | 63,659 | 4.97 |
| **OVERALL** | **0.894** | **43,448** | **876** | **44,324** | **3.69** |