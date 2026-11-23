[中文版 / Chinese version](./README_ZH.md)

# LongMemEval Benchmark

LongMemEval is a benchmark for **long-term memory over multi-session chat histories**. Each item provides
chronologically ordered user–assistant sessions and a probing question whose answer is only recoverable by
reasoning over the user-owned memory.

## 1. Data preparation

Question types include single-session (user / assistant / preference),
multi-session reasoning, knowledge update, and temporal reasoning.

ReMe uses only the **cleaned-S** split, hosted on HuggingFace:
[agentscope-ai/ReMe_longmemeval_clean_s_v2](https://huggingface.co/datasets/agentscope-ai/ReMe_longmemeval_clean_s_v2).
The download script fetches it via the hf-mirror.com mirror; to use a different
mirror, modify `BASE_URL` in [`download.py`](./download.py).

```bash
cd benchmark/longmemeval
python download.py            # saves dataset/longmemeval_s_reme_cleaned.json; skips if already present
```

Ground truth is embedded in the data file.

## 2. Use the LME plugin

### 2.1 Overview

The LME plugin loads the dataset with embedded ground truth, creates an isolated workspace per item, ingests
sessions chronologically, and answers probing questions via agentic (ReAct) mode. Its LLM-as-judge
`answer_judge` job gives binary (yes/no) judgments and reports per-type accuracy. If a custom application
configuration enables `auto_dream`, it triggers when sessions cross the configured hour (default 23:00); the
packaged preset leaves it disabled.

### 2.2 How to use

#### Installation and environment

Install ReMe and the LongMemEval plugin in editable mode from the repository root:

```bash
python -m pip install -e ".[as]"
reme plugins install ./plugins/lme --editable
reme plugins install ./plugins/lme-judge --editable
reme plugins validate lme
```

The runner explicitly enables the installed `lme` plugin and combines its defaults with
ReMe's built-in `benchmark` preset. Editable installation keeps changes under
[`plugins/lme`](../../plugins/lme/README.md) visible without reinstalling the plugin.
Custom application config paths still work through `reme.config` and can use `extends: benchmark`.
Model credentials use the environment variables declared by the shared benchmark configuration.

#### Key configuration — `benchmark/longmemeval/config.yaml`

| Key | Meaning |
| --- | --- |
| `dataset.path` | Dataset file to evaluate (e.g. `longmemeval_s_reme_cleaned.json`); ground truth is included. |
| `dataset.start_index` / `num_items` | Slice of items to evaluate. |
| `dataset.question_types` | Filter by question type; empty = all. |
| `dataset.workspace_root` | Per-item workspace root (`benchmark/longmemeval/workspaces/longmemeval-s`). |
| `evaluation.num_workers` | `0` = auto (cpu-2), `1` = sequential, `>1` = parallel. |
| `evaluation.filter_future_sessions` | Only ingest sessions with timestamp ≤ `question_date`. |
| `reme.config` | ReMe config used (`benchmark`). |
| `reme.dream_trigger_hour` / `dream_scan_days` / `dream_max_units` | Dream triggering behavior. |
| `output.dir` | Results directory (`benchmark/longmemeval/results`). |

#### Run the benchmark

From the repository root:

```bash
python benchmark/longmemeval/run.py
python benchmark/longmemeval/run.py --config benchmark/longmemeval/config.yaml
python benchmark/longmemeval/run.py -q                        # quiet: only eval-level logs
python benchmark/longmemeval/run.py --log-level WARNING       # reduce eval runner logs
python benchmark/longmemeval/run.py --reme-log-level WARNING  # reduce reme internal logs
python benchmark/longmemeval/run.py --eval_only               # reuse existing workspaces, query + judge only
```

### 2.3 Results

Results are JSON files written to `output.dir` as `results_<timestamp>.json`,
with a per-type accuracy summary also printed to the console. Logging
conventions are shared across benchmarks — see the
[top-level README](../README.md#outputs--logs).

**Reference results**

#### cleaned-s

**Results**

agentscope==2.0.4.post1, conda reme env, 32 workers, eval-only (reusing prebuilt memory)
(2026-08-06, 500 items, total 10.0 min)

| Type | Agentic | input tok/q | output tok/q | total tok/q | tool calls/q |
|---|---|---|---|---|---|
| **Overall** | **0.894** | **43.4K** | **0.9K** | **44.3K** | **3.69** |
| knowledge-update | 0.910 | 31.6K | 0.6K | 32.2K | 2.90 |
| multi-session | 0.842 | 52.8K | 1.5K | 54.3K | 4.21 |
| single-session-assistant | 1.000 | 15.6K | 0.3K | 15.9K | 1.89 |
| single-session-preference | 0.633 | 36.8K | 0.8K | 37.6K | 3.60 |
| single-session-user | 0.986 | 27.4K | 0.4K | 27.8K | 2.60 |
| temporal-reasoning | 0.902 | 62.7K | 1.0K | 63.7K | 4.97 |

## 3. Optimize the plugin with HoMe

### 3.1 Overview

HoMe prepares an experiment workspace from the dataset and a Git revision, evaluates the baseline, and
iteratively improves the LME plugin code. The optimizer returns a candidate `commit_sha`, which can then be
evaluated independently on the configured evaluation set.

### 3.2 How to use

#### Experiment setup

Get the dataset above and follow the [HoMe setup and run
guide](../../plugins/home/README.md#4-run-a-benchmark-experiment).
[config_home.yaml](config_home.yaml) defaults to `benchmark/longmemeval/workspaces/home-groups-1-5`
with 250 cases from **groups 1–5** for optimization.
The training IDs and [reviewed evidence annotations](home_evidence_sessions.jsonl) come from
`dev_meta` commit `b53048e44833937df80a52fccceea78d054790e8`. The original split used seed 42
and stratified the dataset by question type.

[config_home_eval.yaml](config_home_eval.yaml) defaults to `benchmark/longmemeval/workspaces/home-test-held-out`
and excludes the source experiment's training IDs. With the default training split, this selects 250 cases
from **groups 6–10**. To choose cases explicitly, set `exclude_training_cases: false` and replace
`all_cases: true` with `case_ids: [...]`.

Prepare the optimization workspace from the repository root:

```bash
python benchmark/longmemeval/run_home.py
```

#### Run the experiment

Run local one-shot optimization:

```bash
python benchmark/longmemeval/run_home.py --optimize
```

Set `CANDIDATE_SHA` to the optimizer's `commit_sha` or a unique hexadecimal prefix, then run local one-shot evaluation:

```bash
python benchmark/longmemeval/run_home_eval.py --commit "$CANDIDATE_SHA"
```

See the [HoMe evaluation guide](../../plugins/home/README.md#evaluate-through-the-service) for service-based
evaluation.

### 3.3 Results

HoMe is optimized on 250 LongMemEval-S training cases. ReMe and HoMe are evaluated on the same 250 held-out
cases.
Token counts and LLM calls are per-query averages for the answering stage.

#### ReMe baseline

| Question category | Score | Total tokens | Input tokens | Output tokens | LLM calls |
|---|---:|---:|---:|---:|---:|
| **Overall** | **0.888** | **43.9K** | **43.2K** | **0.8K** | **4.7** |
| Knowledge Update | 0.921 | 29.0K | 28.5K | 0.6K | 3.8 |
| Multi-session | 0.836 | 54.5K | 53.2K | 1.3K | 5.2 |
| Single-session Assistant | 1.000 | 15.8K | 15.5K | 0.3K | 2.9 |
| Single-session Preference | 0.667 | 31.4K | 30.6K | 0.7K | 4.1 |
| Single-session User | 0.971 | 26.6K | 26.3K | 0.3K | 3.5 |
| Temporal Reasoning | 0.882 | 65.7K | 64.8K | 0.9K | 6.2 |

#### HoMe

| Question category | Score | Total tokens | Input tokens | Output tokens | LLM calls |
|---|---:|---:|---:|---:|---:|
| **Overall** | **0.916** | **17.9K** | **17.2K** | **0.6K** | **3.0** |
| Knowledge Update | 0.974 | 9.7K | 9.3K | 0.5K | 2.2 |
| Multi-session | 0.806 | 19.3K | 18.3K | 1.0K | 3.3 |
| Single-session Assistant | 1.000 | 9.8K | 9.5K | 0.3K | 2.2 |
| Single-session Preference | 0.933 | 11.8K | 11.0K | 0.9K | 2.7 |
| Single-session User | 0.971 | 11.2K | 11.0K | 0.2K | 2.3 |
| Temporal Reasoning | 0.940 | 29.2K | 28.5K | 0.7K | 4.0 |
