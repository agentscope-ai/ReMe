[中文版 / Chinese version](./README_ZH.md)

# ReMe Benchmarks

Reproduction guide for the two memory benchmarks shipped with ReMe:

- **LongMemEval** — long-term memory over multi-session chat histories.
- **BEAM** — memory capability over long-context chat cases with rubric-based judging.

Each benchmark runs its own end-to-end pipeline: ingest sessions into an isolated
per-item workspace, answer probing questions via an agentic (ReAct) mode,
then score answers with an LLM-as-judge.

## 1. Prerequisites

Install ReMe with dev + core extras (Python 3.11+):

```bash
pip install -e ".[dev,core]"
```

Configure model credentials in a project-root `.env` file (copied from `example.env`).
The runners auto-load `.env` from the repository root. Required variables typically include:

```bash
LLM_API_KEY=...
LLM_BASE_URL=...
EMBEDDING_API_KEY=...
EMBEDDING_BASE_URL=...
```

Model names and component wiring live in the ReMe configs referenced by each benchmark
(`reme/config/lme.yaml` and `reme/config/beam.yaml`).

## 2. Download Datasets

See [`datasets/README_EN.md`](datasets/README_EN.md) for full details.

**LongMemEval** (downloaded from a HuggingFace mirror):

```bash
cd benchmark/datasets/longmemeval
python download.py            # downloads the cleaned-S dataset; skips if already present
```

**BEAM** (public repository, cloned into `benchmark/datasets/`):

```bash
cd benchmark/datasets
git clone https://github.com/mohammadtavakoli78/BEAM.git
```

## 3. Run LongMemEval

From the repository root:

```bash
python benchmark/longmemeval/run.py
python benchmark/longmemeval/run.py --config benchmark/longmemeval/config.yaml
python benchmark/longmemeval/run.py -q                        # quiet: only eval-level logs
python benchmark/longmemeval/run.py --log-level WARNING       # reduce eval runner logs
python benchmark/longmemeval/run.py --reme-log-level WARNING  # reduce reme internal logs
python benchmark/longmemeval/run.py --eval_only               # reuse existing workspaces, query + judge only
```

### Pipeline

1. Load the dataset (ground truth is embedded in the data file).
2. For each item, create an isolated workspace and ingest sessions in chronological order.
3. Trigger `auto_dream` when consecutive sessions cross the configured hour (default 23:00).
4. Answer each question via agentic (ReAct) mode.
5. Judge the answer (binary yes/no) with the `answer_judge` job and print per-type accuracy.

### Key config — `benchmark/longmemeval/config.yaml`

| Key | Meaning |
| --- | --- |
| `dataset.path` | Dataset file to evaluate (e.g. `longmemeval_s_reme_cleaned.json`); ground truth is included. |
| `dataset.start_index` / `num_items` | Slice of items to evaluate. |
| `dataset.question_types` | Filter by question type; empty = all. |
| `dataset.workspace_root` | Per-item workspace root (`benchmark/memory_workspaces/longmemeval-s`). |
| `evaluation.num_workers` | `0` = auto (cpu-2), `1` = sequential, `>1` = parallel. |
| `evaluation.filter_future_sessions` | Only ingest sessions with timestamp ≤ `question_date`. |
| `reme.config` | ReMe config used (`lme.yaml`). |
| `reme.dream_trigger_hour` / `dream_scan_days` / `dream_max_units` | Dream triggering behavior. |
| `output.dir` | Results directory (`benchmark/results/longmemeval`). |

## 4. Run BEAM

From the repository root:

```bash
python benchmark/beam/run.py
python benchmark/beam/run.py --config benchmark/beam/config.yaml
python benchmark/beam/run.py -q                        # quiet
python benchmark/beam/run.py --eval_only               # reuse existing workspaces, query + judge only
```

### Pipeline

1. For each case, load `chat.json` and convert each batch into a ReMe session.
2. Ingest sessions in chronological order into an isolated workspace, then `digest_update`.
3. Answer each probing question via agentic (ReAct) mode.
4. Score answers with BEAM's rubric-based `answer_judge` job and print per-type averages.

### Key config — `benchmark/beam/config.yaml`

| Key | Meaning |
| --- | --- |
| `dataset.beam_root` | BEAM dataset root (`benchmark/datasets/BEAM`). |
| `dataset.chat_size` | Variant to run: `100K` / `500K` / `1M` / `10M`. |
| `dataset.case_ids` | Specific cases (e.g. `["1","2"]`); empty = all cases. |
| `dataset.start_index` / `num_items` | Case pagination (`num_items` `0` = all). |
| `dataset.workspace_root` | Per-case workspace root (`benchmark/memory_workspaces/beam`). |
| `evaluation.num_workers` | `0` = auto, `1` = sequential, `>1` = parallel. |
| `reme.config` | ReMe config used (`beam.yaml`). |
| `output.dir` | Results directory (`benchmark/results/beam`). |

## 5. Outputs & Logs

- **Results**: JSON files written to `output.dir`
  (`results_<timestamp>.json` for LongMemEval,
  `results_<chat_size>_<timestamp>.json` for BEAM). A summary with per-type
  accuracy/score is also printed to the console.
- **Logs**: when `output.log_to_file` is enabled, per-run logs are written to
  `logs/<log_prefix>_<timestamp>/` (a `runner.log` plus one `worker-<pid>.log`
  per worker process).

## 6. Stopping a Run

Parallel runs spawn a process tree. To terminate a run and all its workers cleanly:

```bash
bash benchmark/kill.sh <PID>
```

The script gracefully sends `SIGTERM` to the whole process tree, then escalates to
`SIGKILL` for any process that does not exit within 5 seconds.

## 7. Reference Results

Recorded evaluation results are available in:

- [`result-longmemeval.md`](./result-longmemeval.md)
- [`result-beam.md`](./result-beam.md)
