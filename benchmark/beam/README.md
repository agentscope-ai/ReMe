[中文版 / Chinese version](./README_ZH.md)

# BEAM Benchmark

BEAM is a benchmark for **memory capability over long-context chat cases**. Each
case contains a very long chat history split into batches; ReMe converts each
batch into a session, ingests them in chronological order, then answers probing
questions via an agentic (ReAct) mode. Answers are scored with BEAM's
rubric-based `answer_judge` job, which produces both a graded score and a binary
verdict, and per-type averages are reported.

BEAM ships dataset variants by chat size — `100K` / `500K` / `1M` / `10M` — so
memory systems can be stressed at different context lengths. Question types
include abstention, contradiction resolution, event ordering, information
extraction, instruction following, knowledge update, multi-session reasoning,
preference following, summarization, and temporal reasoning.

Install ReMe and its runtime dependencies from the repository root; no `reme plugins install` is needed:

```bash
python -m pip install -e ".[as]"
```

Each worker directly loads the checkout's [plugin source](../../plugins/beam/README.md) and `beam.yaml`
application configuration without installing the plugin or mutating the global registry.
Custom application config paths still work through `reme.config`. Without installation,
use the plugin configuration file path for `extends`, rather than the installed `beam` alias.
This directory continues to own the runner, evaluation settings, dataset and outputs.
Model credentials use the environment variables in the plugin's application configuration.

## 1. Get the Dataset

BEAM is a public repository, cloned into `benchmark/beam/dataset/`:

```bash
mkdir -p benchmark/beam/dataset
cd benchmark/beam/dataset
git clone https://github.com/mohammadtavakoli78/BEAM.git
```

After cloning, `benchmark/beam/dataset/BEAM/` should contain `chats/`, `src/`,
`topics/` and other subdirectories.

## 2. Run

From the repository root:

```bash
python benchmark/beam/run.py
python benchmark/beam/run.py --config benchmark/beam/config.yaml
python benchmark/beam/run.py -q                        # quiet
python benchmark/beam/run.py --eval_only               # reuse existing workspaces, query + judge only
```

## 3. Pipeline

1. For each case, load `chat.json` and convert each batch into a ReMe session.
2. Ingest sessions in chronological order into an isolated workspace, then `digest_update`.
3. Answer each probing question via agentic (ReAct) mode.
4. Score answers with BEAM's rubric-based `answer_judge` job and print per-type averages.

## 4. Key config — `benchmark/beam/config.yaml`

| Key | Meaning |
| --- | --- |
| `dataset.beam_root` | BEAM dataset root (`benchmark/beam/dataset/BEAM`). |
| `dataset.chat_size` | Variant to run: `100K` / `500K` / `1M` / `10M`. |
| `dataset.case_ids` | Specific cases (e.g. `["1","2"]`); empty = all cases. |
| `dataset.start_index` / `num_items` | Case pagination (`num_items` `0` = all). |
| `dataset.workspace_root` | Per-case workspace root (`benchmark/beam/workspaces/beam`). |
| `evaluation.num_workers` | `0` = auto, `1` = sequential, `>1` = parallel. |
| `reme.config` | ReMe config used (`beam.yaml`). |
| `output.dir` | Results directory (`benchmark/beam/results`). |

## 5. Outputs

Results are JSON files written to `output.dir` as
`results_<chat_size>_<timestamp>.json`, with a per-type score summary also
printed to the console. Logging conventions are shared across benchmarks — see
the [top-level README](../README.md#outputs--logs).

## 6. Reference Results

> The results below use the longmemeval-version prompt.

### 100K

agentscope==2.0.4.post1, conda reme env, 20 workers, eval-only (reusing prebuilt memory)
(2026-08-05, 20 cases / 400 Qs, total 46.0 min)

| Type | Agentic | Binary | input tok/q | output tok/q | total tok/q | tool calls/q |
|---|---|---|---|---|---|---|
| abstention | 0.550 | 0.550 | 96,031 | 1,070 | 97,101 | 4.58 |
| contradiction_resolution | 0.438 | 0.412 | 32,263 | 872 | 33,135 | 2.48 |
| event_ordering | 0.501 | 0.423 | 140,195 | 5,163 | 145,358 | 4.70 |
| information_extraction | 0.873 | 0.832 | 50,245 | 883 | 51,128 | 3.15 |
| instruction_following | 0.750 | 0.725 | 37,986 | 848 | 38,834 | 2.67 |
| knowledge_update | 0.688 | 0.675 | 31,198 | 651 | 31,849 | 2.27 |
| multi_session_reasoning | 0.626 | 0.584 | 85,038 | 4,563 | 89,601 | 4.28 |
| preference_following | 0.925 | 0.912 | 34,281 | 989 | 35,270 | 2.50 |
| summarization | 0.623 | 0.461 | 89,657 | 2,056 | 91,713 | 4.12 |
| temporal_reasoning | 0.637 | 0.625 | 34,563 | 1,049 | 35,612 | 2.52 |
| **OVERALL** | **0.661** | **0.620** | **63,146** | **1,814** | **64,960** | **3.33** |

Memory Construction average token consumption (default agent, full build over 20 cases):

| Agent | input tok/case | output tok/case | total tok/case |
|---|---|---|---|
| default | 2,172,316 | 136,697 | 2,309,013 |

### 1M

agentscope==2.0.4.post1, conda reme env, 20 workers, full memory build
(2026-08-05, 35 cases / 700 Qs, total 459.2 min)

| Type | Agentic | Binary | input tok/q | output tok/q | total tok/q | tool calls/q |
|---|---|---|---|---|---|---|
| abstention | 0.429 | 0.429 | 118,707 | 1,178 | 119,886 | 4.20 |
| contradiction_resolution | 0.391 | 0.364 | 49,787 | 810 | 50,597 | 2.50 |
| event_ordering | 0.558 | 0.456 | 201,514 | 3,889 | 205,403 | 4.79 |
| information_extraction | 0.809 | 0.772 | 78,950 | 894 | 79,844 | 3.00 |
| instruction_following | 0.852 | 0.832 | 55,757 | 924 | 56,681 | 2.81 |
| knowledge_update | 0.779 | 0.771 | 45,981 | 665 | 46,646 | 2.37 |
| multi_session_reasoning | 0.658 | 0.612 | 138,133 | 2,873 | 141,006 | 4.40 |
| preference_following | 0.798 | 0.777 | 51,796 | 920 | 52,716 | 2.53 |
| summarization | 0.693 | 0.537 | 158,794 | 2,905 | 161,700 | 4.44 |
| temporal_reasoning | 0.536 | 0.536 | 100,176 | 3,148 | 103,324 | 3.90 |
| **OVERALL** | **0.650** | **0.609** | **99,959** | **1,821** | **101,780** | **3.49** |

Memory Construction average token consumption (default agent, full build over 35 cases):

| Agent | input tok/case | output tok/case | total tok/case |
|---|---|---|---|
| default | 31,943,817 | 1,417,061 | 33,360,878 |
