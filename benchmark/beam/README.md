[中文版 / Chinese version](./README_ZH.md)

# BEAM Benchmark

BEAM is a benchmark for **memory capability over long-context chat cases**. Each case contains a very long
chat history split into batches.

## 1. Data preparation

BEAM ships dataset variants by chat size — `100K` / `500K` / `1M` / `10M` — so
memory systems can be stressed at different context lengths. Question types
include abstention, contradiction resolution, event ordering, information
extraction, instruction following, knowledge update, multi-session reasoning,
preference following, summarization, and temporal reasoning.

BEAM is a public repository, cloned into `benchmark/beam/dataset/`:

```bash
mkdir -p benchmark/beam/dataset
cd benchmark/beam/dataset
git clone https://github.com/mohammadtavakoli78/BEAM.git
```

After cloning, `benchmark/beam/dataset/BEAM/` should contain `chats/`, `src/`,
`topics/` and other subdirectories.

## 2. Use the BEAM plugin

### 2.1 Overview

The BEAM plugin loads `chat.json` for each case, converts each batch into a ReMe session, ingests sessions
chronologically into an isolated workspace, and runs `digest_update`. It then answers probing questions via
agentic (ReAct) mode and scores them with BEAM’s rubric-based `answer_judge` job, reporting graded scores,
binary verdicts, and per-type averages.

### 2.2 How to use

#### Installation and environment

Install ReMe and the BEAM plugin in editable mode from the repository root:

```bash
python -m pip install -e ".[as]"
reme plugins install ./plugins/beam --editable
reme plugins install ./plugins/beam-judge --editable
reme plugins validate beam
```

The runner explicitly enables the installed `beam` plugin and combines its defaults with
ReMe's built-in `benchmark` preset. Editable installation keeps changes under
[`plugins/beam`](../../plugins/beam/README.md) visible without reinstalling the plugin.
Custom application config paths still work through `reme.config` and can use `extends: benchmark`.
Model credentials use the environment variables declared by the shared benchmark configuration.

#### Key configuration — `benchmark/beam/config.yaml`

| Key | Meaning |
| --- | --- |
| `dataset.beam_root` | BEAM dataset root (`benchmark/beam/dataset/BEAM`). |
| `dataset.chat_size` | Variant to run: `100K` / `500K` / `1M` / `10M`. |
| `dataset.case_ids` | Specific cases (e.g. `["1","2"]`); empty = all cases. |
| `dataset.start_index` / `num_items` | Case pagination (`num_items` `0` = all). |
| `dataset.workspace_root` | Per-case workspace root (`benchmark/beam/workspaces/beam`). |
| `evaluation.num_workers` | `0` = auto, `1` = sequential, `>1` = parallel. |
| `reme.config` | ReMe config used (`benchmark`). |
| `output.dir` | Results directory (`benchmark/beam/results`). |

#### Run the benchmark

From the repository root:

```bash
python benchmark/beam/run.py
python benchmark/beam/run.py --config benchmark/beam/config.yaml
python benchmark/beam/run.py -q                        # quiet
python benchmark/beam/run.py --eval_only               # reuse existing workspaces, query + judge only
```

### 2.3 Results

Results are JSON files written to `output.dir` as
`results_<chat_size>_<timestamp>.json`, with a per-type score summary also
printed to the console. Logging conventions are shared across benchmarks — see
the [top-level README](../README.md#outputs--logs).

#### 100K

agentscope==2.0.4.post1, conda reme env, 20 workers, eval-only (reusing prebuilt memory)
(2026-08-05, 20 cases / 400 Qs, total 46.0 min)

| Type | Agentic | Binary | input tok/q | output tok/q | total tok/q | tool calls/q |
|---|---|---|---|---|---|---|
| **Overall** | **0.661** | **0.620** | **63.1K** | **1.8K** | **65.0K** | **3.33** |
| abstention | 0.550 | 0.550 | 96.0K | 1.1K | 97.1K | 4.58 |
| contradiction_resolution | 0.438 | 0.412 | 32.3K | 0.9K | 33.1K | 2.48 |
| event_ordering | 0.501 | 0.423 | 140.2K | 5.2K | 145.4K | 4.70 |
| information_extraction | 0.873 | 0.832 | 50.2K | 0.9K | 51.1K | 3.15 |
| instruction_following | 0.750 | 0.725 | 38.0K | 0.8K | 38.8K | 2.67 |
| knowledge_update | 0.688 | 0.675 | 31.2K | 0.7K | 31.8K | 2.27 |
| multi_session_reasoning | 0.626 | 0.584 | 85.0K | 4.6K | 89.6K | 4.28 |
| preference_following | 0.925 | 0.912 | 34.3K | 1.0K | 35.3K | 2.50 |
| summarization | 0.623 | 0.461 | 89.7K | 2.1K | 91.7K | 4.12 |
| temporal_reasoning | 0.637 | 0.625 | 34.6K | 1.0K | 35.6K | 2.52 |

Memory Construction average token consumption (default agent, full build over 20 cases):

| Agent | input tok/case | output tok/case | total tok/case |
|---|---|---|---|
| default | 2.2M | 136.7K | 2.3M |

#### 1M

agentscope==2.0.4.post1, conda reme env, 20 workers, full memory build
(2026-08-05, 35 cases / 700 Qs, total 459.2 min)

| Type | Agentic | Binary | input tok/q | output tok/q | total tok/q | tool calls/q |
|---|---|---|---|---|---|---|
| **Overall** | **0.650** | **0.609** | **100.0K** | **1.8K** | **101.8K** | **3.49** |
| abstention | 0.429 | 0.429 | 118.7K | 1.2K | 119.9K | 4.20 |
| contradiction_resolution | 0.391 | 0.364 | 49.8K | 0.8K | 50.6K | 2.50 |
| event_ordering | 0.558 | 0.456 | 201.5K | 3.9K | 205.4K | 4.79 |
| information_extraction | 0.809 | 0.772 | 79.0K | 0.9K | 79.8K | 3.00 |
| instruction_following | 0.852 | 0.832 | 55.8K | 0.9K | 56.7K | 2.81 |
| knowledge_update | 0.779 | 0.771 | 46.0K | 0.7K | 46.6K | 2.37 |
| multi_session_reasoning | 0.658 | 0.612 | 138.1K | 2.9K | 141.0K | 4.40 |
| preference_following | 0.798 | 0.777 | 51.8K | 0.9K | 52.7K | 2.53 |
| summarization | 0.693 | 0.537 | 158.8K | 2.9K | 161.7K | 4.44 |
| temporal_reasoning | 0.536 | 0.536 | 100.2K | 3.1K | 103.3K | 3.90 |

Memory Construction average token consumption (default agent, full build over 35 cases):

| Agent | input tok/case | output tok/case | total tok/case |
|---|---|---|---|
| default | 31.9M | 1.4M | 33.4M |

## 3. Optimize the plugin with HoMe

### 3.1 Overview

HoMe prepares an experiment workspace from the dataset and a Git revision, evaluates the baseline, and
iteratively improves the BEAM plugin code. The optimizer returns a candidate `commit_sha`, which can then be
evaluated independently on the configured evaluation set.

### 3.2 How to use

#### Experiment setup

Get the dataset above and follow the [HoMe setup and run
guide](../../plugins/home/README.md#4-run-a-benchmark-experiment).
[config_home.yaml](config_home.yaml) defaults to `benchmark/beam/workspaces/home-100K` and all **100K** cases for optimization.
[config_home_eval.yaml](config_home_eval.yaml) defaults to `benchmark/beam/workspaces/home-test-100K`
and all **100K** cases, which can overlap with training. Change `variant` or select `case_ids` to evaluate a
different set.

Prepare the optimization workspace from the repository root:

```bash
python benchmark/beam/run_home.py
```

#### Run the experiment

Run local one-shot optimization:

```bash
python benchmark/beam/run_home.py --optimize
```

Set `CANDIDATE_SHA` to the optimizer's `commit_sha` or a unique hexadecimal prefix, then run local one-shot evaluation:

```bash
python benchmark/beam/run_home_eval.py --commit "$CANDIDATE_SHA"
```

See the [HoMe evaluation guide](../../plugins/home/README.md#evaluate-through-the-service) for service-based
evaluation.

### 3.3 Results

HoMe is optimized on BEAM-100K and evaluated on BEAM-1M (35 cases, 700 queries).
Token counts and LLM calls are per-query averages for the answering stage.

| Question category | Score | Total tokens | Input tokens | Output tokens | LLM calls |
|---|---:|---:|---:|---:|---:|
| **Overall** | **0.704** | **60.4K** | **57.8K** | **2.6K** | **6.7** |
| Abstention | 0.636 | 43.4K | 41.9K | 1.5K | 6.6 |
| Contradiction Resolution | 0.705 | 22.4K | 21.0K | 1.4K | 4.6 |
| Event Ordering | 0.546 | 123.0K | 116.6K | 6.4K | 12.7 |
| Information Extraction | 0.735 | 32.5K | 31.2K | 1.2K | 5.2 |
| Instruction Following | 0.901 | 30.7K | 29.4K | 1.3K | 5.5 |
| Knowledge Update | 0.671 | 19.2K | 18.3K | 0.9K | 4.0 |
| Multi-session Reasoning | 0.682 | 93.5K | 88.1K | 5.4K | 8.5 |
| Preference Following | 0.833 | 25.0K | 23.8K | 1.2K | 4.8 |
| Summarization | 0.726 | 173.2K | 169.0K | 4.2K | 9.4 |
| Temporal Reasoning | 0.604 | 41.4K | 39.1K | 2.3K | 6.0 |
