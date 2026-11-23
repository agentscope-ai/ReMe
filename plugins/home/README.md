# HoMe

HoMe is a ReMe plugin for evaluating and improving memory system code on BEAM
or LongMemEval. It prepares an experiment from a dataset and a Git revision,
evaluates that version as the baseline, then iterates on candidate changes.
You can run the optimization agent or edit and evaluate candidates yourself.

**Upgrade note.** For this release, prepare a new workspace for each existing
HoMe experiment you want to run again. Keep the old workspace to consult
previous evaluation results.

New evaluations retain the verified source hash and review evidence alongside
each reused snapshot. Final validation checks this retained copy even after the
source evaluation is archived. Older evaluations without the retained source
hash still require their original source artifacts for validation.

## 1. Install and configure

HoMe requires ReMe 0.4.2.4 or newer. From the ReMe repository root, in the
Python environment where you run ReMe:

```bash
pip install -e '.[benchmark-docker]'
reme plugins install plugins/home
reme plugins install plugins/beam-judge
```

For LongMemEval, install `plugins/lme-judge` in place of `plugins/beam-judge`.
Set up Docker and build the [benchmark base image](../../deploy/benchmark_docker/README.md).
Set `LLM_API_KEY`, `EMBEDDING_API_KEY`, and the base URLs for your providers
in the environment or the repository root's `.env`. Use `JUDGE_MODEL_NAME`
to override the judge model. On Linux, the default setup for optimization
and diagnosis requires Bubblewrap and a writable delegated cgroup v2.

## 2. Prepare a workspace

Copy the [example preparation file](examples/prepare.yaml) into a new workspace:

```bash
HOME_EXPERIMENT=/absolute/path/to/new-workspace
mkdir -p "$HOME_EXPERIMENT"
cp plugins/home/examples/prepare.yaml "$HOME_EXPERIMENT/prepare.yaml"
```

In `prepare.yaml`, set `benchmark.source` to your local dataset and select
`case_ids` or use `all_cases: true`. Set `runtime.source_repo` and
`runtime.source_ref` to the local Git checkout and committed revision you want
to start from. `candidate.paths` lists the files or directories that can be
edited during optimization.

For LongMemEval, set `benchmark.name: longmemeval`, point `benchmark.source` at
the dataset JSON, remove `benchmark.variant`, and set `runtime.plugins: [lme]`
and `candidate.paths: [plugins/lme/src]`. Use case IDs from that dataset.

**Using your own dataset.** Convert your conversation histories and questions
to LongMemEval format if you have reference answers, or to BEAM format if you
have grading rubrics. Point `benchmark.source` at the converted data and use
the corresponding benchmark configuration and judge. On Linux, case and query
IDs are used verbatim. On macOS, HoMe also checks for names that collide after
Unicode normalization and case folding. Query IDs must avoid reserved artifact
names such as `result.json`. HoMe checks these IDs during preparation, so you can
correct invalid IDs and retry in the same workspace. Supporting a different
input format or scoring rule requires extending the dataset loader or judge.

Start ReMe in the first terminal:

```bash
reme start config=benchmark workspace_dir="$HOME_EXPERIMENT" 'plugins=["home","beam-judge"]'
```

Use `"lme-judge"` for LongMemEval. In a second terminal, prepare the experiment:

```bash
reme home_prepare spec_path=prepare.yaml
```

Prepare once per workspace. Use a new workspace for another experiment.

## 3. Run HoMe

With the service running, run an optional unscored smoke test and then start
optimization. Start on the initial `init` branch so HoMe can evaluate the
baseline before making changes.

```bash
reme home_evaluate debug=true
reme home_optimize 'instruction=Improve retrieval quality' timeout=86400
```

The optimizer returns the selected candidate's `commit_sha`, its
`evaluation_path`, and an `optimization_report_path`. The CLI `timeout` sets
how long the client waits, in seconds. Adjust it for long experiments.

### Evaluate your own changes

Start with a clean `init` checkout. Run a baseline, create a candidate branch,
and edit the allowed files in `<workspace>/home/code/repo`:

Formal evaluations on `init` must cover every training case and query without
sampling or construction reuse. Create and commit a candidate before evaluating
a subset. Debug evaluations remain available on `init`. Optimizer retries of an
initial baseline also use the full training set.

```bash
reme home_evaluate
reme home_create_candidate_branch branch_name=my-candidate
# Edit the files under candidate.paths before committing.
reme home_commit_candidate 'message=Improve retrieval'
reme home_evaluate
```

`home_evaluate` scores all prepared cases and queries by default. It evaluates
committed code and requires a clean checkout.

### Inspect results

Use the `result_dir` from `home_evaluate` or the `evaluation_path` from
`home_optimize` in place of `RUN_DIR`:

```bash
reme home_inspect_validation validation_path=RUN_DIR
reme home_diagnose 'evaluation_paths=["RUN_DIR"]' 'instruction=Explain the lowest-scoring cases'
```

In evaluation manifests, `construction_sources` records replay provenance per
case. `home_inspect_validation` also shows `construction_source` when all reused
cases share one source. The primary construction-equivalence report is stored
under `automatic_construction_reuse.source_reviews`, keyed by its
`source_validation` path. The evaluation response derives the probability scope
and per-case reuse reasons from the automatic reuse policy and its selected cases.

## 4. Run a benchmark experiment

Install HoMe and the benchmark's judge using the setup above. Get the dataset
and check the default splits and workspace paths in the
[BEAM guide](../../benchmark/beam/README.md#experiment-setup) or the
[LongMemEval guide](../../benchmark/longmemeval/README.md#experiment-setup).
The examples below run from the ReMe repository root and use BEAM. For
LongMemEval, use `benchmark/longmemeval` and the `lme-judge` plugin.

### Prepare and optimize

```bash
python benchmark/beam/run_home.py
```

This reads the benchmark's `config_home.yaml` to prepare the workspace, dataset,
and candidate code. `runtime.source_ref: HEAD` uses the committed source. Use
`--config` to load another runner YAML and `--workspace-dir` to choose a new
workspace. Paths in the runner YAML are relative to the repository root.
Prepare once per workspace.

To optimize through the service, start ReMe using the prepared workspace in
one terminal, then run `reme home_optimize` in a second terminal as described
in [Run HoMe](#3-run-home).

#### Local one-shot optimization

```bash
python benchmark/beam/run_home.py --optimize --instruction "Improve retrieval quality"
```

Reuse any `--config` and `--workspace-dir` options from preparation.
The runner applies the YAML's `sampling_policy`, `max_evaluations` (default 10),
and `validation_cycle_time_budget_minutes` (default 150, null disables reminders).
The formal evaluation budget counts runs once their sandbox starts, excluding
baseline and debug. `memory_limit_mib` sets the host agent sandbox memory limit.
On Linux, use 0 to disable it or omit it to retain the 4096 MiB default.

### Evaluate a candidate in an independent workspace

Pass the optimizer's `commit_sha` or a unique hexadecimal prefix of at least
four characters as `--commit`. Git resolves it in the source workspace's
candidate repository. If a prefix is ambiguous, use a longer one.

The runner copies the fixed runtime and exports the selected commit from the
prepared source workspace into a separate test workspace. It reads dataset
selection and workspace paths from the benchmark's `config_home_eval.yaml`.
Use `--config`, `--source-workspace`, or `--test-workspace` to override those
defaults. The workspaces must not overlap. Rerun with the same candidate and
dataset to reuse unchanged prepared inputs, or use a new test workspace to
change either.

Both modes below rebuild memory and score all selected queries using Docker and
the configured judge. They save results to `home/evaluations/` in the test
workspace with optimization history and sampling disabled. The preparation
record and evaluation manifests record the source candidate under
`test_evaluation`, and `evaluation_commit_sha` identifies its local export.

#### Evaluate through the service

Prepare the test workspace, then start the service and call the job from
separate terminals. Set `CANDIDATE_SHA` to your candidate's ID or unique prefix.
Use the test workspace from your runner config in both terminals.

```bash
python benchmark/beam/run_home_eval.py --commit "$CANDIDATE_SHA" --prepare-only
# Terminal 1
reme start config=benchmark workspace_dir=benchmark/beam/workspaces/home-test-100K 'plugins=["home","beam-judge"]'
# Terminal 2
reme home_evaluate workspace_dir=benchmark/beam/workspaces/home-test-100K record_history=false
```

Pass any evaluation overrides, such as `concurrency` or `timeout_seconds`,
to `reme home_evaluate`.

#### Local one-shot evaluation

```bash
python benchmark/beam/run_home_eval.py --commit "$CANDIDATE_SHA"
```

This prepares and evaluates in one command and uses the runner YAML's
`evaluation` settings.
