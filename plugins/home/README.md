# HoMe

HoMe is a ReMe plugin for evaluating and improving memory system code on BEAM
or LongMemEval. It prepares an experiment from a dataset and a Git revision,
evaluates that version as the baseline, then iterates on candidate changes.
You can run the optimization agent or edit and evaluate candidates yourself.

## 1. Install and configure

From the ReMe repository root, in the Python environment where you run ReMe:

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
the corresponding benchmark configuration and judge. Supporting a different
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

For dataset downloads, runner scripts and evaluation on a separate dataset, see
the [BEAM guide](../../benchmark/beam/README.md#prepare-a-home-experiment) or the
[LongMemEval guide](../../benchmark/longmemeval/README.md#prepare-a-home-experiment).
