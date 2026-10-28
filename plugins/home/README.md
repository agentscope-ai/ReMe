# Home plugin

Adds a self-evolution workspace, the one-time `home_prepare` job, `home_evaluate`, and
`auto_bash` through an independently enabled ReMe plugin.

Source code is grouped by workflow under `src/reme_home/preparation/` and
`src/reme_home/evaluation/`. Preparation contains its component and Step, config,
code export, transaction recovery, and benchmark data adapters. Evaluation contains
its component, Step, input validation, Docker request adapters, and artifacts with
failure classification. Shared workspace and host judge components live at the
package root.

Install it into the same Python environment as ReMe:

```bash
reme plugins install plugins/home
reme plugins install plugins/beam-judge  # Use plugins/lme-judge for LongMemEval.
```

Enable it with the matching judge, e.g. `plugins: [home, beam-judge]`, then start the
service and call its jobs from another terminal:

```bash
reme start config=<your-config.yaml>
# In another terminal:
reme auto_bash command=pwd
```

At application startup, Home automatically adds these directories inside the
configured ReMe `workspace_dir/home/`:

```text
code/repo/
code/runtime/
datasets/training-set/cases/
datasets/debug-cases/cases/
docs/
evaluations/
sampling/plans/
tmp/
logs/
metadata/
```

ReMe's existing memory, resource, and session directories remain in place.
On first initialization, each managed directory under `home/` must be absent or
completely empty (an empty child directory still counts as content). File
conflicts, symbolic links in managed paths, and overlap with configured ReMe
directories cause startup to fail without overwriting their contents.

Home atomically publishes `home/metadata/workspace.json` after creating
the layout. It records layout schema version 1, a workspace UUID, and an aware
creation timestamp. Subsequent starts validate this marker, preserve its identity
and existing contents, and recreate missing directories. Recreating directories
does not recover missing data. Invalid or unsupported markers fail explicitly.
A process lock at `home/metadata/workspace.lock` prevents concurrent
initialization; the lock file remains on disk, but process exit releases the
lock. A competing initializer can retry after the first finishes.

Failed initialization removes only newly created empty layout directories and
temporary manifest files. Existing directories and files are preserved. After
an abrupt process termination before the marker is published, inspect any
nonempty Home directories before retrying; they are not automatically adopted.
Older flat Home workspaces are rejected without moving or deleting files; use a new
workspace for this layout. Relative input paths and `spec_path` still resolve against
the outer ReMe workspace, not `home/`.
Initialization creates the layout only: it does not install datasets, copy
candidate code or documentation, initialize Git, or generate experiment/domain
configuration, sampling state, or evaluation results.

For explicitly one-shot CLI execution, use:

```bash
reme start config=<your-config.yaml> job=auto_bash command=pwd
```

The job defaults to read-only command validation and a 60-second timeout. Set
`is_read_only=false` to disable command validation. Supplying `read_only_paths`
or `writable_paths` enables a file sandbox (Bubblewrap on Linux, Seatbelt on
macOS), with sandbox paths identical to their host paths. On Linux, the
`memory_limit_mib` cgroup limit defaults to 4096 MiB; 0 disables it. Supplying
the limit requires a path policy. macOS has no memory limit and warns when a
nonzero value is supplied.

Standard output and standard error each have a 1 MiB (1,048,576-byte) limit on
all backends. Exceeding either limit terminates the command and returns a
failure with bounded partial output and a diagnostic in standard error.

## Prepare an experiment once

`home_prepare` runs explicitly after startup. It converts BEAM or LongMemEval
inputs, exports a committed ReMe runtime and selected plugins, and initializes a
candidate Git repository on branch `init`. It does not install dependencies, run
source code, build Docker images, call models, or evaluate the baseline.

Copy [examples/prepare.yaml](examples/prepare.yaml) into your workspace as
`prepare.yaml` and edit the source paths, revision and case IDs. For LongMemEval,
set `benchmark.name: longmemeval`, point `source` at its JSON file, remove
`variant`, and choose `runtime.plugins: [lme]` and `candidate.paths:
[plugins/lme/src]`. Optional `benchmark.evidence_source` accepts reviewed JSONL
session evidence. Choose nonempty, unique string `case_ids`, or explicitly set
`all_cases: true` without `case_ids`. Relative input paths are workspace-relative;
explicit absolute dataset and repository paths are read-only source inputs.
`spec_path` must resolve to a file inside the workspace. Home-relative paths and
relative traversal are rejected.

Start a service configured with `plugins: [home]` and your `workspace_dir`, then
call it from another terminal using the same service selection as other jobs:

```bash
reme start config=<your-config.yaml>
# In another terminal:
reme home_prepare spec_path=prepare.yaml
```

For explicitly one-shot local execution:

```bash
reme start config=<your-config.yaml> job=home_prepare spec_path=prepare.yaml
```

Python uses the same Job and normal Application lifecycle; see
[examples/prepare.py](examples/prepare.py). Pass exactly one of `spec_path` or
`spec` (a configuration dictionary):

```python
from reme.application import Application

app = Application(workspace_dir="/experiments/run-1", plugins=["home", "beam-judge"], service={"backend": "cli"})
try:
    await app.start()
    response = await app.run_job("home_prepare", spec=preparation_config)
    if not response.success:
        raise RuntimeError(response.answer)
finally:
    await app.close()
```

All prepared outputs live under `workspace_dir/home/` (paths below are relative to `home/`):

- `domain_spec.yaml`: resolved configuration, including actual selected case IDs
  and the full source commit SHA.
- `datasets/training-set/`: normalized Case/Session/Query JSON and a versioned
  manifest with case IDs and content fingerprints.
- `datasets/debug-cases/`: deterministic small smoke cases and their derivation
  record, or an empty `cases/` directory when debug derivation is disabled.
- `code/runtime/`: fixed committed ReMe sources, packaging files and configurations,
  without plugin projects. Uncommitted source edits are excluded.
- `code/repo/`: an independent Git repository containing complete selected plugin
  projects (including packaging and configuration) and any selected ReMe core paths,
  retaining repository-relative locations. `candidate.paths` defines the editable
  scope, not the complete file inventory. The initial commit preserves the baseline.
  Future runners assemble an isolated evaluation tree from runtime plus repo,
  excluding `.git`; `install_projects` paths are relative to that tree. Install
  plugins from the candidate, and never edit the fixed runtime.
- `docs/architecture-guide.md` and `docs/validation-artifact-guide.md`: guides
  shipped with the installed Home plugin, covering candidate development and
  investigation of evaluation artifacts.
- `metadata/preparation.json`: completion record, config/data/code
  fingerprints, initial candidate commit, plugin and installation plan.

Before preparation, code and docs directories must be empty, dataset directories may
contain only their empty `cases/` skeleton, and `domain_spec.yaml` must be absent.
Candidate paths must be present in the selected source, cannot overlap, and may
select ReMe core or plugin files. Runtime plugins are selected explicitly; the host plugin list is not copied. All selected source files must be regular Git blobs; symlinks and
submodules are rejected. Source references are explicit Git revisions; this
version does not support uncommitted working-tree snapshots or remote fetching.

Preparation holds the same process lock as layout initialization. Inputs are
staged beneath `home/metadata/preparation-transaction/`, flushed, and
published with a durable rollback journal. Reported preparation failures restore
pre-existing files and directories (filesystem timestamps are not part of
the contract); the corrected request can be retried. Interrupted transactions
are recovered at startup or before another preparation. A completed preparation
rejects **every** later call, including identical parameters. There is no reset or
force-overwrite option: use a new workspace for another experiment.

## Evaluate prepared candidates

`home_evaluate` runs committed candidate code through
[the shared Docker pool](../../deploy/benchmark_docker/README.md), then scores
answers through the host Application's `answer_judge` Job. Enable `beam-judge`
for BEAM or `lme-judge` for LongMemEval. Home requires exactly one supported judge
plugin; enabling additional benchmark judge plugins is an error. On a prepared
workspace, startup checks that the judge matches the recorded benchmark. On a
new workspace, `home_prepare` checks the match before preparing data or code.
A disabled or replaced `answer_judge` Job is rejected.
Install ReMe's `benchmark-docker` extra and build the documented base image first.
The host and containers need the configured LLM/embedding environment; the host
judge additionally uses `JUDGE_MODEL_NAME`. Install the corresponding judge plugin
on the host. Its code and prompts come from the host installation, never the
candidate repo; runs record a fingerprint of the judge code and configuration. Credentials are never written into
the result manifest. No initial evaluation is triggered by `home_prepare`.

Start a service with Home enabled, then call from another terminal:

```bash
reme start config=<your-config.yaml>
# In another terminal: all prepared training cases, all their queries
reme home_evaluate
```

Programs inject selections directly, without putting case IDs in the Job YAML:

```python
# app is already started with plugins=["home", "beam-judge"].
result = await app.run_job("home_evaluate", case_ids=["1", "2"], concurrency=4)

# Recover a prior pre-query memory snapshot, skip construction, answer and score again.
replayed = await app.run_job(
    "home_evaluate",
    case_ids=["1"],
    reuse_construction_from="evaluations/init/<commit-prefix>/<run-id>",
)
```

Omitted/null `case_ids` selects the entire installed dataset, including when a
reuse source is supplied. Empty, duplicate and unknown selections fail before
Docker starts. Optional `query_ids` maps case IDs to nonempty query-ID lists;
omitted entries run all queries for that case. Evaluation always uses the current
HEAD of `home/code/repo` and records its commit in the results. Run the baseline
before modifying the prepared candidate. Uncommitted modifications, staged changes
and untracked files reject evaluation before Docker starts, as do changed fixed
runtimes/datasets.
The source snapshot combines fixed ReMe with complete plugins from the selected
candidate commit; candidate core roots replace their baseline roots, including
deleted files. Source snapshots exclude `.git` and credentials.

`reuse_construction_from` must identify a completed training-set run inside this
Home's `evaluations` directory, either absolutely or relative to `home/`. There
is no `eval_only` toggle. Every selected case must have a successful construction
and a compatible, checksum-verified tar; any missing/invalid case rejects the
entire request without silently skipping or rebuilding it. Query-only replay
uses current candidate code over the donor's memory, not donor answers/scores.
The caller must establish construction equivalence when choosing reuse; this
Job does not invoke a code-equivalence review agent. Runs record both commits.

Results use `home/evaluations/<branch>/<commit-prefix>/<ascending-run-id>/`:

```text
manifest.json
summary.json
cases/<case-id>/
  execution.json
  memory_construction/
    result.json
    build.log
    reme_workspace.tar.gz
    reme_workspace/
  evaluation.tar.gz
  queries/
    answers.json
    result.json
    <query-id>/answer.log
```

The memory tar is exported **before** queries, retains session Git checkpoints,
and is independent of the post-query evaluation archive. Replayed runs keep an
independent donor tar and inherited build log without charging construction
tokens again. Answers are persisted before host judging. Per-query results keep
reference/actual answers, scores, judge output, token usage and failures. Summary
scores use the expected-query denominator; infrastructure/judge/artifact errors
make the comparable mean unavailable while retaining the observed-score mean.
The Job returns `success=false` for infrastructure/judge/artifact failures, while
preserving its structured answer and result directory; candidate failures remain
ordinary evaluation observations.
`valid_for_comparison` does not assert that an initial baseline covers the full
dataset: the caller must also verify its code and complete case/query coverage.

Run/case/query `failure.json` files preserve diagnostics. Interrupted/fail-fast
runs publish `partial_summary.json`, not a completed summary. New executions
always allocate a new run; they never overwrite previous results. For scored runs, default
`fail_fast=false` keeps independent cases running. The shared pool executes each
case's build followed by queries, rather than Meta-ReMe's global construction
barrier/query-stealing schedule. `max_construction_failure_rate` defaults to the
old strict-below-3% rule; other pool callers retain strict build-success behavior.

`debug=true` selects prepared debug cases and runs smoke inputs without a judge.
It defaults to `fail_fast=true`; scored runs (`debug=false`, the default) use
`fail_fast=false`. An explicitly supplied `fail_fast` overrides either default.
Debug requires successful construction and nonempty, error-free answers, and
returns `success=false` for candidate failures as well as infrastructure failures.
It cannot reuse construction. Case/query selections refer to the selected mode's
dataset; omitted selections run all its cases/queries.
Debug results retain execution counts and artifacts, but all mean scores are
null and `valid_for_comparison=false`. `expected_scored_query_count` is zero,
while `expected_query_count` still records the expected execution count.
The former `dataset_kind` input is rejected; use `debug` instead. Debug results
must not be used to assess answer correctness or candidate quality. No real
Docker/model evaluation is required by Home's isolated unit tests.

Only Home's writers cooperate with this lock. Keep the preparation target
locations exclusively owned by this task while preparing and recovering; other
processes must not write there. Recovery requires a writable, functioning
filesystem; if rollback fails, its journal remains for the next attempt and the
Job reports the error. Calls with `_allowed_paths` restrictions are rejected
because preparation owns multiple workspace output locations.
