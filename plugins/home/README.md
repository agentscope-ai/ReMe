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
- `datasets/training-set/`: normalized Case/Session/Query JSON and a
  manifest with case IDs and content fingerprints.
- `datasets/debug-cases/`: deterministic small smoke cases with source case IDs
  in their metadata, or an empty `cases/` directory when debug derivation is disabled.
- `code/runtime/` and `code/repo/`: fixed committed runtime and candidate Git
  repository. See the [workspace and modification boundary](src/reme_home/runtime_docs/architecture-guide.md#workspace-and-modification-boundary)
  for their roles and editable scope. Source exports exclude uncommitted edits;
  `install_projects` paths are relative to the assembled evaluation tree.
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
For explicit `reuse_construction_from`, the caller must establish construction
equivalence. Automatic sampled/budgeted selection instead invokes the host's
equivalence reviewer and 60% replay policy. Runs record both commits.

Results use `home/evaluations/<branch>/<commit-prefix>/<ascending-run-id>/`.
The [artifact guide](src/reme_home/runtime_docs/validation-artifact-guide.md#saved-files)
is the reference for saved files, authoritative query records, compaction and
failure evidence. Its [summary reference](src/reme_home/runtime_docs/validation-artifact-guide.md#summary-fields)
covers score denominators, token accounting, debug fields and summary rebuilding.
Before comparing runs, verify completed scored execution, infrastructure outcomes,
candidate code and complete case/query coverage.

The Job returns `success=false` for infrastructure/judge/artifact failures while
preserving its structured answer and result directory; candidate failures remain
ordinary evaluation observations. Successful Jobs returning empty answers count
as candidate failures. Construction failure-rate tolerance applies only to
candidate errors; infrastructure or unclassified failures always reject construction.
See [failure interpretation](src/reme_home/runtime_docs/validation-artifact-guide.md#investigating-a-result)
for structured error classification.

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
Debug results retain execution counts and artifacts, with null mean scores.
See the [summary reference](src/reme_home/runtime_docs/validation-artifact-guide.md#summary-fields)
for coverage fields and manifest mode.

The former `dataset_kind` input is rejected; use `debug` instead. Debug results
must not be used to assess answer correctness or candidate quality. No real
Docker/model evaluation is required by Home's isolated unit tests.

Only Home's writers cooperate with this lock. Keep the preparation target
locations exclusively owned by this task while preparing and recovering; other
processes must not write there. Recovery requires a writable, functioning
filesystem; if rollback fails, its journal remains for the next attempt and the
Job reports the error. Calls with `_allowed_paths` restrictions are rejected
because preparation owns multiple workspace output locations.

## Bounded agent foundation

`reme_home.agent.BoundedAgentWrapper` inherits ReMe's `AsAgentWrapper` and is
registered as the `home_bounded` agent-wrapper backend when Home is enabled.
It reuses ReMe's Job tools, hidden `injected_job_kwargs`, `tool_context_id`,
built-in tools, permissions, model configuration, skills, token accounting and
persisted sessions. Home configures a wrapper for the single-case investigation
Job below; no investigation or optimization runs automatically. Configure other
agents explicitly, alongside the required model and Home plugins:

```yaml
components:
  agent_wrapper:
    investigator:
      backend: home_bounded
      as_llm: default
      react_config:
        max_iters: 40
      time_budget_minutes: 30  # Optional advisory budget; null disables it.
      job_tools: [home_evaluate]
      injected_job_kwargs:
        case_ids: ["1"]  # Hidden from the model and enforced by the host.
```

Call `reply()` or `reply_stream()` through the normal ReMe wrapper interface;
call-time options override component defaults. Each built agent owns its own
timer and request state. The internal `BoundedReActAgent` still subclasses the AS
runtime to implement loop hooks; callers use `BoundedAgentWrapper` for ReMe wiring.
The lower-level `BoundedReActAgent` and `RunTimeBudgetTimer` exports remain available.

Each reply invocation starts and stops its advisory timer, including on errors and
cancellation. Iteration reminders occur at 50%, 75%, 90% and 95%; coincident or
skipped thresholds produce one reminder. Time reminders occur at 60%, 75%, 90%,
100%, then each additional 10%, alternating temporary and retained messages. Time
budgets do not cancel model calls or tools. Completed context replacements are
counted in time reminders. Reminders use native AS hint blocks, preserving the
reply's cumulative token usage across reasoning and tool calls. Use separate
agent instances for concurrent work.

The AgentScope loop owns the iteration cap and its single finalization call.
For text reports, Home removes all tools from that call, retains the original
task, and trims oldest messages if needed to fit the context budget. The final
report is still attempted if its preceding context compression fails; ordinary
compression errors outside finalization and cancellation continue to propagate.
The final response is buffered until complete, including for streaming models; unsolicited
tool calls are discarded. If finalization fails or returns no text, the reply
explicitly identifies any retained prior model output as incomplete. Cancellation
and provider interruption keep AgentScope's interruption behavior. Native
structured-output requests retain the SDK's schema and grace-iteration behavior.
Events and final-message token usage retain AgentScope's existing semantics.

## Candidate versions and code inspection

Home registers the `home.code` Component and ten ordinary Step-backed Jobs. They
operate only on the prepared `home/code/repo`, sharing the preparation/evaluation
process lock. Enable Home and prepare the workspace before calling them.

Branch names must be valid local Git branch names of 1–100 characters and serve as
single evaluation-directory names, so `/` and `\` are forbidden.

| Job | Inputs and behavior |
| --- | --- |
| `home_code_status` | Current branch, full commit SHA, exact changed paths and pending merge state. |
| `home_list_candidate_branches` | Local branches and candidates, excluding protected `init` from the candidate count. |
| `home_create_candidate_branch` | `branch_name`: fork current HEAD and switch to a new candidate. |
| `home_switch_candidate_branch` | `branch_name`: switch to an existing local branch, including `init`. |
| `home_merge_candidate_branch` | `branch_name`: merge another candidate without committing; conflicts are aborted. |
| `home_commit_candidate` | `message`: stage and commit all visible changes within prepared `candidate_paths`. |
| `home_code_diff` | Omit inputs for tracked changes against HEAD; supply `base_revision` and `revision` to compare commits. With only `base_revision`, compare the working tree against that commit. |
| `home_list_code_tree` | Optional `revision`, literal `path`, and `max_results` (default 500, maximum 2000). |
| `home_read_code_file` | Required `path`; optional `revision`, one-based `start_line` and inclusive `end_line`. |
| `home_search_code` | Required Python regex `pattern`; optional `revision`, literal `path`, and `max_results` (maximum 30). |

Start the configured service in one terminal, then use a second terminal:

```bash
reme start config=/absolute/path/to/home.yaml
```

```bash
reme home_code_status
reme home_create_candidate_branch branch_name=retrieval-routing
# Edit files under the prepared candidate_paths, then inspect and commit.
reme home_code_diff
reme home_commit_candidate message='Improve evidence routing'
reme home_evaluate debug=true
reme home_evaluate
```

Branch names must be single path-safe names without slashes. Creation, switching
and merging require a clean checkout, including untracked files and pending
merges. `init` must continue to point to the preparation's initial commit; commits
and merges into it are rejected. Detached candidates, symlinks, submodules and
changes outside `candidate_paths` are rejected by the managed mutation tools.
Merges require an explicit later commit. No remote operations, branch deletion,
amend, reset or rebase are exposed. Empty ordinary commits fail through Git.
These guarantees apply to cooperating Home Jobs; do not grant an optimizer
unrestricted shell writes or direct access to `.git` if it must obey this boundary.
Cancellation waits for already-started Git work to finish before returning.

Code-navigation Jobs resolve `revision` (default `HEAD`) to a full SHA once per
call, read raw committed objects and never execute candidate code. Dirty working
files and Git archive attributes do not affect these reads. Pin the same SHA
across a diagnostic session through ReMe's hidden injection:

```yaml
job_tools: [home_list_code_tree, home_read_code_file, home_search_code]
injected_job_kwargs:
  revision: "<full-commit-sha>"
```

The wrapper hides injected fields from the model and rejects attempts to supply
them. For comparisons, configure two agent invocations or two named Job aliases
with separately pinned revisions. `HomeCodeStep` takes a fixed `operation` in
Step configuration; repository location and operation are never model-selected.
The Python entry point is `await home_code.run(operation, **arguments)`.

Reads are limited to 300 lines and 100,000 content characters, with explicit
truncation flags. Search accepts Python multiline regex, merges adjacent match
contexts, and displays at most five lines per result block and 100,000 content
characters. Files above 5 MB and searches above 100 MB are rejected; regex
matching runs in a separate process with a ten-second timeout. Use narrower
paths or line ranges to continue inspecting. These tool-operation timeouts are
independent of the bounded agent's advisory time budget.

Evaluation also checks the standalone repository identity and protected baseline
before taking its existing clean-HEAD snapshot. Formal evaluation deduplication,
automatic ancestor/donor selection and optimization scheduling remain separate
policies; these Jobs do not enable an optimizer or change evaluation selection.

## Evaluation sampling and history

`home_evaluate` is the single evaluation Job. Set `sampling=true` for adaptive
case/query selection, or leave it false for a full baseline or an explicit
host-provided `case_ids`/`query_ids` selection. `sampling=true` conflicts with
explicit selections in scored mode. `debug=true` ignores the sampling flag,
uses debug cases and never reads, writes or registers evaluation history.
The former `home_validate` Job is removed; use `home_evaluate sampling=true`.

Scored evaluations reject a Git tree already evaluated anywhere in this workspace,
including another branch or an empty commit. This applies to full, sampled and
construction-replay runs, even with `record_history=false`. Changing selections or
evaluation options does not bypass the check. The error points to the previous
result; inspect it and commit a code change before evaluating again. The Job returns
`success=false` with an answer containing `status="rejected"`,
`error_code="UNCHANGED_CANDIDATE"`, `error` and the previous `result_dir`.
`metadata.result_dir` also points to that result. Debug runs neither trigger nor
participate in this check.

An infrastructure failure or interrupted attempt can retry the exact same commit
with a new run ID; the previous artifacts remain intact. Recovery uses exactly the
failed attempt's case/query IDs and order, even if the caller changes sampling,
budgets or explicit selections. It reruns the selected batch rather than only
failed queries, and records an explicit selection plan with `recovery_from`
pointing to the prior run. Other execution options are not frozen; construction
reuse still follows `reuse_construction_from`. Candidate-code failures
and valid low scores require a code change. A completed summary with infrastructure
errors allows retry even when other cases succeeded; an incomplete attempt with
candidate-failure evidence and no infrastructure failure does not. An empty commit
cannot convert a recoverable attempt into a new candidate. The guard reads saved
artifacts under the formal evaluation lock and does not depend on sampling caches
or successful-history registration.

Selection and optimization-history recording are independent. `record_history`
defaults to `true` for compatibility: a first full optimization baseline contributes
observations even with `sampling=false`. Set `record_history=false` to score code
without feeding optimization history. This still saves the manifest, scores and
result artifacts, but never registers a history entry, increments either counter
or updates sampling observations. Rebuilding history preserves that exclusion.
Debug always skips history regardless of either flag.

Evaluation parameters have distinct names and lifetimes:

| Parameter | Where it belongs | Meaning |
| --- | --- | --- |
| `sampling` | Job input, default `false` | Enable adaptive selection for this invocation. |
| `record_history` | Job input, default `true` | Record started rounds and successful results for optimization. |
| `sampling_policy` | Component defaults or host Job input | Case/query limits and weighting settings. |

Home registers these Component defaults automatically; override the policy in
application configuration when needed:

```yaml
components:
  home.evaluation:
    default:
      backend: home_evaluation
      sampling_policy:
        max_cases: 5
        max_queries: null
        temperature: 1.0
        ewma_alpha: 0.3
        uniform_exploration: 0.2
```

The Component policy is immutable. Each invocation validates its own policy,
merging a partial `sampling_policy` over Component defaults and persisting the
effective values with that run. Explicit null `max_queries` removes the query
budget. The previous Component configuration key `sampling` is now
`sampling_policy`; passing a policy object as the Job's boolean `sampling` fails.
Normal ReMe Job defaults also work: set `sampling: true` and optionally
`sampling_policy` directly under `jobs.home_evaluate`. Call-time values take
precedence over same-named Job defaults; a call-time policy object replaces the
Job-default policy object, then merges with Component defaults.

With the configured service running, use a second terminal. These are alternative
invocations; commit a code change between scored evaluations:

```bash
reme home_evaluate sampling=false          # Full baseline.
reme home_evaluate sampling=false record_history=false  # Standalone full evaluation.
reme home_evaluate sampling=true           # Adaptive formal evaluation.
reme home_evaluate debug=true sampling=true  # Debug only; no history feedback.
```

Host Python callers can override the policy for one invocation without changing
the next invocation's defaults:

```python
result = await app.run_job(
    "home_evaluate",
    sampling=True,
    sampling_policy={"max_cases": 3, "max_queries": 12},
)

# After committing a code change, score specified prepared cases without
# affecting the optimizer's history.
standalone = await app.run_job(
    "home_evaluate",
    sampling=False,
    record_history=False,
    case_ids=["case-a", "case-b"],  # Omit to run the full prepared dataset.
)
```

`sampling_policy` and explicit case/query IDs stay out of the default agent tool
schema. An optimizer that must obey a fixed budget should receive hidden,
enforced values through the normal wrapper injection:

```yaml
job_tools: [home_evaluate]
injected_job_kwargs:
  sampling: true
  record_history: true
  sampling_policy:
    max_cases: 3
    max_queries: 12
```

The wrapper hides injected parameters and rejects model attempts to provide
those keys. Use ordinary call arguments when the host should decide the mode
per invocation; use injection when that decision must remain fixed for an agent.

Case/query selection retains dev_meta b53048e4's EWMA loss, staleness, observation
uncertainty, un-rechecked error priority, normalized adaptive/uniform mixture,
weighted drawing without replacement and fair-share query budget allocation.
Policy defaults are unchanged. Positive integer case/query limits, positive
finite temperature, alpha in `(0, 1]` and uniform exploration in `[0, 1]` are
validated. Seeds, draw weights, eligibility and exact selections are saved.
An explicit construction donor uses the existing dev_meta eligibility rules and
checksum checks. A later full rebuild considers the full training set again.

### Successful evaluation history

The `home.evaluation_history` Component discovers successful scored training
results and rebuilds sampling observations from saved evidence. Low scores remain
valid observations; infrastructure failures, debug and incomplete runs are excluded.
See [history and sampling evidence](src/reme_home/runtime_docs/validation-artifact-guide.md#evaluation-history-and-adaptive-sampling-evidence)
for eligibility, started/successful rounds, saved plans and cache recovery.
The full summary is the durable completion boundary: cancellation during later
history bookkeeping does not relabel the completed evaluation. Failed registration
discards only derived caches, which can rebuild from saved run evidence.

```python
history = app.context.components["home.evaluation_history"]["default"]
results = history.list_results()  # Typed EvaluationRecord objects, chronological.
result = history.get_result(results[-1].run_key)
history.rebuild()                # Rebuild sampling observations from saved artifacts.
```

These synchronous filesystem methods own the workspace lock; async host code can
run them in a worker thread. Formal runs remain serialized by
`home/metadata/validation.lock`; debug bypasses that lock. Rebuilding changes only
derived state; preserve source manifests, plans and result artifacts.

## Evaluation result inspection

Read-only inspection includes debug, failed, cancelled, running and standalone
(`record_history=false`) runs, independently of successful optimization history.
It never registers results, updates sampling, rebuilds summaries or writes artifacts.

| Job | Use |
| --- | --- |
| `home_list_validation_results` | Find runs, optionally filtered by `branch_name` and `commit_sha`. |
| `home_inspect_validation` | Inspect a `validation_path`, then narrow with `case_id` and `query_id`. |
| `home_trace_query_evidence` | Link a query to source sessions, memory checkpoints and saved evidence without a model call. |
| `home_inspect_memory_history` | Page through one case's exported memory Git checkpoints. |
| `home_compare_memory` | Compare committed memory within a run or across evaluations of the same case. |

With the configured service running, use a second terminal:

```bash
reme home_list_validation_results branch_name=candidate limit=10
reme home_inspect_validation validation_path=evaluations/candidate/abcdef0/1
reme home_inspect_validation validation_path=evaluations/candidate/abcdef0/1 case_id=case1 query_id=query1
reme home_trace_query_evidence validation_path=evaluations/candidate/abcdef0/1 case_id=case1 query_id=query1
reme home_inspect_memory_history validation_path=evaluations/candidate/abcdef0/1 case_id=case1
reme home_compare_memory validation_path=evaluations/candidate/abcdef0/1 case_id=case1 base_ref=HEAD~1
reme home_compare_memory validation_path=evaluations/candidate/abcdef0/1 target_validation_path=evaluations/candidate/1234567/1 case_id=case1
```

`validation_path` is absolute or relative to `home/`. See
[read-only evaluation inspection](src/reme_home/runtime_docs/validation-artifact-guide.md#read-only-evaluation-inspection)
for pagination, commit-prefix resolution, previews and source pointers, and
[memory checkpoint tools](src/reme_home/runtime_docs/validation-artifact-guide.md#memory-checkpoint-tools)
for evidence provenance, missing-history states, comparison references and limits.

Host callers use the same Jobs through `app.run_job()`. For case-scoped agent
tools, pin `validation_path` and `case_id` through `injected_job_kwargs`;
the wrapper hides those fields and rejects model overrides. These Jobs reject
path escapes, symlinks and restricted `_allowed_paths` contexts.

## Case-study batches

`home_case_study(evaluation_path, case_ids, focus=...)` investigates up to eight
unique case IDs in input order, sharing the evaluation and unchanged focus.
Cases run concurrently, each with fresh agent state, sandbox and timer. The
host Step setting `concurrency` defaults to 4, matching dev_meta, and must be a
positive integer; set it to 1 for sequential execution. Results preserve input
order regardless of completion order. Cancelling the batch cancels running and
queued cases and waits for their cleanup before returning. The
whole selection is validated before model calls; a case execution failure is
returned individually without cancelling later cases. It follows dev_meta's
case-study workflow: trace judging and answer failures backward through retrieval,
memory construction, source sessions and
the evaluated implementation. The text report cites evidence paths/lines or
JSON fields and distinguishes supported causes, hypotheses and missing evidence;
it does not prescribe fixes.

The `home_case_study` component uses `home_case_study_agent`, a thin subclass of
the bounded wrapper, defaulting to the dedicated `as_llm: home_case_study`
component with model `qwen3.7-max`, 60 iterations and a 30-minute advisory
budget, matching dev_meta's default budgets. It uses `LLM_BACKEND`, `LLM_API_KEY`
and `LLM_BASE_URL` without changing the evaluation judge's model. Override
`components.as_llm.home_case_study.model` to choose another model, or
`components.agent_wrapper.home_case_study.as_llm` to select another host component. ReMe owns model
startup, token accounting and saved session transcripts. The response answer is
`{"results": [...]}`; each item contains `case_id`, `success`, and either `report`
with its session ID/usage or `error`. Metadata includes `evaluation_path`, the
evaluated commit and total `usage` (null if any case fails or usage is unavailable).
Response success is false if any case fails; completed sibling reports remain
available. No case scratch directory or intermediate analysis files are created.

`validation_path` remains an alias for `evaluation_path`. Legacy `case_id` calls
still return a text report with per-case metadata. Do not supply both path names
or both case selectors. Other inspection Jobs retain their `validation_path` argument.

The case-study toolkit and mandatory read-only OS sandbox are described in the
[artifact guide](src/reme_home/runtime_docs/validation-artifact-guide.md#single-case-diagnostic-agent).
The host pins case/evaluation selectors and code revision. Direct code reads can
see current worktree changes; use revision-pinned Jobs to establish evaluated behavior.

`python_execute` accepts either `code` or an existing `file`, optional file-only
`args`, and a `timeout` in seconds (default 60, maximum 600). It uses `python -I -B`
and returns returncode/stdout/stderr/timed_out. Both executors reuse bounded output,
timeouts, cancellation cleanup and network isolation from the platform backend.

Linux requires Bubblewrap and, with the default 4096-MiB limit, a writable
delegated cgroup v2. macOS requires Seatbelt (`sandbox-exec`); memory limits are
not enforced there. The Job probes sandbox startup before calling the model and
never falls back to an unrestricted executor. The command whitelist parser is
not used by this agent. Batch concurrency, sandbox policy, output-report token guidance and executor
memory policy are host Step configuration, not model arguments:

```yaml
jobs:
  home_case_study:
    steps:
      - backend: home_case_study_step
        agent_wrapper: home_case_study
        concurrency: 4
        output_token_limit: 2500
        # memory_limit_mib: 0  # Explicitly disable Linux cgroup memory limiting if needed.
```

With the configured service running, call from another terminal:

```bash
reme home_case_study evaluation_path=evaluations/candidate/abcdef0/1 'case_ids=["case1","case2"]' 'focus=Trace low-scoring queries'
```

Calls with `_allowed_paths` restrictions fail closed, as other Home artifact
Jobs do. Structured inspection tools run in the host with their existing
containment checks; only Bash/Python subprocesses run inside the OS sandbox.

## Diagnostic and optimizer agents

`home_diagnose` and `home_optimize` provide the minimal Meta-ReMe workflow using
Home's existing Jobs. Enable Home with a benchmark judge and prepare the
workspace first. Start the configured service in one terminal:

```bash
reme start config=/absolute/path/to/home.yaml
```

Call from another terminal:

```bash
reme home_diagnose 'evaluation_paths=["evaluations/init/abcdef0/1"]' 'instruction=Explain the lowest-scoring cases'
reme home_optimize 'instruction=Improve evidence retrieval while reducing query tokens'
```

Diagnosis requires explicit, unique result directories, including failed runs
with saved evidence. Each call creates a fresh diagnostic agent, delegates deep
case investigation to `home_case_study`, and returns ranked, evidence-supported
causes without prescribing fixes. An optimizer invocation automatically injects
its available caller messages and compression summary; the model cannot supply
or override that trajectory. Standalone calls record an empty trajectory.
Reports and trajectories are saved under `home/logs/diagnostic-reports/` and
`home/logs/diagnostic-trajectories/` with their evidence paths and session IDs.

Optimization reuses a completed full initial baseline covering every prepared
training case and query, with no infrastructure errors. Otherwise it runs that
baseline first, requiring clean `init` to be checked out. It never switches away
from an existing candidate to repair a missing baseline. Baseline observations
feed sampling history. Candidate execution failures retain Home's existing
failure thresholds; low scores are valid evidence. Infrastructure failures stop
baseline startup and are distinguished from candidate defects during iteration.

The optimizer diagnoses, edits declared `candidate_paths`, commits through
managed Git Jobs, runs debug until execution passes, then evaluates using the
host's adaptive sampling policy. The host automatically chooses whether to
rebuild memory or replay compatible construction snapshots, as described below.
The agent cannot change execution settings, selections, sampling mode, or
construction reuse through its evaluation tool; only `debug` is model-selected.
Normal `home_evaluate` callers retain their existing interface and replay support.
Unchanged candidates and infrastructure retries follow the existing repetition
rules. There is no architecture agent in this workflow.

Sampled or budgeted formal evaluations automatically find the nearest ancestor
commit with a completed scored training run and reusable construction snapshots.
The host checks dataset/runtime/config/model fingerprints, archive integrity,
successful source answers and the existing 5% construction-failure limit. Invalid
sources are skipped while looking for an older usable ancestor. The
`home_construction_equivalence` agent compares the exact source and candidate
commits. If construction is unchanged, the host draws once: 60% query-only replay,
40% full construction and evaluation. Changed or uncertain construction always
rebuilds. Substantive changes to `auto_memory.py` or its YAML/hyphenated variants
conservatively force rebuilding without a model call, matching dev_meta.

The optimizer cannot request query-only evaluation, choose a source, alter the
probability or bypass review. Its evaluation tool still exposes only `debug`.
Baseline and debug always rebuild; ordinary unsampled, unbudgeted calls and
explicit host `reuse_construction_from` calls retain their existing interfaces.
Replay sampling stays within reusable source cases and may select new queries
from those cases. Full rebuild sampling considers all prepared cases. An
infrastructure retry preserves the previous mode, source and frozen selection
without another review or probability draw. Both full and automatically reviewed
query-only results can support the optimizer's final selection; the initial
baseline must still be full and cover all cases/queries.

The equivalence agent receives one read-only export under
`home/tmp/construction-equivalence-*/`: both pinned source trees, their complete
diff, a copy of the fixed runtime and configuration, and only the chosen source's
pre-query construction workspaces, build logs and minimal construction metadata.
Git metadata is removed from exported workspaces. It cannot read live code,
datasets, query results, scores, other evaluations, optimizer trajectory or host
logs. Read, Glob, Grep and Bash all use the OS sandbox; there are no writable
paths, Home Job tools or network access. Reports are host-written under
`home/logs/construction-equivalence/`. Temporary exports, including partial copies,
are deleted when review finishes, fails or is cancelled. Reports retain commit and
source-evaluation provenance; `source_path` records the temporary location used
during review and is no longer available afterward. Model or malformed-report failures are
recorded and force a full rebuild; sandbox setup failure stops evaluation.
Both construction-equivalence and candidate-admission reports accept JSON wrapped
in prose or a code fence, ignore think blocks, and repair trailing commas outside
strings. If parsing or schema validation still fails, the reviewer makes one
format-only structured-output call with thinking disabled and no tools. This call
retains the system prompt, compression summary and investigation history, including
the invalid final reply. Successful corrections are saved in the session and their
token usage is included in the report. If correction fails, the existing fallback
applies: construction is rebuilt or candidate admission allows evaluation.
Reviewer shell commands receive an explicit minimal environment instead of
inheriting host model credentials or other environment values.
Cancellation propagates, and a branch/commit change during review fails before
sampling or spending a validation slot.

Evaluation manifests and responses retain `automatic_construction_reuse`,
including source/base/candidate commits, the equivalence report and, when drawn,
the probability, random draw and selected mode. Configure the reviewer through
`components.as_llm.home_construction_equivalence` (default `qwen3.8-max`, 128,000
context tokens) and
`components.agent_wrapper.home_construction_equivalence.react_config.max_iters`
(default 100). Optional
`components.home.evaluation.default.construction_equivalence_memory_limit_mib`
uses the existing OS sandbox memory-limit contract.

During the first `max_evaluations // 2` non-init formal starts, a fresh
`home_candidate_gate` agent reviews each new candidate before sampling or starting
an evaluation sandbox. It compares the candidate with its nearest previously
evaluated ancestor (falling back to `init`). It accepts runtime mechanism changes
and concrete execution, safety or contract fixes; it rejects prompt/parameter-only
tuning, cosmetic changes and dead code during this early exploration phase.
Ambiguous evidence is accepted. Baselines, debug, infrastructure retries and later
refinement bypass admission. Ordinary unbudgeted `home_evaluate` calls remain
ungated; supplying `max_evaluations` enables the same admission policy.

The reviewer receives only read-only source exports of the two exact commits and
their complete diff under `home/tmp/candidate-gate-*/`. Its Read, Glob, Grep and
Bash tools all use the OS sandbox, with no writable project paths or network.
The live candidate repository, Git metadata, fixed runtime, datasets, evaluations,
logs and optimizer trajectory are not granted. It cannot invoke Home Jobs.
Sandbox setup failure stops evaluation; model or malformed-report failures are
recorded and allow evaluation, following dev_meta's admission fallback policy.
Cancellation propagates. A candidate branch/commit change during review requires
a retry rather than evaluating unreviewed code.

`CANDIDATE_GATE_REJECTED` returns the reason and review path without drawing a
sample, allocating evaluation artifacts or spending a validation slot. Host-written
reports under `home/logs/candidate-gates/` retain both commits, decision, reason,
session and usage; accepted reviews are also linked from evaluation manifests.
Source exports, including partial copies, are deleted on completion, failure or
cancellation; the report's `source_path` records their former temporary location.
Configure the model via `components.as_llm.home_candidate_gate` and the 60-iteration
default via `components.agent_wrapper.home_candidate_gate.react_config.max_iters`.
The model defaults to `qwen3.8-max` with a 128,000-token context. Optional
`components.home.evaluation.default.candidate_gate_memory_limit_mib` uses the
existing OS sandbox memory-limit contract.

Both agents share the case-study OS sandbox implementation. All file tools,
Bash and Python use it, including file metadata and Edit/Write checks. Evidence,
fixed runtime and Git metadata are read-only. Each call receives a private
scratch directory under `home/tmp/`; only the optimizer additionally receives
writes to declared candidate roots. Network is denied. Diagnostic writes are
limited to scratch. Host Jobs alone write reports, evaluation artifacts and Git
metadata. Sandbox startup is probed before model calls and never falls back to
unrestricted execution. Declared write roots must exist and remain real paths.
Only one optimizer can run per Home workspace; cancellation releases its lock
after nested tool cleanup. Separate diagnostic calls and case studies may run
concurrently with fresh state.

The final optimizer response must identify `branch_name`, full `commit_sha`,
`evaluation_path` and `report`. The host verifies a clean checkout at that exact
branch/commit and a matching completed, scored, end-to-end evaluation without
infrastructure failures. Sampled formal evidence and host-selected,
construction-equivalent query-only evidence qualify; debug evidence does not.
Keeping the initial baseline is valid. Invalid final claims
fail while preserving changes and the raw report in
`home/logs/optimization-reports/`. The host verifies provenance rather than
ranking candidates by a single average score.

Dedicated `as_llm` and `agent_wrapper` components are named `home_optimizer` and
`home_diagnostic`; override them through ordinary application configuration.
Both default to `qwen3.8-max`, with context sizes 160,000 and 128,000, respectively.
The optimizer has 3,000 iterations and a 150-minute advisory validation-cycle timer;
configure `components.agent_wrapper.home_optimizer.validation_cycle_time_budget_minutes`
(null disables timed reminders). The previous `time_budget_minutes` setting is a
fallback when the cycle setting is absent. Diagnosis retains 80 iterations and
a 60-minute whole-invocation timer. Neither timer enforces a time limit.

The optimizer follows dev_meta b53048e4: before reasoning it checks 50%, 75%, 90%,
then 100% and each additional 10% of the cycle budget. Reminders include elapsed
time, budget percentage, completed context compressions and guidance to finish
editing, commit, pass the smoke test and begin validation. Temporary and durable
reminders alternate, beginning with temporary. The cycle includes nested diagnosis,
debug and evaluation time. Only a new counted formal evaluation returning
`success` or `completed` resets the clock, thresholds, alternation and compression
count; scores and candidate failures do not control reset. As in dev_meta, a
completed result can reset even when its case results contain infrastructure errors;
an incomplete failure, rejected call, cancellation, debug or pre-sandbox failure
cannot. The initial full baseline precedes the first optimizer cycle and is excluded
from its formal-validation numbering. A durable hint records the completed
validation number and commit prefix, expires that cycle's old timer reminders and
announces the next cycle with the remaining formal-validation count.
Persisted old reminders are retained as in dev_meta.
Hints use AgentScope's current reply message so accumulated token usage is retained.
Evaluation tools queue transitions in request state; the optimizer injects them
before the next reasoning call, after all tool results have been written. Thus a
transition never separates an API tool call from its matching result.

`components.agent_wrapper.home_optimizer.max_evaluations` defaults to 10, matching
dev_meta. This setting and the `home_evaluate` argument were renamed from
`max_validations`; update existing configuration and callers to `max_evaluations`.
The old name is rejected rather than silently falling back to the default budget.
`get_validation_budget` reports consumed and remaining slots, the search
phase and next-candidate guidance. A non-init formal run consumes one slot when
its first sandbox initializes successfully, including runs that later fail or are
cancelled. Initial baselines, debug, rejected calls and pre-start failures are free.
Accounting reuses persistent started-run evidence and survives optimizer restarts;
removing the rebuildable sampling cache does not restore the budget. The evaluator
checks the limit while holding the formal-validation lock, before starting another
run. At exhaustion it returns `VALIDATION_BUDGET_EXHAUSTED`; the optimizer selects
from existing evidence. The first `max_evaluations // 2` starts favor structural
exploration; later starts favor evidence-driven refinement. This guidance is advisory.
Ordinary `home_evaluate` calls remain unbounded unless the host supplies
`max_evaluations`; all recorded non-init formal starts in that workspace count.

Sessions and model usage follow ReMe's wrapper conventions; each nested call reports its own usage.
The optimizer follows dev_meta's compression recovery policy. Recoverable model
or summary-output failures leave context intact below the emergency ratio
`(trigger_ratio + 1) / 2`. At or above that ratio (90% for an 80% trigger), old
messages are atomically saved under `logs/context-offloads/`; recent context and
the existing summary are retained, with a reminder pointing to the readable JSON
file. This fallback makes no additional model call and counts as one completed
compression. Cancellation, invalid configuration and uncompressible system prompts
still propagate during normal operation. Diagnostic and case-study agents retain
their existing compression behavior. Bounded draft notebook tools are not included.

Host-only execution settings belong in the optimizer Step; model, iteration and
timer settings belong in its wrapper. For example:

```yaml
jobs:
  home_optimize:
    steps:
      - backend: home_optimize_step
        agent_wrapper: home_optimizer
        evaluation_options:
          concurrency: 4
          timeout_seconds: 7200
          max_construction_failure_rate: 0.03
        # memory_limit_mib: 0
components:
  agent_wrapper:
    home_optimizer:
      react_config:
        max_iters: 3000
      validation_cycle_time_budget_minutes: 150
      max_evaluations: 10
```

`evaluation_options` also accepts `base_image`, `fail_fast` and
`max_artifact_bytes`. Omit `fail_fast` to retain Home's debug/scored defaults.
Restricted `_allowed_paths` calls fail closed. No real model or Docker benchmark
is required by the isolated workflow tests.
