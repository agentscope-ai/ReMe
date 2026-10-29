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
  memory_construction/
    result.json
    build.log
    reme_workspace.tar.gz
    reme_workspace/
  queries/
    result.json
    <query-id>/answer.log
```

Structured scores in query results and summaries use at most four decimal places.
Raw judge text and reference answers are preserved.

The memory tar is exported **before** queries, retains session Git checkpoints,
and is independent of the query execution state. Replayed runs keep an
independent donor tar and inherited build log without charging construction
tokens again. Answers are persisted before host judging. Per-query results keep
reference/actual answers, scores, compact judge output, reported answer token usage
and failures. `queries/result.json` is the sole authoritative answer record;
reconstructable prompts and repeated payloads are removed using the reviewed
`dev_meta` rules. Construction results similarly compact redundant links, Job
counts, token records and cumulative daily-index snapshots. Summary
scores use the expected-query denominator; infrastructure/judge/artifact errors
make the comparable mean unavailable while retaining the observed-score mean.
The Job returns `success=false` for infrastructure/judge/artifact failures, while
preserving its structured answer and result directory; candidate failures remain
ordinary evaluation observations.
Base jobs preserve structured exception types, HTTP status codes and causes in
`Response.metadata["reme.job_error"]`; the Docker worker carries them into failure
artifacts. Network/provider failures anywhere in the cause chain invalidate the
comparison. Failed jobs without structured error details (including older
candidates) are conservatively treated as infrastructure errors, without parsing
their error text. Successful jobs returning empty answers still count as candidate
failures. The construction failure-rate tolerance applies only to candidate
errors: infrastructure or unclassified failures always reject construction, even
when another failed job was a candidate error.
The manifest retains run identity, selection, mode and execution status. The
summary retains only outcome counts, expected/scored query counts, score means,
low-score cases, question-type statistics and compact case/query mappings.
Candidate answer token summaries average only total tokens per model; detailed
construction costs remain in construction results. Temporary post-query archives
are discarded after extracting logs; separate execution, checksum and raw-answer
JSON copies are not saved. See the
[artifact guide](src/reme_home/runtime_docs/validation-artifact-guide.md) for the
exact fields, rebuild helper and migration from the previous verbose format.
Before comparing runs, verify completed scored execution, infrastructure outcomes,
candidate code and complete case/query coverage.

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
null. `scored_query_count` and `query_status_counts.unscored` are zero, while
`expected_query_count` still records the expected execution count. Debug/scored
mode is recorded in the manifest.
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

With the configured service running, use a second terminal:

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

# Score specified prepared cases without affecting the optimizer's history.
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

`reme_home.evaluation.history.HomeEvaluationHistory` is the dedicated
`home.evaluation_history` Component, automatically bound by `HomeEvaluation`.
It owns successful-result registration, lookup, deduplication, cache rebuilding
and the observations used for the next draw. Steps remain invocation adapters;
they do not own cross-evaluation state.

A successful history entry requires `record_history=true` and a started, completed, scored training-set
run with a full summary and no infrastructure errors, following the existing
`Response.success` contract. Low or zero quality scores remain valid observations.
Classified candidate failures retain the existing response semantics; only their
actual numeric scores train EWMA. A run with any infrastructure error is entirely
excluded, including scores from its healthy cases. Debug, cancelled, fail-fast
and pre-start failures never enter successful history. A full summary is the durable
completion boundary: cancelling the caller during subsequent history bookkeeping
does not relabel that already-completed evaluation. Failed registration discards
only derived caches so they can rebuild from the final source run status.

```python
history = app.context.components["home.evaluation_history"]["default"]
results = history.list_results()  # Typed EvaluationRecord objects, chronological.
result = history.get_result(results[-1].run_key)
history.rebuild()                # Rebuild sampling observations from saved artifacts.
```

The Component owns the workspace lock for these synchronous filesystem methods;
async host code can run them in a worker thread. Successful results are discovered
from each run's manifest and summary; no `history.json` index is written or read.
Legacy index files are ignored and left untouched. Each returned record points to
its run via `run_key`; questions, answers and memory files stay in that result directory.

The separate startup ledger advances only after the first successfully initialized
sandbox, including when a later worker or execution fails. Extra/replacement
workers do not count twice. `home/sampling/state.json` distinguishes started
`round` from `successful_round`. Only successful history advances the probability
clock or changes EWMA; failed runs cannot age cases or alter the next draw's
weights. Started plans live under
`home/sampling/plans/<branch>/<commit-prefix>/<run-id>.json`. Their `round` is the
startup ledger round; `selection_round` records the successful-history snapshot
used for an adaptive draw.

For `record_history=false`, full or explicit evaluation does not access the history
manager. Adaptive selection can still read existing observations, while leaving
them unchanged; its draw is saved locally as `manifest.selection_plan`, without
a startup-ledger round or shared sampling plan. `manifest.options.record_history`
persists the requested setting, and the response's `record_history` indicates
whether the run participates in history (always false for debug). Legacy manifests
without the setting retain the previous history behavior.

Formal runs remain serialized by `home/metadata/validation.lock`; debug bypasses
that lock. Missing caches reconstruct from started plans and successful scored
artifacts. Older sampling caches are rebuilt under the success-only rule.
`history.rebuild()` rewrites only the catalog and sampling cache. Preserve source
manifests, plans and result artifacts when rebuilding; candidate code, datasets
and memory files are never modified.

## Evaluation result inspection

Two read-only Jobs navigate saved evaluations independently of successful
optimization history. They include debug, failed, cancelled, running and standalone
(`record_history=false`) runs. Reads never register results, update sampling,
rebuild summaries or write artifacts.

`home_list_validation_results` accepts optional `branch_name` and `commit_sha`
filters plus `offset=0`, `limit=20` (1–100). Without filters it lists all runs,
newest first, with a stable path tie-breaker. It returns `row_count`, `rows` and
`next_offset`, compact identity/options, saved summary counts and scores, failure
previews and artifact paths. A malformed run is reported as a `read_error` row
without stopping the unfiltered catalog.

`commit_sha` accepts a full SHA or hexadecimal prefix. A prefix must match exactly
one of **all commits reachable from the selected local branch**, including commits
that have never been evaluated. Omitting `branch_name` with a commit filter uses
the current candidate branch. Missing or ambiguous matches fail explicitly; the
response includes the resolved branch and full SHA. A branch-only filter can also
browse historical runs from a deleted branch.

`home_inspect_validation` requires `validation_path` (absolute or relative to
`home/`) and accepts optional `case_id`, `query_id`, `offset=0` and `limit=20`:

| Selection | Returned rows |
| --- | --- |
| Run only | Case summaries, missing scores first, then low scores. |
| Run and case | Query summaries in manifest order, plus independently paged failed construction Jobs. |
| Run, case and query | One query's question, answer, rubric, score and judge/error previews. |

`query_id` requires `case_id`; both must be selected by the run manifest. Cases or
queries without published evidence are explicitly marked `missing`, with null
scores. The tool uses a saved full summary, or an existing partial summary when
the full summary is absent. Missing summaries are reported without reconstruction.
Errors are previewed to 600 characters, and selected query fields to 2,000 each;
truncated fields carry a marker. `rows_source`, JSON `pointer` values, and log and
memory paths identify the original evidence. Failed construction Jobs have their
own `row_count`, `rows`, `next_offset` under `construction.failed_jobs`; their
pagination is independent of the query rows, using the supplied offset/limit.

With the configured service running, use a second terminal:

```bash
reme home_list_validation_results branch_name=candidate commit_sha=abc123 limit=10
reme home_inspect_validation validation_path=evaluations/candidate/abc1234/1
reme home_inspect_validation validation_path=evaluations/candidate/abc1234/1 case_id=case-a query_id=query-a
```

Host callers use the same Jobs through `app.run_job()`. For a case-scoped agent,
inject `validation_path` and `case_id` with `injected_job_kwargs`; the wrapper hides
and enforces these values. Both Jobs reject path escapes and symlinks, and fail
closed for restricted `_allowed_paths` contexts, like the other Home Jobs.

### Memory checkpoint inspection

Read-only Jobs investigate a case's exported memory Git repository. Locate
an evaluation and case with `home_list_validation_results` and
`home_inspect_validation` first. These Jobs never change artifacts, Git refs, the
index, evaluation history or sampling state.

- `home_trace_query_evidence(validation_path, case_id, query_id,
  evidence_session_ids=null)` joins a selected query to its source sessions and
  memory checkpoints without calling a model. Dataset evidence annotations take
  precedence over optional inferred session IDs, including an empty annotation.
  Returns `query` (question, expected/actual answer, score, error and
  `result_missing`), `evidence_source` (`independent_annotation`,
  `diagnostician_inference`, or `unavailable`), `evidence_is_certain`,
  `memory_status`, `sessions` (ID, checkpoint and changed files), and `paths` to
  the normalized case, memory repository, query results and query artifacts.
  A null checkpoint means no matching `session: <id>` commit was found; missing
  results keep scores null. Answer/question previews are limited to 2,000
  characters and errors to 600. At most 50 sessions and 50 changed files per
  session are returned, with omitted counts when needed. Session text, diffs,
  later history and full answer/judge results remain at the returned paths or
  in the inspection/comparison Jobs below. Original transcripts and derived
  memories can both be searchable; these coordinates alone do not prove a cause.
- `home_inspect_memory_history(validation_path, case_id, path=null, offset=0,
  limit=20)` returns `status`, `workspace_exists`, `has_git_history`, `head`,
  `repository`, and newest-first checkpoint `rows` with `row_count` and
  `next_offset`. Limits are 1–100. Each row includes the full commit SHA, parents,
  timestamp, subject (up to 2,000 characters) and a `session_id` when the subject
  follows `session: <id>`. `path` follows a literal workspace-relative file or
  directory. Missing exports, absent Git history and empty repositories have
  distinct statuses; corrupt repositories fail explicitly.
- `home_compare_memory(validation_path, case_id, base_ref=null,
  target_ref="HEAD", target_validation_path=null, path=null)` compares two memory
  commits. Within one evaluation, `base_ref` is required. For another evaluation,
  supply `target_validation_path`; both sides then default to `HEAD`. The case
  must be selected in both manifests. Returns resolved commits, candidate commit
  identities, manifest fingerprints, added/removed/modified paths, content-tree
  hashes, a unified `diff`, and `truncated`.

References accept HEAD-relative expressions and hexadecimal SHA prefixes unique
among commits reachable from that memory repository's HEAD. Resolved expressions
must also be HEAD-reachable. This is independent of candidate-code branch history.
Comparisons read committed regular files from Git objects, ignoring dirty or
untracked checkout contents. File mode changes alone are not content differences.
Binary changes are identified without printing their bytes. Each snapshot is
limited to 2,000 files, 20 MB total and 5 MB per file; diff output is limited to
100,000 characters. Narrow `path` if a limit is reached. Tree hashes describe the
selected path scope. Linked Git storage, alternates, shallow/partial histories and
symbolic links in Git metadata are rejected.

After starting the service with the Home configuration, call from another terminal:

```bash
reme home_inspect_memory_history validation_path=evaluations/candidate/abcdef0/1 case_id=case1
reme home_trace_query_evidence validation_path=evaluations/candidate/abcdef0/1 case_id=case1 query_id=query1
reme home_compare_memory validation_path=evaluations/candidate/abcdef0/1 case_id=case1 base_ref=HEAD~1
reme home_compare_memory validation_path=evaluations/candidate/abcdef0/1 target_validation_path=evaluations/candidate/1234567/1 case_id=case1
```

`validation_path` is absolute or relative to `home/`; for example the relative
form is `evaluations/candidate/abcdef0/1`. For agent tools, host `injected_job_kwargs` can pin
`validation_path` and `case_id`; the wrapper hides injected fields and rejects
model overrides. Like the other Home inspection Jobs, these Jobs reject
`_allowed_paths` restrictions rather than bypassing them. Use these tools directly
or through the single-case investigator below.

## Single-case investigation

`home_case_study(validation_path, case_id, focus=...)` runs one fresh diagnostic
agent. It follows dev_meta's case-study workflow: trace judging and answer
failures backward through retrieval, memory construction, source sessions and
the evaluated implementation. The text report cites evidence paths/lines or
JSON fields and distinguishes supported causes, hypotheses and missing evidence;
it does not prescribe fixes. There is no batch fan-out or parallel case-study
dispatcher. Each invocation has its own agent state and timer.

The `home_case_study` component uses `home_case_study_agent`, a thin subclass of
the bounded wrapper, defaulting to
the host's existing `as_llm: judge` model, 60 iterations and a 30-minute advisory
budget, matching dev_meta's default budgets. Configure a separate host model by
overriding `components.agent_wrapper.home_case_study.as_llm`. ReMe owns model
startup, token accounting and saved session transcripts. The response answer is
the report; metadata includes the validation, case, evaluated commit, session ID
and `usage`. No case scratch directory or intermediate analysis files are created.

The toolkit includes validation inspection, compact query-evidence tracing,
memory history/comparison, and the three committed-code navigation Jobs.
Validation/case selectors and the evaluated code revision are hidden and enforced
by the host; memory comparison stays within the selected evaluation. General
Read/Grep/Glob, Write/Edit, validation and code-mutation tools are not exposed.

`AutoBash` and `python_execute` share a required OS sandbox. Commands have no
static whitelist (`is_read_only=false`); filesystem permissions enforce access.
Read-only roots are the selected case artifact directory, normalized source-case
JSON, validation manifest, all of `home/code/`, and `home/docs/`. No host writable
roots are granted. System/interpreter runtime paths and disposable backend
temporary storage remain available. Direct code reads can see current worktree
changes; use the revision-pinned Jobs to establish the evaluated behavior.
`python_execute` accepts either `code` or an existing `file`, optional file-only
`args`, and a `timeout` in seconds (default 60, maximum 600). It uses `python -I -B`
and returns returncode/stdout/stderr/timed_out. Both executors reuse bounded output,
timeouts, cancellation cleanup and network isolation from the platform backend.

Linux requires Bubblewrap and, with the default 4096-MiB limit, a writable
delegated cgroup v2. macOS requires Seatbelt (`sandbox-exec`); memory limits are
not enforced there. The Job probes sandbox startup before calling the model and
never falls back to an unrestricted executor. The command whitelist parser is
not used by this agent. Sandbox policy, output-report token guidance and executor
memory policy are host Step configuration, not model arguments:

```yaml
jobs:
  home_case_study:
    steps:
      - backend: home_case_study_step
        agent_wrapper: home_case_study
        output_token_limit: 2500
        # memory_limit_mib: 0  # Explicitly disable Linux cgroup memory limiting if needed.
```

With the configured service running, call from another terminal:

```bash
reme home_case_study validation_path=evaluations/candidate/abcdef0/1 case_id=case1 'focus=Trace low-scoring queries'
```

Calls with `_allowed_paths` restrictions fail closed, as other Home artifact
Jobs do. Structured inspection tools run in the host with their existing
containment checks; only Bash/Python subprocesses run inside the OS sandbox.
