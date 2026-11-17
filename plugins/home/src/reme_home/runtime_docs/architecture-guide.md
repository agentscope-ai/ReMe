# ReMe Candidate Development Guide

Use this guide when improving the code prepared by Home. The initial candidate is
a starting implementation; its internal design may change while the fixed callers
and interfaces continue to work.

## Workspace and modification boundary

All paths below are relative to the outer ReMe workspace:

- `home/domain_spec.yaml` records the resolved experiment configuration. Read
  `candidate.paths` for the editable paths, `runtime.plugins` for the selected
  plugins, and `runtime.config` for the application configuration.
- `home/code/repo/` is the candidate Git repository. Paths in `candidate.paths`
  are relative to this repository. It contains complete selected plugin projects,
  including packaging and configuration, even if only their `src/` is editable.
- `home/code/runtime/` is the fixed ReMe source. Evaluation assembles an isolated
  tree from this runtime and the candidate commit. Plugins are installed from the
  candidate; the fixed runtime has no duplicate plugin projects.
- `home/datasets/` contains prepared training and debug inputs. Treat them as
  fixed evaluation inputs. Do not encode benchmark answers or case-specific
  shortcuts in candidate code, prompts, or resources.
- `home/metadata/preparation.json` records source and initial candidate commits,
  fingerprints, and the installation plan.

Keep edits within `candidate.paths`. Inspect the fixed runtime, prepared inputs,
and evaluation artifacts as needed, but do not modify them to improve a score.
Preserve the benchmark-facing Jobs, especially `auto_memory` and
`agentic_answer`, their accepted inputs, and their response contracts. Inspect
the actual callers and configured Steps before changing an interface. Use
dependencies available in the evaluation environment.

## ReMe architecture

ReMe is local-first and file-native. User-owned workspace files are durable
source data; indexes, catalogs, graphs, and caches must remain rebuildable.
Keep candidate workspace access within ReMe's path safety boundary.

- A Component is application-scoped. Use it for shared stores, indexes, clients,
  locks, and resources with startup/shutdown requirements.
- A Job orchestrates Steps. Each invocation has a fresh RuntimeContext, a shared
  Response, and fresh Step instances. `RuntimeContext.data` carries inputs and
  intermediate values between Steps in that invocation.
- Step fields and `self.kwargs` are invocation-scoped. They do not persist across
  Job calls. Use workspace files for durable state and namespaced, synchronized
  `ApplicationContext.metadata` for application-lifetime state.
- A typed `Ref` resolves dependencies from Step kwargs, then RuntimeContext, then
  application Components. Reuse existing stores and indexes through this wiring.
- Plugin `plugin.yaml` exposes importable `module:Class` backends and configures
  Jobs, Steps, and Components under `application_defaults`. Keep registration,
  configuration, prompt variables, and implementation consistent.

Inspect the bundled base classes and nearby plugin code for current signatures
and lifecycle behavior. Candidate prompts, helper modules, memory formats, and
retrieval strategies can change within the configured editable scope.

## Evaluating a candidate

Commit every candidate change before calling `home_evaluate`. Modified tracked
files, staged changes, and untracked files in `home/code/repo/` reject evaluation.
The Job uses its current HEAD and records that commit; it accepts no `code_ref`.
Run the initial full evaluation before changing the initial candidate.

The host program can inject `case_ids` and a `query_ids` mapping into the Job.
Omitted case IDs select all prepared cases; omitted query selections use every
query in each selected case. Debug runs use `debug=true` and
check execution only, without judging answer correctness. They select prepared
debug cases and default to `fail_fast=true`; scored runs default to false. An
explicit `fail_fast` overrides the mode default. Debug cannot reuse construction.

Setting `reuse_construction_from` restores a compatible completed run's pre-query
memory tar, skips construction, and answers and scores again using current code.
The caller must establish construction equivalence. Every selected case needs a
valid donor tar; missing cases cause a preflight error. There is no `eval_only`
flag. Host-installed benchmark judge plugins score answers outside Docker and
outside the candidate repository.

See `validation-artifact-guide.md` for saved evidence and score interpretation.

## Candidate versions

Use the Home code Jobs to list, create, switch, merge and commit candidates in
`home/code/repo`. `init` is the protected preparation baseline. Branch operations
require clean code; merges stay uncommitted for inspection. Commit all intended
changes with `home_commit_candidate` before debug or scored evaluation. Mutations
are limited to prepared `candidate_paths`; keep repository metadata and runtime
outside the optimizer's write permissions.

`home_list_code_tree`, `home_read_code_file` and `home_search_code` inspect raw
committed files. Their `revision` defaults to current HEAD and returns a resolved
SHA. For diagnosis or version comparisons, the host should inject a fixed SHA
through `injected_job_kwargs` for each agent/Job alias, so subsequent checkout
changes cannot affect the evidence. See the Home plugin README for inputs and
output limits.
