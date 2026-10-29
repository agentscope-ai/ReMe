# Home Evaluation Artifact Guide

`home_evaluate` stores each run at
`home/evaluations/<branch>/<commit-prefix>/<run-id>/` inside the outer ReMe
workspace. Commit prefixes start at seven characters and extend on collision.
Run IDs increase within that commit directory; prior results are preserved.
Result compaction and summary fields follow the reviewed `dev_meta` format.

## Saved files

Paths in this table are relative to one run directory:

| Path | Meaning |
| --- | --- |
| `manifest.json` | Candidate commit, selected case/query IDs, dataset, execution mode, input/configuration fingerprints, host judge identity, options, timestamps and run status. Retained for reproducibility and rebuilding summaries; identity is not repeated in the summary. |
| `summary.json` | Rebuildable evidence from a completed execution. Its fields are described below. |
| `partial_summary.json` | Available evidence after cancellation or failure; not a completed baseline. |
| `failure.json` | Run-level failure details, when present. |
| `cases/<case-id>/memory_construction/result.json` | Ordered construction Job results with success, errors, metadata, answers, reported token usage and inner Job call counts. Replays record their source and snapshot checksum without charging construction again. |
| `cases/<case-id>/memory_construction/build.log` | Construction log. Replays inherit the donor log. |
| `cases/<case-id>/memory_construction/reme_workspace.tar.gz` | Checksum-verified workspace snapshot exported before answering; the donor for construction reuse. |
| `cases/<case-id>/memory_construction/reme_workspace/` | Extracted construction workspace, including memory files and session Git checkpoints when exported. |
| `cases/<case-id>/queries/result.json` | The sole authoritative query record: question, reference and actual answer, question type, score, compact answer/judge responses and failures. All returned answers are frozen here before extraction and host judging; scores are published progressively. |
| `cases/<case-id>/queries/<query-id>/answer.log` | Query execution log, when available. |
| Case/query `failure.json` | Failure stage and diagnostics, when present. |
| `cases/<case-id>/full.tar.gz` | Best-effort sandbox diagnostics for execution failures, when available. |

Successful exports retain query logs, then discard the temporary post-query
`evaluation.tar.gz`. There is no separate `execution.json`, `artifacts.json`,
`queries/answers.json` or per-query `result.json`. A failed extraction can leave
its archive for diagnosis. Exported files depend on execution progress; inspect
failure records before interpreting an absent workspace or log.

Query compaction removes reconstructable `sys_prompt` values, duplicate nested
answers/questions/errors, the enclosing Job's one call, judge token usage,
duplicate parsed judgements when raw judgements exist, and repeated judge inputs.
Distinct errors, raw judgements, rubric assessments and reported answer token
metrics remain. Unreported token backends are omitted.

Construction compaction removes checkpoint descriptors already available in Git,
one enclosing Job call, unreported token backends, and transcript links identified
by a sibling session ID. Recognized cumulative daily-index `notes` snapshots become
`notes_count`; unrelated plugin payloads remain. All-zero construction token
records become `0`. Detailed construction costs stay in this case artifact.
Small scalar containers are written on one line, following `dev_meta` formatting.

Structured scores in query results and summaries use at most four decimal places.
Reference answers, raw judge strings and non-score metrics retain their original values.

New run manifests omit the unused schema version, duplicate code and judge fingerprints,
`options.debug` (use `dataset_kind`/`mode`), and duplicate sampling policy (use the plan).
The remaining `fingerprints.dataset` and `fingerprints.construction_config` guard memory
reuse compatibility; `judge.fingerprint` identifies the trusted scoring implementation.
Legacy manifests remain readable. Workspace markers, preparation recovery journals and
sampling state retain versions used for validation, rollback and migration.

## Summary fields

The run summary contains only:

- `case_count`, `case_status_counts`, `construction_status_counts`, and
  `failure_stage_case_counts` (construction/query; tolerated construction failures
  still count as construction errors).
- `scoring_enabled`, `expected_query_count`, `scored_query_count`, and `query_status_counts`
  (completed, candidate failure, infrastructure error, unscored).
- `mean_query_score`, `observed_mean_query_score`, `low_score_cases`,
  `question_type_statistics`, and `cases`.

Each `cases[case_id]` contains construction `status`, `success_job_count` and
`fail_job_count`; `mean_score`; `average_query_total_token_usage`; and a `queries`
map. Each query entry contains only `question_type` and `score`. Failed queries
also appear in `query_errors`, keyed by query ID, with `failure_kind`
(`candidate_failure` or `infra_error`), `stage` (null if unavailable), and an
error excerpt limited to 500 characters plus a truncation marker. Full errors
remain in the phase artifacts. A null score alone does not identify a failure:
successful queries also remain unscored when `scoring_enabled` is false.
`stage`, `construction_error_count` and `query_error_count` appear only when
applicable. There is no redundant case status or per-case coverage accounting.
Questions, answers, full errors, Job details and construction token usage stay in
phase artifacts, outside the summary.

Token summaries contain only average candidate answer **total** tokens per model,
divided by all returned queries, including failures and queries without reported
usage. Zero and unknown token totals are omitted. Judge/construction costs and
input/output token breakdowns are not aggregated. Question-type statistics contain
query count, average of available scores (rounded to four decimals), and the same
answer token averages across cases. Missing/empty types form `unknown`.

Candidate failures count as zero in the expected-query score denominator.
Construction failures leave queries unassessed rather than inventing query
failures. Infrastructure, artifact, cleanup or judge errors make the comparable
run mean unavailable; `observed_mean_query_score` averages available scores only.
`low_score_cases` contains `{case_id: mean_score}` for cases with at least one
scored query below `1`, sorted by mean then case ID; unavailable case means are
omitted.

Run status and debug/scored mode are in the manifest. Debug runs skip judging;
run and case means are null, `scored_query_count` and the unscored count are zero,
while `expected_query_count` records execution coverage. They cannot measure
answer quality. Only compare completed scored runs after checking infrastructure
outcomes, expected coverage and the candidate selection.

To rebuild the appropriate summary from a run's manifest and phase results:

```python
from pathlib import Path
from reme_home.evaluation.artifacts import rebuild_summary

summary = rebuild_summary(Path("/absolute/path/to/run"))
```

New results omit the former `schema_version`, `status`, `execution_complete`,
`execution_mode`, `expected_case_count`, `expected_scored_query_count`,
`scoring_required`, `scoring_complete`, `valid_for_comparison`, `scored_count`,
`unscored_count`, and top-level `query_error_count` from summaries. Use manifest
identity/status/mode and the retained counts instead. The Job returns the run
summary without `cases`, plus `result_dir`, `commit_sha`, `execution_mode` and
execution `status`. Existing result directories are not rewritten automatically.

## Investigating a result

1. Read the manifest for candidate, selection, mode, judge and construction source.
2. Read the summary, then an affected case's `queries/result.json`. Compare the
   answer, reference and judge assessment. Frozen answers survive judge failures.
3. Read `answer.log` for retrieval/answer execution; the host judge is outside it.
4. Read construction `result.json` and `build.log` for Job failures and costs.
5. Inspect the pre-query workspace and its Git history with read-only `git log`,
   `git show` and `git diff` to trace session memory checkpoints. Memory/session
   directory names depend on runtime configuration.

Failure records preserve structured exception types, HTTP status codes and cause
chains from `Response.metadata["reme.job_error"]`. Network/provider failures in
any construction Job or exception cause invalidate comparison. Failed Jobs without
structured exception details are treated as infrastructure errors. Construction
failure-rate tolerance applies only to classified candidate errors.
Construction reuse answers and judges again with the current candidate; it does
not reuse donor answers or scores.

## Evaluation history and adaptive sampling evidence

`home_evaluate sampling=true` enables adaptive selection; false preserves full or
explicit evaluation. Debug always bypasses history. `sampling_policy` is the
host policy input, distinct from the boolean toggle. The effective policy is
captured in the started plan, or `manifest.selection_plan` for untracked adaptive runs.

`record_history=true` (the default) includes a successful full baseline even when
`sampling=false`. `record_history=false` runs the same scoring and artifact pipeline
without registering results or advancing any optimization/sampling counters. The
choice is stored in `manifest.options.record_history` and respected by history
registration and rebuilding; a missing legacy setting means true. Debug always
skips history. The response reports effective participation as `record_history`.
An untracked adaptive run may read existing observations and saves its draw locally
in `manifest.selection_plan`, without a shared started plan or ledger round.

`home.evaluation_history` discovers successful results directly from run manifests
and summaries. No `history.json` index is generated or read; legacy copies are ignored.
Entries require completed scored training runs
with `success=true` (or equivalent legacy summary counts) and no infrastructure
errors. Low scores remain useful. Failed/cancelled/debug/partial runs are excluded;
a mixed run with an infrastructure failure contributes no quality observations.
Source manifests, summaries and query results remain authoritative.

`sandbox_started_at` marks the first successfully initialized sandbox. The
history-enabled manifest's `sampling.plan_path` points to its policy, seed, weights, selection
and started round under `home/sampling/plans/`. Pre-start failures have no marker
or plan, and additional workers do not create extra rounds. The derived sampling
cache separates started `round` and `successful_round`: only successful history
advances staleness/recheck timing and trains EWMA. Adaptive plans preserve their
probability clock in `selection_round`. Actual numeric scores alone count as
quality observations; missing/unjudged answers never become zero scores.

Missing sampling caches and `HomeEvaluationHistory.rebuild()` reconstruct
sampling observations from saved evidence. Result listings always scan the source artifacts. Rebuilding never modifies datasets,
candidate code or memory files. See the Home README for configuration, host
per-call policy overrides and enforced agent injection examples.

## Read-only evaluation inspection

Use `home_list_validation_results` to find runs and pass a returned
`validation_path` to `home_inspect_validation`. The catalog includes failed,
debug, running and untracked runs, not only successful optimization history.
Both Jobs page with `offset` and `limit` (default 20, maximum 100), returning
`row_count`, `rows`, `next_offset`. Catalog filters are `branch_name` and
`commit_sha`; a hexadecimal commit prefix must uniquely match among all commits
reachable from the specified branch, or the current branch when omitted.

Inspection without `case_id` returns case summaries; add `case_id` for query
summaries and `query_id` for bounded answer/judge fields. Each row's JSON pointer
is relative to `rows_source`. Failed construction Jobs have separate pagination
and pointers relative to `construction.source`. Query fields are limited to
2,000-character previews and errors to 600 characters. Follow returned source
paths for complete evidence. Missing files/results remain explicit, with null
scores, and partial summaries are used when full summaries are absent. Reads do
not rebuild summaries or modify evaluation history, sampling state or artifacts.

## Memory checkpoint tools

Start a failed-query investigation with
`home_trace_query_evidence(validation_path, case_id, query_id)`.
It returns compact question/answer/score/error previews, source evidence session
IDs, matching Git checkpoints, changed files and paths to complete evidence. It
does not call a model or include transcripts, diffs or judge-result bodies.
Dataset `evidence_session_ids` annotations take precedence; only when absent may
you supply inferred `evidence_session_ids`. Respect `evidence_source` and
`evidence_is_certain`: inferred or uncertain sessions are leads, not established
causes. No annotation or supplied IDs yields `evidence_source: unavailable`.
Missing query results are explicit and scores remain null. `memory_status`
distinguishes unavailable histories; a null session checkpoint means no matching
commit was found. Question/answer previews have a 2,000-character limit, errors
600, and session/changed-file lists 50 entries with omitted counts. Follow
`paths.source_case`, `paths.query_result` and `paths.query_artifacts` for details;
use the tools below for history and focused diffs. Both original transcripts and
derived memory may be searchable, so absence from a summary alone proves no
retrieval failure.

After locating one case, call `home_inspect_memory_history` with its
`validation_path` and `case_id` to page through exported memory checkpoints.
Optional `path` follows a literal memory-workspace-relative file or directory.
The response distinguishes a missing workspace, absent history and an empty
repository; invalid history fails explicitly. Session IDs are extracted only
from the established `session: <id>` commit-subject convention.

Use `home_compare_memory` with `base_ref` and optional `target_ref` (default
`HEAD`) to inspect how memory changed within that evaluation. Add
`target_validation_path` to compare the same case across evaluations; both
references then default to `HEAD`. References must resolve to commits reachable
from each memory HEAD, and hexadecimal prefixes must be unique within that
reachable history. Compare candidate identities and fingerprints before drawing
cross-evaluation conclusions.

Diffs describe committed tracked regular-file contents rather than dirty checkout
files, untracked files or mode-only changes. Binary changes are identified without
printing bytes. Narrow `path` when the 2,000-file, 20-MB snapshot, 5-MB file or
100,000-character diff limit is reached. Hashes cover the selected path scope.
These tools neither reconstruct missing exports nor modify Git/evaluation/history
artifacts. Raw logs and source JSON remain available through the existing result
inspection paths.

## Single-case diagnostic agent

Call `home_case_study(validation_path, case_id, focus=...)` to investigate exactly
one case. It returns a text report with evidence citations and uncertainty, using
fresh agent state, bounded iterations and advisory time reminders. There is no
parallel case dispatcher. Scope and evaluated code revision are host-injected
into the structured tools. AutoBash and Python share a mandatory read-only file
sandbox covering the selected case artifacts, source JSON, validation manifest,
home/docs and all home/code, plus required system runtimes. Neither executor has
host-writable roots or a case scratch directory. They allow analysis commands
and Python rather than a static command whitelist. The directly readable code
worktree can differ from the evaluated commit; use pinned code tools to verify
version-sensitive claims. No write/edit, validation or code-mutation tools are
exposed. Sandbox startup failure prevents model execution; it never falls back
to an unrestricted process. See the Home README for model, budget and platform
sandbox configuration.
