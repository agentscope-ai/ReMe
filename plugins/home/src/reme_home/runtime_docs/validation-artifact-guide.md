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

## Summary fields

The run summary contains only:

- `case_count`, `case_status_counts`, `construction_status_counts`, and
  `failure_stage_case_counts` (construction/query; tolerated construction failures
  still count as construction errors).
- `expected_query_count`, `scored_query_count`, and `query_status_counts`
  (completed, candidate failure, infrastructure error, unscored).
- `mean_query_score`, `observed_mean_query_score`, `low_score_cases`,
  `question_type_statistics`, and `cases`.

Each `cases[case_id]` contains construction `status`, `success_job_count` and
`fail_job_count`; `mean_score`; `average_query_total_token_usage`; and a `queries`
map. Each query entry contains only `question_type`, `score` and `status`.
`stage`, `construction_error_count` and `query_error_count` appear only when
applicable. There is no redundant case status or per-case coverage accounting.
Questions, answers, errors, Job details and construction token usage stay in
phase artifacts, outside the summary.

Token summaries contain only average candidate answer **total** tokens per model,
divided by all returned queries, including failures and queries without reported
usage. Zero and unknown token totals are omitted. Judge/construction costs and
input/output token breakdowns are not aggregated. Question-type statistics contain
query count, average of available scores (rounded to three decimals), and the same
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
