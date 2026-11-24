# Home Evaluation Guide

## Evaluation inputs

Start from the host-provided baseline. When targeting a regression, take case
and query IDs from the earlier run. After evaluation, check `selection.notices`
and the manifest to verify that the intended questions actually ran.

Reused memory still produces new answers and scores. When comparing runs, check
which cases rebuilt memory so you can distinguish construction changes from
answer-side changes.

## Evaluation outputs

- Confirm `commit_sha` and check `case_status_counts` and `query_status_counts`
  even when the run is `completed`. Read the reason for a rejection before
  retrying. Diagnose candidate failures and low scores before changing the code.
- The formal `mean_query_score` averages over expected questions, with
  candidate failures contributing zero. Failed construction can leave questions
  unassessed. Infrastructure failures can leave the comparable mean null even
  when `observed_mean_query_score` is available.
  The observed mean covers only scored questions. Check `scored_query_count`
  against `expected_query_count` for coverage.
- Compare completed, infrastructure-free formal runs. When coverage differs,
  limit conclusions from `home_compare_validations` to shared case/query pairs.

## Inspecting a result

Pass `home_evaluate`'s `answer.result_dir` as `validation_path` to inspection
tools. After narrowing the run with `home_inspect_validation`, carry the same
case and query IDs into `home_trace_query_evidence` when investigating a cause.

Within the case directory, `queries/result.json` contains the question, available
reference answer and rubric (`groundtruth`), actual answer, judgment, score, and
errors. Use `queries/<query-id>/answer.log` for retrieval and answer execution, and
`memory_construction/build.log` for construction. For failed or interrupted runs,
inspect `failure.json` or `partial_summary.json` where available.

## Memory evidence tools

To see what an evidence session changed, take its `sessions[].checkpoint` from
`home_trace_query_evidence` as `target_ref`. Find its parent in
`home_inspect_memory_history` and pass that SHA as `base_ref` to
`home_compare_memory` when the parent is available.

Inspect partial construction even when answers succeeded.

Check `evidence_source` and `evidence_is_certain` before attributing a failure.
Use inferred or uncertain sessions as leads. Original transcripts and derived
memory may both be searchable. Check retrieval and answer logs to establish
what reached the answerer. Check the run's failure records before drawing
conclusions from an incomplete export.

## Focused case studies

During `home_diagnose`, group affected cases by evaluation before batching them
with `home_case_study`. A case ID can recur across runs. If a batch fails, check
each case's `status` and keep the completed reports.
