# Home Evaluation Guide

Use evaluation to check committed candidate code, inspect failures, and decide what to change next. The host selects cases, manages memory construction or reuse, runs questions, and judges formal answers. Use returned selections and evidence to understand what was actually tested.

## Evaluation inputs

Commit candidate changes before calling `home_evaluate`. In the optimizer workflow, run `debug=true` to check execution, then `debug=false` for scored formal evaluation. Debug does not measure answer quality.

| Input | Meaning |
| --- | --- |
| `debug` | Defaults to false. True runs prepared debug cases without judging. |
| `sampling` | Enables host-managed case/question selection. False evaluates the full dataset unless the host supplies an exact selection. The optimizer supplies its evaluation mode and limits. |
| `target_queries` | Optional priorities for sampled formal evaluation, keyed by case ID. `[]` prioritizes the case with host-selected questions; `null` prioritizes all its questions; a list prioritizes those query IDs. Debug ignores targets. |
| `record_history` | Defaults to true. False saves results without contributing to optimization history. Debug always skips history. |
| `reuse_construction_from` | Explicit prior result directory inside `home/evaluations` for callers that manage replay. The optimizer manages reuse automatically. |

Targets express priorities within host limits; they do not guarantee an exact test set. Other cases/questions may be included. Unknown IDs are reported and ignored; a valid case with no valid requested questions remains a case priority. Read the returned `selection` for actual counts, invalid IDs and trimming notices. Question identity is the pair `(case_id, query_id)`.

Leave execution settings and budgets supplied by the host unchanged. Tool parameter descriptions define available overrides. Internal exact-selection inputs are not model-selected tool arguments.

Memory may be rebuilt or restored from compatible, construction-equivalent prior results. Reuse still executes questions and judges answers with the current candidate; it does not reuse old answers or scores. A run can mix rebuilt and reused cases. The host handles selection and reuse decisions automatically.

## Evaluation outputs

A completed call returns:

| Field | How to use it |
| --- | --- |
| `result_dir` | Pass this directory to inspection tools as `validation_path`. |
| `commit_sha`, `evaluated_branch` | Identify the evaluated candidate. |
| `status` | Check completion. Completion alone does not establish answer quality or freedom from infrastructure errors. |
| `execution_mode` | `full_validation` rebuilds memory, `query_only_replay` reuses it, and `mixed_validation` does both. |
| `selection` | Actual case/question counts and target-selection feedback, when present. |
| `case_status_counts`, `query_status_counts` | Distinguish completed execution, candidate failures and infrastructure errors. |
| `mean_query_score` | Overall comparable score; may be null when infrastructure or judging failed. |
| `observed_mean_query_score` | Average of available scores only; may represent incomplete evidence. |
| `expected_query_count`, `scored_query_count` | Check execution and scoring coverage. |
| `low_score_cases`, `question_type_statistics` | Locate weak cases and question types. |
| `record_history` | Whether this run participates in optimization history. |

Additional fields may point to comparisons, review evidence or the remaining validation budget. Follow returned paths when needed. Rejected evaluations report `status: rejected` with a reason; they are not new scored evidence. Unchanged scored candidates require a code change before another evaluation. Infrastructure failures may retry the same commit using the host's saved plan.

Candidate failures count as zero in the expected-query score denominator; construction failures leave questions unassessed. Infrastructure failures can make the comparable mean unavailable even when some answers have scores. Debug scores are null. Compare completed formal runs only after checking infrastructure outcomes, coverage and candidate identity. Low scores are useful evidence.

## Inspecting a result

1. Find runs with `home_list_validation_results`, or use `result_dir` from evaluation.
2. Call `home_inspect_validation(validation_path)` for case summaries. Add `case_id` for question summaries, then `query_id` for answer, reference and judge previews.
3. Investigate a weak question with `home_trace_query_evidence(validation_path, case_id, query_id)`.
4. Inspect memory checkpoints or request a focused case study when answers and logs do not explain the failure.

Listings and inspection use `offset` and `limit` (default 20, maximum 100). Follow `next_offset` for more rows and returned source paths for full evidence; previews are bounded. Listings include failed, debug, running and untracked runs. Catalog filters include `branch_name` and `commit_sha`.

Each run is saved under `home/evaluations/<branch>/<commit-prefix>/<run-id>/` in the outer ReMe workspace. Paths below are relative to that run:

| Path | Evidence |
| --- | --- |
| `manifest.json` | Evaluated commit, actual case/question selection, mode, status and memory sources. |
| `summary.json` | Completed execution summary, including per-case and per-question scores. |
| `partial_summary.json`, `failure.json` | Available evidence and diagnostics after interruption or failure. |
| `cases/<case-id>/queries/result.json` | Authoritative question, reference answer, actual answer, judge result, score and errors. Answers remain available when judging fails. |
| `cases/<case-id>/queries/<query-id>/answer.log` | Retrieval and answer execution log; host judging happens separately. |
| `cases/<case-id>/memory_construction/result.json`, `build.log` | Construction results, failures and reported costs. |
| `cases/<case-id>/memory_construction/reme_workspace/` | Pre-query memory workspace and exported Git checkpoints. |

Case/query failure records and diagnostic archives may also be present. Missing exports can reflect incomplete execution; inspect failure records before drawing conclusions. A null score alone does not identify a failure. Summaries are compact; use source artifacts for full answers, errors, Job details and token usage.

## Memory evidence tools

`home_trace_query_evidence` returns question/answer/score previews, evidence session IDs, matching memory checkpoints, changed files and paths to complete evidence. Dataset annotations take precedence over optional inferred `evidence_session_ids`. Respect `evidence_source` and `evidence_is_certain`: inferred sessions are leads, not established causes. Missing evidence remains explicit. Original transcripts and derived memory may both be searchable; absence from a memory summary alone does not establish a retrieval failure.

`home_inspect_memory_history(validation_path, case_id)` pages through exported memory checkpoints. Optional `path` narrows inspection to a workspace-relative file or directory. Missing workspaces and histories are reported explicitly.

`home_compare_memory` compares committed memory using `base_ref` and optional `target_ref` (default `HEAD`). Add `target_validation_path` to compare the same case across evaluations; both references then default to `HEAD`. Use returned history references, narrow `path` for large diffs, and check candidate identities before interpreting cross-evaluation differences. These tools are read-only.

## Focused case studies

Call `home_case_study(evaluation_path, case_ids, focus=...)` for up to eight unique cases sharing one investigation focus. It returns a `results` entry per case with a report/session/usage or error. Check each entry's status: only `completed` succeeds, and a failed case does not cancel the rest.

The diagnostic agent has read-only access to selected evidence and candidate code. The working tree may differ from the evaluated commit; use pinned code tools for version-sensitive claims. Code-tool paths are candidate-repository-relative; read artifact paths with AutoBash or Python. Keep read errors visible: a denied read is not evidence that content is missing. Case studies do not edit code or run new evaluations.
