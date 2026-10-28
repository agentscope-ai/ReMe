# Home Evaluation Artifact Guide

`home_evaluate` stores each run at
`home/evaluations/<branch>/<commit-prefix>/<run-id>/` inside the outer ReMe
workspace. Commit prefixes start at seven characters and extend on collision.
Run IDs increase within that commit directory; prior results are preserved.

## Saved files

Paths in this table are relative to one run directory:

| Path | Meaning |
| --- | --- |
| `manifest.json` | Candidate commit, case/query selection, execution mode, input/configuration fingerprints, host judge identity, and run status. |
| `summary.json` | Completed run summary: aggregate scores, expected/scored counts, case/query outcomes, low-score cases, and comparison validity. |
| `partial_summary.json` | Partial results after an interrupted or failed run; not a completed baseline. |
| `failure.json` | Run-level failure details, when present. |
| `cases/<case-id>/execution.json` | Docker pool execution outcome, including execution/cleanup failures. |
| `cases/<case-id>/artifacts.json` | Checksums of exported archives. |
| `cases/<case-id>/memory_construction/result.json` | Construction outcome and ordered Job results, including available errors, metadata, and token usage. |
| `cases/<case-id>/memory_construction/build.log` | Construction log. Replayed runs inherit the donor log. |
| `cases/<case-id>/memory_construction/reme_workspace.tar.gz` | Workspace snapshot exported before answering queries; the donor for construction reuse. |
| `cases/<case-id>/memory_construction/reme_workspace/` | Extracted construction workspace, including memory files and session Git checkpoints when exported. |
| `cases/<case-id>/evaluation.tar.gz` | Exported post-query evaluation artifacts. |
| `cases/<case-id>/queries/answers.json` | Raw answers saved by the host before judging and artifact extraction. |
| `cases/<case-id>/queries/result.json` | Per-query question, reference and actual answer, score, answer response, judge response, and failure information. |
| `cases/<case-id>/queries/<query-id>/answer.log` | Query execution log, when available. |
| Case/query `failure.json` | Failure stage and diagnostics, when present. |

Exported files depend on how far execution progressed. Infrastructure failures
and timeouts may prevent logs or full workspace archives from being recovered.
Inspect failure records before assuming that an absent artifact means the
candidate produced no memory or answer.

## Investigating a result

1. Read the manifest to confirm candidate commit, selected cases/queries, judge,
   and whether construction was built or reused.
2. Read the summary, then inspect `queries/result.json` for an affected case.
   Compare the actual answer with the reference and judge response.
3. Use `answer.log` to trace retrieval and answer generation. Raw
   `queries/answers.json` remains available if host judging failed.
4. Inspect `memory_construction/result.json` and `build.log` for construction
   failures or unexpected Job behavior.
5. Inspect the extracted pre-query workspace. Its configured memory/session
   directories and Git history show memory changes at construction checkpoints.
   Use read-only Git commands such as `git log`, `git show`, and `git diff` in
   that exported repository. Directory names depend on the runtime configuration.

Memory construction and query answering use different snapshots. Use the
pre-query memory tar to investigate construction or choose a reuse donor; the
post-query archive may contain changes made while answering.

## Scores and comparisons

Candidate failures count as zero in the expected-query denominator.
Infrastructure, artifact, or judge errors make the comparable mean unavailable;
`observed_mean_query_score` reports the available scored observations separately.
Check `valid_for_comparison`, failure details, and coverage before comparing runs.
That flag alone does not establish a complete initial baseline: also verify the
initial candidate commit and full training case/query coverage.

Debug runs test execution without host judging and cannot measure answer quality.
Construction reuse does not reuse answers or scores: the current candidate
answers again and the host judge scores those new answers.
