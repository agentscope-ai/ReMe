# ReMe Candidate Development Guide

## Workspace and modification boundary

Read `candidate.paths` in `home/domain_spec.yaml` for the editable roots relative
to `home/code/repo/`. Use `runtime.config` and `runtime.plugins` in the same file
to locate the active configuration. Treat `home/code/runtime/`, datasets,
evaluation evidence, and Git metadata as read-only.

Improve general memory behavior. Keep benchmark answers and case-specific rules
out of candidate code, prompts, and resources.

## ReMe architecture

- Preserve the inputs and responses of benchmark-facing Jobs such as
  `auto_memory` and `agentic_answer`. Check their callers, configured Steps, and
  plugin configuration before editing an interface. Use dependencies available
  in the evaluation environment.
- Step fields and kwargs last for one Job invocation. Use `RuntimeContext.data`
  to pass intermediate values between Steps. Use application-scoped Components
  for shared resources and workspace files for durable state.
- Keep user-owned files as the source of truth and derived indexes and caches
  rebuildable. Follow ReMe's workspace path constraints.

## Candidate versions

Fork from `init` or a protected evaluated branch before editing. After a merge,
review `home_code_diff` before committing. Follow the
[evaluation guide](evaluation-artifact-guide.md) to test the commit.

When diagnosing a run, use its `commit_sha` as `revision` wherever the code tool
exposes that argument. Carry the same revision and the path from
`home_search_code` into `home_read_code_file` to verify the surrounding code.
