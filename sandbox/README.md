# ReMe Docker sandbox

[简体中文](README_ZH.md)

This directory runs benchmark cases in AgentScope `DockerWorkspace` containers.
It never starts the ReMe HTTP service: the uploaded worker constructs a ReMe
`Application` and calls `run_job()` directly.

The harness uses a small `DockerWorkspace` subclass that retains AgentScope's
Docker lifecycle and `DockerBackend`, but skips its MCP gateway build/start.
The gateway is not used by direct ReMe jobs and would otherwise add a second
network-dependent image build and background process to every base image.

Install the host-side Docker dependency with:

```bash
pip install -e ".[sandbox]"
```

## Isolation model

- One candidate may create many container sandboxes.
- `create_cases()` gives every case a separate container; `reset_case()` can
  instead reuse one container for sequential cases.
- The active case writes to `/workspace/case/reme_workspace`, which is deleted
  before the container is assigned to the next case.
- Source-candidate bytes are snapshotted once and reused unchanged across all
  cases created by the same factory.
- Export happens before container shutdown.
- Case IDs must start with an ASCII letter or digit, contain only letters,
  digits, `.`, `_`, or `-`, and be at most 128 characters. IDs passed to one
  `create_cases()` call must be unique.

## Build the dependency-only base image

Run from the repository root:

```bash
docker build \
  -f sandbox/Dockerfile.base \
  -t reme-sandbox-base:agentscope-2.0.4-post1 \
  .
```

The image installs the dependencies from `pyproject.toml`, verifies AgentScope
`2.0.4.post1`, and then uninstalls the dummy ReMe package. Candidate source is
therefore the only importable ReMe implementation after case preparation.

## Mode 1: changing source candidate

```python
import asyncio
import os

from sandbox import DockerReMeSandboxFactory, SourceCandidate, SourceSnapshot


async def main():
    # Do this once per candidate, not once per case.
    snapshot = SourceSnapshot.from_directory(".")
    factory = DockerReMeSandboxFactory(
        SourceCandidate(snapshot),
        env={
            name: os.environ[name]
            for name in ("LLM_API_KEY", "LLM_BASE_URL")
            if name in os.environ
        },
    )

    # Each ID gets a different container and runtime workspace, while both
    # receive exactly the same snapshot bytes.
    cases = await factory.create_cases(["session-001", "session-002"])
    try:
        for case in cases:
            await case.ingest_session(
                session_id=case.case_id,
                messages=[{"role": "user", "content": "Remember this."}],
            )
            await case.commit_memory_history(f"session: {case.case_id}")
            answer = await case.answer(query="What should you remember?")
            await case.judge(
                query="What should you remember?",
                agent_answer=str(answer.answer),
                golden_answer="Remember this.",
            )
            await case.export(f"artifacts/{case.case_id}.tar.gz")
    finally:
        await asyncio.gather(*(case.close() for case in cases))


asyncio.run(main())
```

Source archives exclude VCS state, `.reme`, virtual environments, caches,
`.env` credential files, benchmark datasets, existing memory workspaces, logs
and build output. Other secrets are not discovered automatically and must not
be placed in the candidate tree. Dataset inputs should be passed to cases
explicitly rather than copied into every candidate. Symbolic links are
rejected so a snapshot cannot accidentally capture files outside the
candidate root.

## Mode 2: candidate preinstalled in an image

For a stable candidate, use the repository's `.dockerignore` (an equivalent
example is kept at `sandbox/.dockerignore.example`) and run:

```bash
docker build \
  -f sandbox/Dockerfile.candidate \
  --build-arg BASE_IMAGE=reme-sandbox-base:agentscope-2.0.4-post1 \
  -t reme-candidate:my-candidate \
  .
```

Use that candidate without uploading source per case:

```python
from sandbox import DockerReMeSandboxFactory, ImageCandidate

factory = DockerReMeSandboxFactory(
    ImageCandidate("reme-candidate:my-candidate", candidate_id="git-or-content-hash"),
    env={...},
)
case = await factory.create_case("session-001")
```

## Direct single-job and artifact contract

The legacy convenience methods map to the existing `lme.yaml` jobs:

- `ingest_session()` → `auto_memory`, then optionally `index_update`
- `answer()` → `agentic_answer`
- `judge()` → `answer_judge`, with `yes=1`, `no=0`
- `run_job()` → any explicitly named ReMe job

`export()` defaults to an analysis-focused gzip archive for this single-job
flow. It contains the accumulated files under the legacy `logs/` directory,
the command audit log, per-job JSON results, canonical answer and score files,
persisted agent sessions, and the user-owned workspace files, including files
written outside the expected daily/digest paths. It also includes a validated
runtime layout and a manifest. It omits request copies, temporary files, source
resources, and rebuildable indexes/caches. Raw dialog sessions under the
configured `session_dir/dialog` path are included. Use
`export(profile="full")` or `export_full()` to download the complete disposable
case tree. `include_candidate=True` additionally appends an uploaded source
candidate; it is invalid for an image candidate. Environment variable names
are recorded, but their secret values are never written to the manifest.

The batch artifacts described below live under `build_log/` and `queries/` and
are intentionally not part of the legacy analysis profile. Use
`export_evaluation()` for them, or the full profile when the entire case is
needed.

## Build and multi-query batches

For multi-query evaluation, use the batch APIs instead of starting one
Application per job. `run_build()` executes every construction job in one
Application. `run_queries()` then starts one new Application and reuses it for
all answer-and-judge pairs. Query execution is sequential so token deltas and
the process-global log sink remain attributable to one query. The restart at
the build/query boundary also guarantees that evaluation depends on persisted
workspace state rather than construction-only in-memory state.

Each method publishes one phase for the active case. `run_build()` requires at
least one job, stops at the first failed job, and refuses to append to an
existing `build_log/build.log`. `run_queries()` requires at least one unique
query ID, continues after individual query failures, and refuses to append if
`queries/summary.json` or any requested query directory already exists. Call
`reset_case()` before starting another evaluation in the same container.

```python
from sandbox import EvaluationQuery, JobRequest

build = await case.run_build(
    [
        JobRequest("auto_memory", {"session_id": "session-1", "messages": messages}),
        JobRequest("index_update"),
    ],
)
assert build["success"]
await case.commit_memory_history("constructed memory")

evaluation = await case.run_queries(
    [
        EvaluationQuery(
            query_id="question-1",
            question="What should be remembered?",
            golden_answer="Remember this.",
            judge_arguments={
                "query": "What should be remembered?",
                "golden_answer": "Remember this.",
            },
        ),
    ],
)
assert evaluation["success"]
await case.export_evaluation("artifacts/case-1.tar.gz")
```

`export_evaluation()` produces exactly three top-level directories:

```text
reme_workspace/
build_log/
  build.log
queries/
  summary.json
  <query-id>/
    answer.log
    result.json
```

Unlike the analysis profile, this export includes the complete
`reme_workspace/`, including local Git history and rebuildable metadata,
resources, indexes, or caches that are present. Export requires both
`build_log/build.log` and `queries/summary.json`; call it only after both batch
phases have published their artifacts.

The query ID is used verbatim as its directory name and must therefore be one
safe path component: it cannot be empty, `.`, `..`, `summary.json`, contain
`/`, `\`, or NUL, or exceed 255 UTF-8 bytes. Each `answer.log` contains the
answer and judge logs for that query only; Application startup and shutdown
logs are excluded. Each `result.json` contains the question, golden answer,
answer, normalized score, raw answer/judge results, token deltas, and any
error. `queries/summary.json` records `case_id`, counts, mean score, and each
query ID and score without a redundant directory field. A failed query
receives a `null` score and an error, while later queries still run.

Because IDs remain verbatim, callers are responsible for extraction-platform
portability. For example, `:` is valid in the Linux container and archive but
is not a valid Windows filename character.

`EvaluationQuery` defaults to the LongMemEval convention: the answer is
injected into the judge argument `agent_answer`, and judge answers `yes` and
`no` map to `1.0` and `0.0`. `answer_arguments` and `judge_arguments` hold the
remaining job arguments. `answer_job`, `judge_job`, `judge_answer_argument`,
`score_path`, and `score_mapping` can override those conventions. Scores must
normalize to the inclusive range `[0, 1]`.

BEAM-style judges can select a numeric score from judge metadata:

```python
EvaluationQuery(
    query_id="information_extraction:1",
    question=question,
    golden_answer=rubric,
    judge_answer_argument="llm_response",
    judge_arguments={
        "rubric": rubric,
        "probing_question": question,
        "question_type": "information_extraction",
    },
    score_path="metadata.llm_judge_score",
    score_mapping=None,
)
```

## Local Git history for memory

The runtime workspace is also a local Git repository. Session ingestion never
commits implicitly: the host chooses checkpoint boundaries by calling
`commit_memory_history(message)`. It may checkpoint after every session or
after any batch of sessions, and controls the commit message. Each checkpoint
commits only the configured `daily_dir`; empty commits are retained as explicit
boundaries. The exported workspace includes `.git`, so the daily-memory
construction history can be inspected offline without a remote or push. A
`reset_case()` removes this repository together with the old runtime workspace
and initializes a fresh repository for the next case.

For an ephemeral Docker workspace, download every artifact that must survive
before `close()`: use `export()`, `export_full()`, `export_evaluation()`, or
`export_workspace()` according to the required contract.

## Reuse one built memory for parallel queries

Use `export_workspace()` after constructing the memory, then give the snapshot
to each independent query case with `upload_workspace()`. Upload clears the
target runtime workspace by default, so the cases start from the same memory
state but never share mutable files or indexes.

```python
memory_case = await factory.create_case("build-memory")
try:
    await memory_case.ingest_session(session_id="source", messages=[...])
    await memory_case.run_job("index_update")
    snapshot = await memory_case.export_workspace("artifacts/built-memory.tar.gz")
finally:
    await memory_case.close()

query_cases = await factory.create_cases(["query-1", "query-2"])
try:
    await asyncio.gather(*(case.upload_workspace(snapshot) for case in query_cases))
    answers = await asyncio.gather(
        *(case.answer(query=query) for case, query in zip(query_cases, queries))
    )
finally:
    await asyncio.gather(*(case.close() for case in query_cases))
```

`upload_workspace()` also accepts a host directory or a legacy `export()`
archive containing both `manifest.json` and `reme_workspace/`. It validates
and repacks the directory or input archive before extraction; symbolic links,
special files, duplicate archive paths, and traversal paths are rejected. An
`export_evaluation()` archive is not an upload source because it intentionally
has no manifest; use `export_workspace()` for portable snapshots. Pass
`clear=False` only when an intentional merge into the target workspace is
required. Upload always runs `git init`; existing history is preserved, while
a directory without `.git` receives a fresh repository.

## Reuse one container for sequential cases

When container startup and source installation dominate the benchmark, one
container can process multiple cases sequentially. Finish and optionally
export the active case, then reset the disposable case directory before
uploading the next case's sessions:

```python
case = await factory.create_case("session-001")
try:
    await case.ingest_session(session_id="session-001", messages=[...])
    await case.commit_memory_history("session: session-001")
    first = await case.answer(query="...")
    await case.export("artifacts/session-001.tar.gz")

    await case.reset_case("session-002")
    await case.ingest_session(session_id="session-002", messages=[...])
    await case.commit_memory_history("session: session-002")
    second = await case.answer(query="...")
finally:
    await case.close()
```

`reset_case()` deletes the ReMe runtime workspace, requests, legacy logs,
results, `build_log/`, `queries/`, manifest, case-scoped temporary files, and
temporary export archives. It retains the installed candidate, candidate
virtual environment, and benchmark worker. Jobs, batch phases, exports,
uploads, commits, and resets share one lock, so they never overlap within a
container. Do not retain paths or state from the previous case after reset
returns.
