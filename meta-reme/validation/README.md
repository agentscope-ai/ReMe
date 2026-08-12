# Meta-ReMe validation

Running `meta-reme/run.py` prepares the configured dataset and initial bundle,
then automatically validates the clean current commit on the `init` branch against every installed case.
The first result is stored under `evaluations/init/<commit-prefix>/initial/`; a completed result
is reused on later invocations. Configure concurrency with
`validation.concurrency` in `config_meta_reme.yaml`. Set
`validation.fail_fast: true` to abort the run on its first case or query error;
the default keeps other cases running.

Validate prepared workspace cases against the current committed code revision:

```bash
python meta-reme/validation/run.py \
  --workspace /path/to/workspace \
  --case-id case-1 \
  --case-id case-2 \
  --concurrency 2 \
  --fail-fast
```

`--case-id` may be repeated. Validation always uses the branch currently checked out in
`<workspace>/code/repo/reme`. Detached HEAD and branch names containing `/` are rejected.
Before creating any results, validation requires Git status to be clean, including staged,
unstaged, untracked, and submodule changes. It then resolves the full HEAD commit and builds
an immutable candidate snapshot from that commit. This keeps the evaluated code stable even
if the repository changes after validation begins.
Results are written without overwriting earlier runs:

```text
<workspace>/evaluations/<branch-name>/<commit-prefix>/<validation-id>/
```

`<commit-prefix>` starts with the first seven characters of the full commit SHA.
If that prefix is already used by a different commit on the same branch, it is
extended one character at a time until it is unique. Result manifests always
retain the full SHA.

`--fail-fast` is optional. When enabled, a construction failure, query
answer/judge error, or query infrastructure error is persisted before sibling
workers are cancelled and their containers are closed. The run raises an error
and writes a run-level `failure.json`; it does not publish a completed
`summary.json`. Without this flag, errors remain isolated to their cases.

Each case stores the two execution phases independently. Validation executes
each construction and query only once; retry policy belongs to the sandbox
implementation. Memory construction is exported before queries begin, so its
workspace is an exact
post-construction snapshot rather than the workspace left after evaluation:

```text
cases/<case-id>/
  case_result.json
  memory_construction/
    result.json
    reme_workspace.tar.gz
    reme_workspace/
    build.log
  queries/
    result.json
    <query-id>/
      answer.log
      result.json
```

`reme_workspace.tar.gz` is produced by the sandbox workspace export API and can
be passed directly to `upload_workspace()` on a fresh sandbox case before
running more queries. The same snapshot is safely extracted beside the archive
as `reme_workspace/` for direct inspection. After each session's successful
`auto_memory` and `index_update` pair, validation creates a local Git checkpoint
that commits only the configured `daily_dir`. The build result records the
checkpoint commit SHA, session message, and tracked path. The exported `.git`
directory therefore preserves the session-by-session evolution of source
memory while excluding derived metadata and indexes. Query artifacts use a
temporary archive only for transfer and safe extraction; the archive and its
redundant `summary.json` are deleted immediately afterward.

Validation uses a strict two-phase schedule. Reusable workers first construct
all case memories. A successful construction is exported as the case's
immutable query snapshot; a candidate construction failure is recorded and its
queries are skipped. Only after every construction reaches a terminal
state do workers begin answering queries.

Query cases have a preferred owner, while individual queries use fenced leases
so idle workers can steal remaining work. A worker first continues the exact
`case_id` already loaded in its container, then prefers its owned case, an
unowned case, and finally the case with the largest remaining query tail. Case
IDs are unique within one validation and containers do not cross validation
runs, so the exported snapshot hash remains an audit field rather than part of
the cache identity. This keeps workspace uploads uncommon without making a long
case an indivisible scheduling unit. Results are stored in the original case
and query slots, so completion order does not change summaries. Infrastructure
failures are persisted as terminal validation results. The affected container
is retired so it cannot contaminate later independent work, but the failed
construction or query is not requeued. Structured construction or answer/judge
failures likewise remain candidate results.

Containers are named by the factory rather than by their current case and call
`reset_case()` before switching snapshots, retaining only the installed
candidate and harness. All containers are shut down after validation finishes.

The same workflow is available as a project API. Use the synchronous entry point
from regular code:

```python
from validation import run_validation

result_dir = run_validation(workspace, case_ids, concurrency=2, fail_fast=True)
```

Applications that already run an event loop should use the native async entry
point instead:

```python
from validation import run_validation_async

result_dir = await run_validation_async(workspace, case_ids, concurrency=2)
```
