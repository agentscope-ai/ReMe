# Meta-ReMe validation

Running `meta-reme/run.py` prepares the configured dataset and initial bundle,
then automatically validates the `init` branch against every installed case.
The first result is stored under `evaluations/init/initial/`; a completed result
is reused on later invocations. Configure the full-case concurrency with
`validation.concurrency` in `config_meta_reme.yaml`.

Validate prepared workspace cases against an immutable code revision:

```bash
python meta-reme/validation/run.py \
  --workspace /path/to/workspace \
  --case-id case-1 \
  --case-id case-2 \
  --code-id <branch-name> \
  --concurrency 2
```

`--case-id` may be repeated. The code ID is the name of a local Git branch; commit
hashes and tags are not accepted. The branch is resolved to a full Git commit before
execution, and the candidate snapshot is built from that commit rather than the
current working tree. Code IDs must be path-safe branch names (for example,
`candidate-001`, without `/`) because the same value keys the result directory.
Results are written without overwriting earlier runs:

```text
<workspace>/evaluations/<code-id>/<validation-id>/
```

Each case attempt stores the two execution phases independently. Memory
construction is exported before queries begin, so its workspace is an exact
post-construction snapshot rather than the workspace left after evaluation:

```text
cases/<case-id>/attempt-<n>/
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
as `reme_workspace/` for direct inspection. Query artifacts use a temporary
archive only for transfer and safe extraction; the archive and its redundant
`summary.json` are deleted immediately afterward.

Each case uses a fresh sandbox. The concurrency limit covers the complete case
lifecycle: sandbox creation, memory construction, queries, artifact export, and
shutdown.

The same workflow is available as a project API. Use the synchronous entry point
from regular code:

```python
from validation import run_validation

result_dir = run_validation(workspace, case_ids, code_id, concurrency=2)
```

Applications that already run an event loop should use the native async entry
point instead:

```python
from validation import run_validation_async

result_dir = await run_validation_async(workspace, case_ids, code_id, concurrency=2)
```
