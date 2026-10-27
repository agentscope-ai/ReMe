# Home plugin

Adds a self-evolution workspace, the one-time `home_prepare` job, and
`auto_bash` through an independently enabled ReMe plugin.

Install it into the same Python environment as ReMe:

```bash
reme plugins install plugins/home
```

Enable it in an application config with `plugins: [home]`, then start the
service and call its jobs from another terminal:

```bash
reme start config=<your-config.yaml>
# In another terminal:
reme auto_bash command=pwd
```

At application startup, Home automatically adds these directories inside the
configured ReMe `workspace_dir`:

```text
code/repo/
code/runtime/
datasets/training-set/cases/
datasets/debug-cases/cases/
docs/
evaluations/
sampling/plans/
tmp/
logs/
```

ReMe's existing memory, resource, and session directories remain in place.
On first initialization, each Home top-level directory must be absent or
completely empty (an empty child directory still counts as content). File
conflicts, symbolic links in managed paths, and overlap with configured ReMe
directories cause startup to fail without overwriting their contents.

Home atomically publishes `{metadata_dir}/home/workspace.json` after creating
the layout. It records layout schema version 1, a workspace UUID, and an aware
creation timestamp. Subsequent starts validate this marker, preserve its identity
and existing contents, and recreate missing directories. Recreating directories
does not recover missing data. Invalid or unsupported markers fail explicitly.
A process lock at `{metadata_dir}/home/workspace.lock` prevents concurrent
initialization; the lock file remains on disk, but process exit releases the
lock. A competing initializer can retry after the first finishes.

Failed initialization removes only newly created empty layout directories and
temporary manifest files. Existing directories and files are preserved. After
an abrupt process termination before the marker is published, inspect any
nonempty Home directories before retrying; they are not automatically adopted.
Initialization creates the layout only: it does not install datasets, copy
candidate code or documentation, initialize Git, or generate experiment/domain
configuration, sampling state, or evaluation results.

For explicitly one-shot CLI execution, use:

```bash
reme start config=<your-config.yaml> job=auto_bash command=pwd
```

The job defaults to read-only command validation and a 60-second timeout. Set
`is_read_only=false` to disable command validation. Supplying `read_only_paths`
or `writable_paths` enables a file sandbox (Bubblewrap on Linux, Seatbelt on
macOS), with sandbox paths identical to their host paths. On Linux, the
`memory_limit_mib` cgroup limit defaults to 4096 MiB; 0 disables it. Supplying
the limit requires a path policy. macOS has no memory limit and warns when a
nonzero value is supplied.

Standard output and standard error each have a 1 MiB (1,048,576-byte) limit on
all backends. Exceeding either limit terminates the command and returns a
failure with bounded partial output and a diagnostic in standard error.

## Prepare an experiment once

`home_prepare` runs explicitly after startup. It converts BEAM or LongMemEval
inputs, exports a committed ReMe runtime and selected plugins, and initializes a
candidate Git repository on branch `init`. It does not install dependencies, run
source code, build Docker images, call models, or evaluate the baseline.

Copy [examples/prepare.yaml](examples/prepare.yaml) into your workspace as
`prepare.yaml` and edit the source paths, revision and case IDs. For LongMemEval,
set `benchmark.name: longmemeval`, point `source` at its JSON file, remove
`variant`, and choose `runtime.plugins: [lme]` and `candidate.paths:
[plugins/lme/src]`. Optional `benchmark.evidence_source` accepts reviewed JSONL
session evidence. Choose nonempty, unique string `case_ids`, or explicitly set
`all_cases: true` without `case_ids`. Relative input paths are workspace-relative;
explicit absolute dataset and repository paths are read-only source inputs.
`spec_path` must resolve to a file inside the workspace. Home-relative paths and
relative traversal are rejected.

Start a service configured with `plugins: [home]` and your `workspace_dir`, then
call it from another terminal using the same service selection as other jobs:

```bash
reme start config=<your-config.yaml>
# In another terminal:
reme home_prepare spec_path=prepare.yaml
```

For explicitly one-shot local execution:

```bash
reme start config=<your-config.yaml> job=home_prepare spec_path=prepare.yaml
```

Python uses the same Job and normal Application lifecycle; see
[examples/prepare.py](examples/prepare.py). Pass exactly one of `spec_path` or
`spec` (a configuration dictionary):

```python
from reme.application import Application

app = Application(workspace_dir="/experiments/run-1", plugins=["home"], service={"backend": "cli"})
try:
    await app.start()
    response = await app.run_job("home_prepare", spec=preparation_config)
    if not response.success:
        raise RuntimeError(response.answer)
finally:
    await app.close()
```

The prepared workspace contains:

- `domain_spec.yaml`: resolved configuration, including actual selected case IDs
  and the full source commit SHA.
- `datasets/training-set/`: normalized Case/Session/Query JSON and a versioned
  manifest with case IDs and content fingerprints.
- `datasets/debug-cases/`: deterministic small smoke cases and their derivation
  record, or an empty `cases/` directory when debug derivation is disabled.
- `code/runtime/`: committed ReMe sources, packaging files, configurations and
  requested plugin projects. Uncommitted source edits are excluded.
- `code/repo/`: an independent Git repository containing only the candidate
  paths, retaining their original repository-relative locations. Future runners
  overlay these paths onto a copy of `code/runtime/`; never edit the fixed base.
- `{metadata_dir}/home/preparation.json`: completion record, config/data/code
  fingerprints, initial candidate commit, plugin and installation plan.

Before preparation, code directories must be empty, dataset directories may
contain only their empty `cases/` skeleton, and `domain_spec.yaml` must be absent.
Candidate paths must be present in the exported runtime, cannot overlap, and may
select ReMe core or plugin files. Runtime plugins are selected explicitly; the host plugin list is not copied. All selected source files must be regular Git blobs; symlinks and
submodules are rejected. Source references are explicit Git revisions; this
version does not support uncommitted working-tree snapshots or remote fetching.

Preparation holds the same process lock as layout initialization. Inputs are
staged beneath `{metadata_dir}/home/preparation-transaction/`, flushed, and
published with a durable rollback journal. Reported preparation failures restore
pre-existing files and directories (filesystem timestamps are not part of
the contract); the corrected request can be retried. Interrupted transactions
are recovered at startup or before another preparation. A completed preparation
rejects **every** later call, including identical parameters. There is no reset or
force-overwrite option: use a new workspace for another experiment.

Only Home's writers cooperate with this lock. Keep the preparation target
locations exclusively owned by this task while preparing and recovering; other
processes must not write there. Recovery requires a writable, functioning
filesystem; if rollback fails, its journal remains for the next attempt and the
Job reports the error. Calls with `_allowed_paths` restrictions are rejected
because preparation owns multiple workspace output locations.
