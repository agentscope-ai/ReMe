# Benchmark Docker runner

This package runs ReMe benchmark jobs directly inside isolated AgentScope
Docker workspaces. It does not start the ReMe HTTP service. Use AgentScope
**2.0.9** on the host and in the image.

From the repository root:

```bash
pip install -e '.[benchmark-docker]'
docker build -f deploy/benchmark_docker/Dockerfile.base \
  -t reme-benchmark-base:agentscope-2.0.9 .
```

`SourceCandidate` uploads an immutable source snapshot and installs the root
project and any explicitly selected plugin projects into a container-local
environment. The base image contains dependencies, but no ReMe candidate.
`ImageCandidate` instead uses a prebuilt image with ReMe and plugins installed.
The included `Dockerfile.candidate` is an example of that image mode.

```python
import asyncio
from pathlib import Path

from deploy.benchmark_docker import (
    BenchmarkDockerPool, CasePlan, DockerReMeSandboxFactory,
    EvaluationQuery, JobRequest, SourceCandidate, SourceSnapshot,
)


async def main():
    snapshot = SourceSnapshot.from_directory(".")
    factory = DockerReMeSandboxFactory(
        SourceCandidate(snapshot, install_projects=(".", "plugins/lme")),
        config="benchmark.yaml",
        plugins=("lme",),
        env={"LLM_API_KEY": "...", "EMBEDDING_API_KEY": "..."},
    )
    plans = [
        CasePlan(
            case_id="case-1",
            build_jobs=(JobRequest("auto_memory", {"session_id": "s1", "messages": []}),),
            queries=(EvaluationQuery("q1", "Question?", "Private answer"),),
            export_workspace_to=Path("artifacts/case-1-workspace.tar.gz"),
            export_evaluation_to=Path("artifacts/case-1-evaluation.tar.gz"),
        ),
    ]
    return await BenchmarkDockerPool(factory, concurrency=2).run(plans)


asyncio.run(main())
```

Each worker completes one case's build and queries before taking the next
case. A case can set `reuse_workspace_from="case-1"` to receive the source
case's workspace **as it stood immediately after construction**, or provide
`workspace_source` pointing at a host workspace directory or a saved archive.
A reused workspace may be queried without additional build jobs.
`export_workspace_to` persists a portable snapshot beyond the pool run.
To build memory once and query it in later runs using the same factory:

```python
memory = Path("artifacts/memory.tar.gz")
pool = BenchmarkDockerPool(factory, concurrency=2)
built = (await pool.run([
    CasePlan("build-memory", build_jobs=(JobRequest("auto_memory", {
        "session_id": "s1", "messages": [],
    }),), export_workspace_to=memory),
]))[0]
if not built.success:
    raise RuntimeError(built.error or "memory build failed")

# This may run later in another process with a compatible factory/config.
answers = await pool.run([
    CasePlan("query-memory", workspace_source=memory,
             queries=(EvaluationQuery("q1", "Question?", "Private answer"),)),
])
```

Use compatible candidate code, plugins, and workspace path configuration when
restoring memory. Uploading a snapshot does not change the receiving config.
`export_workspace()` archives contain workspace contents directly. Case archives
from `export_memory_construction()`, `export_evaluation()`, and `export()` carry
a `manifest.json` format marker and restore only `reme_workspace/`, including
its Git history; build logs and query artifacts are excluded from the upload.
Evaluation exports contain the workspace after queries; use a construction or
workspace export before queries to preserve the original built memory.
Query-only case exports cannot be uploaded as workspaces. Older full case
archives remain supported. Older partial archives with a `reme_workspace`
wrapper but no manifest are rejected; to reuse one, extract it and supply its
`reme_workspace` directory.

`export_evaluation_to` saves the workspace and query artifacts for both build
and query-only cases; it includes `build_log` when that phase ran. Set
`export_build_log_to` to retain the construction log separately. On a failed
build, the pool also returns its text in `CaseResult.build_log`. `command_timeout`
(default 1800 seconds) bounds the entire shell command, including worker startup,
jobs, and Application shutdown. On timeout, cancellation, or a shell backend
failure, the sandbox force-stops the container without candidate or plugin
cooperation, attempts to cache the build log for up to 5 seconds, and removes the
container before propagating the error. Docker stop and removal each have a
30-second timeout; cleanup failures include the container ID in exception notes.
Only the cached build log remains available after removal. The pool creates a
replacement when more cases need a worker.
Custom `workspace_builder` implementations must provide an async `stop()` that
force-stops execution while retaining files for diagnostics, in addition to `close()`.
Replacement requires successful cleanup of the old worker. Cleanup failures
are reported in `CaseResult.error` and make `success` false for cases that used
that worker; failures for an unused worker apply to the whole run. Cancellation
waits for owned-container cleanup before propagating. A failed `close()` retains
unfinished Docker handles so another call can retry removal.
The build log also records configuration, construction, build job, checkpoint,
and Application lifecycle failures, including available tracebacks.

For direct control, call `create_case()` or `create_cases()`, then use
`run_job()`, `run_build()`, `run_queries()`, `export_workspace()`,
`upload_workspace()`, `reset_case()`, and `close()`. Always close a case in a
`finally` block or use its async context manager. Export before closing.
The default `export("analysis.tar.gz")` also saves available construction and
query logs under `analysis.tar.gz.artifacts/build_log/` and
`analysis.tar.gz.artifacts/queries/`. These logs stay outside the archive;
choose a new destination if that artifacts directory already exists.

The candidate receives only query ID, question, answer Job, and answer
arguments. Golden answers, rubrics, and scoring remain on the host. Workspace
archives reject links, special files, duplicate paths, and traversal paths.
Source snapshots exclude VCS state, credentials, benchmark data, and generated
artifacts; review custom source trees for other secrets before uploading.

`tests/unit/test_benchmark_docker.py` runs without Docker. The credential-free
container smoke tests build the base image and temporary candidate images when
explicitly enabled:

```bash
REME_DOCKER_INTEGRATION=1 pytest tests/integration/test_benchmark_docker.py -v
```

The host must have AgentScope 2.0.9 and a working Docker daemon. To use an
already built base image, set `REME_BENCHMARK_TEST_IMAGE` to its tag; the tests
do not remove a caller-supplied image.
