# ReMe Docker sandbox

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

- One candidate may create many case sandboxes.
- Every case is a separate Docker container.
- Every case writes to its own `/workspace/case/reme_workspace`.
- Source-candidate bytes are snapshotted once and reused unchanged across all
  cases created by the same factory.
- Export happens before container shutdown.

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
credentials, benchmark datasets, existing memory workspaces, logs and build
output. Dataset inputs should be passed to cases explicitly rather than copied
into every candidate. Symbolic links are rejected so a snapshot cannot
accidentally capture files outside the candidate root.

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

## Direct job and artifact contract

The high-level methods map to the existing `lme.yaml` jobs:

- `ingest_session()` → `auto_memory`, then optionally `index_update`
- `answer()` → `agentic_answer`
- `judge()` → `answer_judge`, with `yes=1`, `no=0`
- `run_job()` → any explicitly named ReMe job

`export()` downloads a gzip archive containing the complete ReMe runtime
workspace, ReMe logs, command audit log, per-job JSON results, canonical answer
and score files, and a manifest. Environment variable names are recorded, but
their secret values are never written to the manifest.

For an ephemeral Docker workspace, always call `export()` before `close()`.
