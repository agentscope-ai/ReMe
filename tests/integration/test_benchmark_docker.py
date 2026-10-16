"""Opt-in, credential-free smoke test against real AgentScope Docker containers.

Run with ``REME_DOCKER_INTEGRATION=1 pytest tests/integration/test_benchmark_docker.py -v``.
The base image is built from the repository Dockerfile unless
``REME_BENCHMARK_TEST_IMAGE`` names an existing one. Temporary image tags are
removed after the test; workspace files stay under pytest's temporary root.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import uuid

import pytest

from deploy.benchmark_docker import (
    BenchmarkDockerPool,
    CasePlan,
    DockerReMeSandboxFactory,
    EvaluationQuery,
    ImageCandidate,
    JobRequest,
    SourceCandidate,
    SourceSnapshot,
)

ROOT = Path(__file__).resolve().parents[2]
SMOKE_CONFIG = """\
enable_logo: false
service:
  backend: cli
jobs:
  smoke_write:
    backend: base
    steps:
      - backend: write_step
  smoke_read:
    backend: base
    steps:
      - backend: read_step
"""


def _docker(*arguments: str, cwd: Path | None = None) -> None:
    subprocess.run(["docker", *arguments], cwd=cwd, check=True, timeout=1200)


@pytest.fixture(scope="module", name="base_image")
def _base_image():
    """Build the repository base image unless the caller supplies one."""
    if os.environ.get("REME_DOCKER_INTEGRATION") != "1":
        pytest.skip("set REME_DOCKER_INTEGRATION=1 to run the Docker integration test")
    supplied = os.environ.get("REME_BENCHMARK_TEST_IMAGE")
    if supplied:
        yield supplied
        return

    tag = f"reme-benchmark-smoke:{uuid.uuid4().hex[:12]}"
    _docker("build", "-f", "deploy/benchmark_docker/Dockerfile.base", "-t", tag, ".", cwd=ROOT)
    try:
        yield tag
    finally:
        _docker("image", "rm", tag)


def _candidate_source(tmp_path: Path) -> Path:
    source = tmp_path / "candidate"
    source.mkdir()
    shutil.copytree(
        ROOT / "reme",
        source / "reme",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )
    shutil.copy2(ROOT / "pyproject.toml", source / "pyproject.toml")
    shutil.copy2(ROOT / "README.md", source / "README.md")
    (source / "smoke.yaml").write_text(SMOKE_CONFIG, encoding="utf-8")
    return source


def _container_exists(name: str) -> bool:
    result = subprocess.run(
        ["docker", "container", "inspect", name],
        capture_output=True,
        check=False,
        timeout=30,
    )
    return result.returncode == 0


def _query(query_id: str) -> EvaluationQuery:
    return EvaluationQuery(
        query_id=query_id,
        question="Read the saved note",
        golden_answer="docker smoke",
        answer_job="smoke_read",
        answer_arguments={"path": "note.md"},
    )


@pytest.mark.asyncio
async def test_real_docker_timeout_kills_blocked_thread_pool(base_image):
    """A host deadline must terminate a worker blocked in executor shutdown."""
    from deploy.benchmark_docker.direct_docker_workspace import DirectDockerWorkspace
    from deploy.benchmark_docker.docker_sandbox import DockerReMeSandbox, SandboxCommandTimeoutError

    container_id = f"timeout-{uuid.uuid4().hex[:12]}"
    workspace = DirectDockerWorkspace(workspace_id=container_id, base_image=base_image)
    case = DockerReMeSandbox(
        case_id="timeout",
        container_id=container_id,
        candidate=ImageCandidate(base_image),
        workspace=workspace,
        config="unused",
        plugins=(),
        command_timeout=2,
        environment_names=[],
    )
    try:
        await workspace.initialize()
        case.backend = workspace.get_backend()
        # Reproduce the shutdown condition without importing ReMe or a plugin.
        script = (
            "import threading; from pathlib import Path; from concurrent.futures import ThreadPoolExecutor; "
            "pool = ThreadPoolExecutor(1); pool.submit(threading.Event().wait); "
            "log = Path('/workspace/case/build_log/build.log'); log.parent.mkdir(parents=True); "
            "log.write_text('waiting for thread pool shutdown'); pool.shutdown(wait=True)"
        )
        # pylint: disable=protected-access
        async with case._operation_lock:
            with pytest.raises(SandboxCommandTimeoutError):
                await asyncio.wait_for(case._exec("blocked-close", ["python3", "-c", script]), timeout=75)
        assert not _container_exists(f"as_ws_{container_id}")
        assert not case.reusable
        assert await case.read_build_log() == "waiting for thread pool shutdown"
    finally:
        await case.close()


@pytest.mark.asyncio
async def test_real_docker_source_pool_workspace_reuse(base_image, tmp_path):
    """Run build, query, workspace reuse, and query-only export in a real container."""
    source = _candidate_source(tmp_path)
    factory = DockerReMeSandboxFactory(
        SourceCandidate(SourceSnapshot.from_directory(source), base_image=base_image),
        config="/workspace/candidate/smoke.yaml",
    )
    evaluation = tmp_path / "child-evaluation.tar.gz"
    plans = [
        CasePlan(
            "parent",
            build_jobs=(JobRequest("smoke_write", {"path": "note.md", "content": "docker smoke"}),),
            queries=(_query("parent-query"),),
        ),
        CasePlan(
            "child",
            reuse_workspace_from="parent",
            queries=(_query("child-query"),),
            export_evaluation_to=evaluation,
        ),
    ]

    results = await asyncio.wait_for(BenchmarkDockerPool(factory, concurrency=1).run(plans), timeout=360)

    assert all(result.success for result in results), results
    assert "docker smoke" in str(results[1].queries["queries"][0]["answer"])
    with tarfile.open(evaluation, "r:gz") as archive:
        names = archive.getnames()
        assert any(name.endswith("reme_workspace/note.md") for name in names)
        assert any(name.endswith("queries/child-query/answer.log") for name in names)
        assert not any(name.endswith("build_log/build.log") for name in names)


@pytest.mark.asyncio
async def test_real_docker_source_plugin_install(base_image, tmp_path):
    """Upload and install a plugin project in the candidate environment."""
    source = _candidate_source(tmp_path)
    shutil.copytree(
        ROOT / "plugins" / "lme",
        source / "plugins" / "lme",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )
    factory = DockerReMeSandboxFactory(
        SourceCandidate(
            SourceSnapshot.from_directory(source),
            base_image=base_image,
            install_projects=(".", "plugins/lme"),
        ),
        plugins=("lme",),
        config="/workspace/candidate/smoke.yaml",
    )
    async with await asyncio.wait_for(factory.create_case("plugin-smoke"), timeout=180) as case:
        container_name = f"as_ws_{case.container_id}"
        assert _container_exists(container_name)
    assert not _container_exists(container_name)


@pytest.mark.asyncio
async def test_real_docker_prebuilt_image(base_image, tmp_path):
    """Install a fixed candidate into an image and execute its jobs."""
    source = _candidate_source(tmp_path)
    dockerfile = source / "Dockerfile.smoke"
    dockerfile.write_text(
        f"FROM {base_image}\nCOPY . /opt/reme\nRUN python -m pip install --no-deps --no-build-isolation /opt/reme\n",
        encoding="utf-8",
    )
    image = f"reme-benchmark-candidate-smoke:{uuid.uuid4().hex[:12]}"
    _docker("build", "-f", dockerfile.name, "-t", image, ".", cwd=source)
    try:
        factory = DockerReMeSandboxFactory(ImageCandidate(image), config="/opt/reme/smoke.yaml")
        async with await factory.create_case("image-smoke") as case:
            container_name = f"as_ws_{case.container_id}"
            build = await asyncio.wait_for(
                case.run_build([JobRequest("smoke_write", {"path": "note.md", "content": "image smoke"})]),
                timeout=120,
            )
            assert build["success"], build
            query = await asyncio.wait_for(case.run_queries([_query("image-query")]), timeout=120)
            assert query["success"], query
            assert "image smoke" in str(query["queries"][0]["answer"])
        assert not _container_exists(container_name)
    finally:
        _docker("image", "rm", image)
