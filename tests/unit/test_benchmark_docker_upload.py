"""Candidate build exclusions and nonblocking workspace upload without Docker."""

# pylint: disable=missing-function-docstring,protected-access

import asyncio
import io
from pathlib import Path
import tarfile
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from deploy.benchmark_docker import ImageCandidate
from deploy.benchmark_docker import docker_sandbox as sandbox_module
from deploy.benchmark_docker.docker_sandbox import _archive_workspace


def test_candidate_image_preserves_root_exclusions_after_plugin_exceptions():
    root = Path(__file__).resolve().parents[2]
    root_rules = (root / ".dockerignore").read_text(encoding="utf-8").splitlines()
    candidate_rules = (
        (root / "deploy/benchmark_docker/Dockerfile.candidate.dockerignore").read_text(encoding="utf-8").splitlines()
    )
    exceptions = [rule for rule in candidate_rules if rule.startswith("!")]
    assert set(exceptions) == {"!plugins/lme", "!plugins/beam"}
    assert "plugins/*" in candidate_rules
    last_exception = max(candidate_rules.index(rule) for rule in exceptions)
    exclusions = candidate_rules[last_exception + 1 :]
    # New root protections must also cover the Dockerfile-specific context,
    # including secrets inside otherwise allowed plugin directories.
    assert set(root_rules) - {"plugins", ""} <= set(exclusions)
    assert {"baselines", "meta-reme", "output", "reme_studio"} <= set(exclusions)


@pytest.mark.asyncio
@pytest.mark.parametrize("archive_error", [False, True])
async def test_workspace_archiving_keeps_event_loop_responsive(tmp_path, monkeypatch, archive_error):
    backend = AsyncMock()
    backend.exec_shell.return_value = SimpleNamespace(exit_code=0, stdout=b"", stderr=b"", ok=lambda: True)
    case = sandbox_module.DockerReMeSandbox(
        case_id="archive-thread",
        container_id="unused",
        candidate=ImageCandidate("unused"),
        workspace=AsyncMock(),
        config="default",
        plugins=(),
        command_timeout=5,
        environment_names=[],
    )
    case.backend = backend
    source = tmp_path / "memory"
    source.mkdir()
    (source / "note.md").write_text("saved memory", encoding="utf-8")
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    release = threading.Event()
    event_loop_thread = threading.get_ident()

    def blocking_archive(path):
        assert threading.get_ident() != event_loop_thread
        loop.call_soon_threadsafe(started.set)
        if not release.wait(timeout=5):
            raise TimeoutError("test did not release archive worker")
        if archive_error:
            raise ValueError("invalid workspace archive")
        return _archive_workspace(path)

    monkeypatch.setattr(sandbox_module, "_archive_workspace", blocking_archive)
    upload = asyncio.create_task(case.upload_workspace(source))
    try:
        await asyncio.wait_for(started.wait(), timeout=2)
        await asyncio.sleep(0)
        assert not upload.done()
        # No workspace clearing or upload may happen before validation finishes.
        backend.exec_shell.assert_not_awaited()
        backend.write_file.assert_not_awaited()
        release.set()
        if archive_error:
            with pytest.raises(ValueError, match="invalid workspace archive"):
                await upload
            backend.exec_shell.assert_not_awaited()
            backend.write_file.assert_not_awaited()
        else:
            await upload
            backend.write_file.assert_awaited_once()
            archive_path, payload = backend.write_file.await_args.args
            assert archive_path == sandbox_module.WORKSPACE_ARCHIVE
            with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
                assert archive.extractfile("note.md").read() == b"saved memory"
    finally:
        release.set()
        await asyncio.gather(upload, return_exceptions=True)
        await case.close()
