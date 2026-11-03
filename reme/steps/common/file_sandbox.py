"""Resolve the Python executor's mandatory, server-configured file sandbox."""

from ...utils.file_sandbox import create_file_sandbox_backend
from ...utils.file_sandbox.file_sandbox_backend import FileSandboxBackend
from ..base_step import BaseStep


def file_sandbox_for_step(step: BaseStep) -> FileSandboxBackend:
    """Default to a read-only workspace; callers cannot supply path grants."""
    assert step.context is not None
    for name in ("read_only_paths", "writable_paths"):
        if name in step.context:
            raise ValueError(f"{name} must be configured on the Step, not supplied by the caller")
    workspace = step.workspace_path.resolve()
    read_only_paths = step.kwargs.get("read_only_paths", [str(workspace)])
    writable_paths = step.kwargs.get("writable_paths", [])
    memory_limit_mib = step.context.get("memory_limit_mib", step.kwargs.get("memory_limit_mib"))
    return create_file_sandbox_backend(
        cwd=workspace,
        read_only_paths=read_only_paths,
        writable_paths=writable_paths,
        memory_limit_mib=memory_limit_mib,
    )
