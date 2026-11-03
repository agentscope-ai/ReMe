"""Resolve executor sandbox policies."""

from ...utils.file_sandbox import create_file_sandbox_backend
from ...utils.file_sandbox.file_sandbox_backend import FileSandboxBackend
from ..base_step import BaseStep


def file_sandbox_for_step(step: BaseStep) -> FileSandboxBackend | None:
    """Resolve optional path grants for executors that support host execution."""
    assert step.context is not None
    read_only_paths = step.context.get("read_only_paths", step.kwargs.get("read_only_paths"))
    writable_paths = step.context.get("writable_paths", step.kwargs.get("writable_paths"))
    memory_limit_mib = step.context.get("memory_limit_mib", step.kwargs.get("memory_limit_mib"))
    if read_only_paths is None and writable_paths is None:
        if memory_limit_mib is not None:
            raise ValueError("memory_limit_mib requires read_only_paths or writable_paths to enable the sandbox")
        return None
    return create_file_sandbox_backend(
        cwd=step.workspace_path,
        read_only_paths=[] if read_only_paths is None else read_only_paths,
        writable_paths=[] if writable_paths is None else writable_paths,
        memory_limit_mib=memory_limit_mib,
    )


def python_sandbox_for_step(step: BaseStep) -> FileSandboxBackend:
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
