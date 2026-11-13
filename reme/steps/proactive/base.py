"""Common runtime binding and request checks for PA steps."""

from pathlib import Path

from ..base_step import BaseStep, Ref
from ..file_io._path import _check_path_permission
from ...components import PaRuntime


class BasePaStep(BaseStep):
    """Bind one runtime and enforce request and workspace boundaries."""

    pa: PaRuntime = Ref(PaRuntime, "pa")

    def require_paths(self, *paths: Path) -> None:
        """Reject the entire operation before accessing an unauthorized path."""
        if any(
            not _check_path_permission(self.workspace_path, path, self.context.get("_allowed_paths")) for path in paths
        ):
            raise PermissionError("no permission to access PA files")

    def _validate(self, *parameters: str) -> None:
        """Reject unsupported request fields before invoking a PA domain."""
        if self.app_context is None:
            raise RuntimeError("PA jobs require an ApplicationContext")
        unexpected = self.context.data.keys() - {*parameters, "metadata"}
        if unexpected:
            raise ValueError(f"Unexpected PA parameters: {', '.join(sorted(unexpected))}")
