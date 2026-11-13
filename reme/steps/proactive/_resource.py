"""Resolve the temporary-resource root shared by collection and retention."""

from pathlib import Path

from ..base_step import BaseStep
from ..file_io._path import resolve_path
from ...schema import ApplicationConfig


def resource_root(step: BaseStep) -> Path:
    """Require a contained, non-root resource directory without symlink aliases."""
    workspace = step.workspace_path.resolve()
    config = step.app_context.app_config if step.app_context else ApplicationConfig()
    raw = str(config.resource_dir)
    root, error = resolve_path(workspace, raw)
    if error or root == workspace or (workspace / raw).absolute() != root:
        raise ValueError(error or "resource_dir must stay inside the workspace without symlinks")
    return root
