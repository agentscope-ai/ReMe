"""Application-owned Git work serialized with Home preparation and snapshots."""

import asyncio

from reme.components.base_component import BaseComponent

from ..workspace import HomeWorkspace, _initialization_lock, _validate_directory
from . import repository, snapshot
from .models import ARGUMENTS

OPERATIONS = {
    "status": repository.status,
    "list_branches": lambda repo, _record: repository.branches(repo),
    "create_branch": repository.create_branch,
    "switch_branch": repository.switch_branch,
    "merge_branch": repository.merge_branch,
    "commit": repository.commit,
    "diff": repository.diff,
    "list_code_tree": snapshot.list_code_tree,
    "read_code_file": snapshot.read_code_file,
    "search_code": snapshot.search_code,
}


class HomeCode(BaseComponent):
    """Manage only the prepared Home candidate; no remotes or history rewrites."""

    component_type = "home.code"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.workspace = self.bind("default", HomeWorkspace, optional=False)
        self._pending: set[asyncio.Task] = set()

    async def _close(self):
        if self._pending:
            await asyncio.gather(*self._pending, return_exceptions=True)

    async def run(self, operation: str, **kwargs) -> dict:
        """Validate before dispatch, and finish owned Git work before propagating cancellation."""
        if not self.is_started:
            raise RuntimeError("Start the Home application before managing candidate code")
        if operation not in OPERATIONS:
            raise ValueError(f"Unknown code operation: {operation}")
        arguments = ARGUMENTS[operation].model_validate(kwargs).model_dump()
        task = asyncio.create_task(asyncio.to_thread(self._run, operation, arguments))
        self._pending.add(task)
        cancelled = False
        try:
            while True:
                try:
                    result = await asyncio.shield(task)
                    if cancelled:
                        raise asyncio.CancelledError
                    return result
                except asyncio.CancelledError:
                    if task.done():
                        raise
                    cancelled = True
                except Exception:
                    if cancelled:
                        raise asyncio.CancelledError from None
                    raise
        finally:
            self._pending.discard(task)

    def _run(self, operation: str, arguments: dict) -> dict:
        """Hold the same cross-process lock used when evaluations snapshot HEAD."""
        _validate_directory(self.workspace_path.resolve(), self.workspace.state_path)
        with _initialization_lock(self.workspace.state_path / "workspace.lock"):
            repo, record = repository.validate_repository(self.workspace.home_path)
            return OPERATIONS[operation](repo, record, **arguments)
