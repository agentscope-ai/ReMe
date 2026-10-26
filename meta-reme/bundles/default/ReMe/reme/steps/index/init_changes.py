"""One-shot change producer: diff watched files against file_store/file_catalog."""

from pathlib import Path
from typing import Iterable

from ._change_batch import coalesce_changes
from ._watch_rules import WatchRule, build_context_watch_rules, collect_existing
from ..base_step import BaseStep
from ...components import R
from ...schema import FileNode


@R.register("init_changes_step")
class InitChangesStep(BaseStep):
    """Scan once, write ``context["changes"]``, then dispatch change handlers."""

    def __init__(
        self,
        monitor_type: str | None = None,
        monitor_name: str = "default",
        store: str | None = None,
        recursive: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.monitor_type = monitor_type or store
        self.monitor_name = monitor_name
        self.recursive = recursive
        if self.monitor_type in {"file_store", "file_catalog"}:
            self.kwargs.setdefault(self.monitor_type, monitor_name)

    def _get_watch_rules(self) -> list[WatchRule]:
        assert self.context is not None
        app_config = self.app_context.app_config if self.app_context else None
        return build_context_watch_rules(app_config, self.workspace_path, self.context)

    async def _load_indexed_nodes(self) -> Iterable[FileNode]:
        if self.monitor_type == "file_store":
            return await self.file_store.get_nodes()
        if self.monitor_type == "file_catalog":
            if self.file_catalog is None:
                raise RuntimeError("file_catalog is not initialized!")
            return await self.file_catalog.get_nodes()
        raise ValueError("init_changes_step.monitor_type must be 'file_store' or 'file_catalog'")

    @staticmethod
    def diff(existing: dict[str, float], nodes: Iterable[FileNode], workspace_path: Path) -> tuple[
        list[dict],
        dict[str, int],
    ]:
        """Compute added/modified/deleted vs ``nodes`` and return (changes, counts)."""
        indexed: dict[str, float] = {
            str(Path(n.path) if Path(n.path).is_absolute() else workspace_path / n.path): n.st_mtime for n in nodes
        }
        to_delete = list(indexed.keys() - existing.keys())
        to_add = list(existing.keys() - indexed.keys())
        to_modify = [p for p in existing.keys() & indexed.keys() if existing[p] != indexed[p]]
        changes: list[dict] = (
            [{"change": "added", "path": p} for p in to_add]
            + [{"change": "modified", "path": p} for p in to_modify]
            + [{"change": "deleted", "path": p} for p in to_delete]
        )
        counts = {"added": len(to_add), "modified": len(to_modify), "deleted": len(to_delete)}
        return changes, counts

    async def execute(self):
        assert self.context is not None
        rules = self._get_watch_rules()
        existing = collect_existing(rules, recursive=self.recursive)
        nodes = await self._load_indexed_nodes()
        changes, counts = self.diff(existing, nodes, self.workspace_path)
        changes = coalesce_changes(changes)
        self.context["changes"] = changes
        if changes:
            self.logger.info(f"[{self.name}] scan {self.monitor_type}:{self.monitor_name}: {counts}")
            await self.dispatch_steps(self.dispatch_step_specs, changes=changes)
        else:
            self.logger.info(f"[{self.name}] {self.monitor_type}:{self.monitor_name} is up to date")
        self.context.response.metadata["counts"] = counts
        return self.context.response
