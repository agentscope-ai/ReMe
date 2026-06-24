"""Dream catalog persistence step."""

from pathlib import Path

from ...base_step import BaseStep
from ....components import R
from ....schema import DreamState, FileNode
from .utils import state_from_context, store_state, workspace_dir


@R.register("dream_finish_step")
class DreamFinishStep(BaseStep):
    """Persist dream catalog and render final auto-dream response."""

    def __init__(self, persist: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.persist = persist

    async def execute(self):
        assert self.context is not None
        state = state_from_context(self)
        workspace = Path(state.workspace).resolve() if state.workspace else workspace_dir(self)
        if self.file_catalog is None:
            raise RuntimeError("dream_finish_step requires file_catalog")

        checkpoint = [p for p in state.changed_paths if p not in set(state.failed_paths)]
        day_index_paths = [f"{state.daily_dir}/{day}.md" for day in (state.dates or [state.date]) if day]
        interest_paths = state.interests_paths or ([state.interests_path] if state.interests_path else [])
        upsert_paths = checkpoint + [p for p in [*interest_paths, *day_index_paths] if p]
        upserts = self._nodes(workspace, upsert_paths)
        if upserts:
            await self.file_catalog.upsert(upserts)
        if self.persist and (upserts or state.deleted_paths):
            await self.file_catalog.dump()

        state.checkpoint_paths = [n.path for n in upserts if n.path in checkpoint]
        state.summary = render_summary(state)
        store_state(self, state)
        self.context.response.success = not state.failed_units and not state.errors
        self.context.response.answer = state.summary
        return self.context.response

    @staticmethod
    def _nodes(workspace: Path, paths: list[str]) -> list[FileNode]:
        out: list[FileNode] = []
        for rel in paths:
            try:
                out.append(FileNode(path=rel, st_mtime=(workspace / rel).stat().st_mtime))
            except OSError:
                continue
        return out


def render_summary(state: DreamState) -> str:
    """Render summary."""
    interest_paths = state.interests_paths or ([state.interests_path] if state.interests_path else [])
    lines = [
        f"[AutoDream] date={state.date} dates={','.join(state.dates or [state.date])} "
        f"scanned={state.files_scanned} changed={state.files_changed} "
        f"unchanged={state.files_unchanged} deleted={state.files_deleted}",
        f"  - extract: {len(state.units)} unit(s), {len(state.topics)} topic(s)",
        f"  - integrate: {len(state.integrate_results)} ok, {len(state.failed_units)} failed",
        f"  - topics: {state.topics_written} written" + (f" to {', '.join(interest_paths)}" if interest_paths else ""),
        f"  - catalog: checkpointed {len(state.checkpoint_paths)} changed path(s)",
    ]
    if state.failed_paths:
        lines.append(f"  - failed paths: {', '.join(state.failed_paths)}")
    if state.errors:
        lines.append(f"  - errors: {'; '.join(state.errors)}")
    return "\n".join(lines)
