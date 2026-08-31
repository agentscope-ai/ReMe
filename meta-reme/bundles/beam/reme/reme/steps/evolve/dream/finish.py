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
        self.logger.info(
            f"[{self.name}] start changed={len(state.changed_paths)} failed_paths={len(state.failed_paths)} "
            f"checkpoint={len(checkpoint)} interest_paths={len(interest_paths)} day_indexes={len(day_index_paths)} "
            f"deleted={len(state.deleted_paths)} persist={self.persist}",
        )
        upserts = self._nodes(workspace, upsert_paths)
        if upserts:
            self.logger.info(f"[{self.name}] catalog upsert start nodes={len(upserts)}")
            await self.file_catalog.upsert(upserts)
            self.logger.info(f"[{self.name}] catalog upsert done nodes={len(upserts)}")
        if self.persist and (upserts or state.deleted_paths):
            self.logger.info(
                f"[{self.name}] catalog dump start upserts={len(upserts)} deleted={len(state.deleted_paths)}",
            )
            await self.file_catalog.dump()
            self.logger.info(f"[{self.name}] catalog dump done")

        state.checkpoint_paths = [n.path for n in upserts if n.path in checkpoint]
        state.summary = render_summary(state)
        store_state(self, state)
        self.context.response.success = not state.failed_units and not state.errors
        self.context.response.answer = state.summary
        self.logger.info(
            f"[{self.name}] finish success={self.context.response.success} "
            f"checkpointed={len(state.checkpoint_paths)} failed_units={len(state.failed_units)} "
            f"errors={len(state.errors)}",
        )
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
    """Render a concise user-facing summary."""
    interest_paths = state.interests_paths or ([state.interests_path] if state.interests_path else [])
    dates = ", ".join(state.dates or [state.date])
    lines = [
        "AutoDream completed",
        "",
        f"- Date: {state.date}",
        f"- Scan window: {dates}",
        (
            f"- Files: {state.files_scanned} scanned, {state.files_changed} changed, "
            f"{state.files_unchanged} unchanged, {state.files_deleted} deleted"
        ),
        f"- Extracted: {len(state.units)} unit(s), {len(state.topics)} topic candidate(s)",
        f"- Integrated: {len(state.integrate_results)} ok, {len(state.failed_units)} failed",
        f"- Topics: {state.topics_written} written" + (f" to {', '.join(interest_paths)}" if interest_paths else ""),
        f"- Catalog: checkpointed {len(state.checkpoint_paths)} changed path(s)",
    ]
    if state.nodes_created:
        lines.append(f"- Created: {', '.join(state.nodes_created)}")
    if state.nodes_updated:
        lines.append(f"- Updated: {', '.join(state.nodes_updated)}")
    if state.integrate_results:
        lines.extend(["", "Changes:"])
        for item in state.integrate_results:
            action = str(item.get("action") or "").strip() or "UPDATED"
            target = str(item.get("target_path") or "").strip() or "(unknown target)"
            note = str(item.get("note") or "").strip()
            suffix = f": {note}" if note else ""
            lines.append(f"- [{target}][{action}]{suffix}")
    if state.failed_paths:
        lines.append(f"- Failed paths: {', '.join(state.failed_paths)}")
    if state.errors:
        lines.append(f"- Errors: {'; '.join(state.errors)}")
    return "\n".join(lines)
