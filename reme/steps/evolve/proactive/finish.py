"""Proactive finish step: checkpoint only the proactive catalog (F3)."""

from ...base_step import BaseStep
from .._evolve import passthrough_response
from ....components import R
from ....schema import FileNode, ProactiveState
from ..dream.utils import workspace_dir


@R.register("proactive_finish_step")
class ProactiveFinishStep(BaseStep):
    """Upsert this round's changed material into the proactive catalog.

    interests.yaml is deliberately NOT checkpointed (v5 R6): the catalog only
    serves change-detection watermarking and interests.yaml is excluded from
    the material set (INV-11), so no consumer would ever read it back.
    Never calls ``refresh_day_index`` and never touches the dream catalog
    (INV-2). When the short-circuit flag is set, no checkpoint happens so the
    same material stays "changed" for the next round (INV-7).
    """

    def __init__(self, persist: bool = True, skip_key: str = "proactive_skip", **kwargs):
        super().__init__(**kwargs)
        self.persist = persist
        self.skip_key = skip_key

    async def execute(self):
        assert self.context is not None
        if self.context.get(self.skip_key):
            return passthrough_response(self, self.skip_key)
        if self.file_catalog is None:
            raise RuntimeError("proactive_finish_step requires file_catalog")
        raw_state = self.context.get("proactive")
        if not raw_state:
            self.context.response.success = True
            self.context.response.answer = "Skipped finish: no proactive extract state in context"
            return self.context.response
        state = ProactiveState.model_validate(raw_state)
        if state.early_exit:
            self.context.response.success = True
            self.context.response.answer = f"Skipped finish: {state.early_exit}"
            return self.context.response
        ws = workspace_dir(self)

        checkpoint = [rel for rel in state.changed_paths if (ws / rel).is_file()]
        nodes: list[FileNode] = []
        for rel in dict.fromkeys(checkpoint):
            try:
                nodes.append(FileNode(path=rel, st_mtime=(ws / rel).stat().st_mtime))
            except OSError:
                continue
        self.logger.info(f"[{self.name}] start checkpoint={len(nodes)} persist={self.persist}")
        if nodes:
            await self.file_catalog.upsert(nodes)
        if self.persist and nodes:
            await self.file_catalog.dump()

        state.checkpoint_paths = [n.path for n in nodes]
        data = state.model_dump()
        self.context["proactive"] = data
        self.context.response.metadata["proactive"] = data
        self.context.response.success = True
        self.context.response.answer = f"Proactive finished: checkpointed {len(nodes)} path(s)"
        self.logger.info(f"[{self.name}] finish checkpointed={len(nodes)}")
        return self.context.response
