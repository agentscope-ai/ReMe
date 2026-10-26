"""Draft accumulation steps scoped by agent tool context."""

from typing import Final

from ..base_step import BaseStep
from ...components import R


@R.register("add_draft_step")
class AddDraftStep(BaseStep):
    """Append one draft text item to the current tool context."""

    TOOL_CONTEXTS_KEY: Final[str] = "tool_contexts"
    DRAFTS_KEY: Final[str] = "drafts"

    def _tool_context_store(self, tool_context_id: str) -> dict:
        if self.app_context is not None:
            contexts = self.app_context.metadata.setdefault(self.TOOL_CONTEXTS_KEY, {})
        else:
            contexts = self.kwargs.setdefault(self.TOOL_CONTEXTS_KEY, {})
        return contexts.setdefault(tool_context_id, {})

    async def execute(self):
        assert self.context is not None
        text = self.context.get("text", "")
        tool_context_id: str = (self.context.get("tool_context_id", "") or "").strip()

        if not tool_context_id:
            self.context.response.success = False
            self.context.response.answer = "Error: tool_context_id is required"
            return self.context.response
        if text is None or not isinstance(text, str):
            self.context.response.success = False
            self.context.response.answer = "Error: text must be a string"
            return self.context.response

        store = self._tool_context_store(tool_context_id)
        drafts = store.setdefault(self.DRAFTS_KEY, [])
        drafts.append(text)

        self.context.response.answer = text
        self.context.response.metadata["draft_count"] = len(drafts)
        return self.context.response


@R.register("read_all_draft_step")
class ReadAllDraftStep(AddDraftStep):
    """Read all draft text items for the current tool context."""

    async def execute(self):
        assert self.context is not None
        tool_context_id: str = (self.context.get("tool_context_id", "") or "").strip()

        if not tool_context_id:
            self.context.response.success = False
            self.context.response.answer = "Error: tool_context_id is required"
            return self.context.response

        store = self._tool_context_store(tool_context_id)
        drafts = store.setdefault(self.DRAFTS_KEY, [])
        self.context.response.answer = "\n".join(drafts)
        self.context.response.metadata["draft_count"] = len(drafts)
        return self.context.response
