"""Draft accumulation steps scoped by agent tool context."""

from typing import Final

from ._tool_context import _ToolContextMixin
from ..base_step import BaseStep
from ...components import R


@R.register("add_draft_step")
class AddDraftStep(_ToolContextMixin, BaseStep):
    """Append one draft text item to the current tool context."""

    DRAFTS_KEY: Final[str] = "drafts"

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
