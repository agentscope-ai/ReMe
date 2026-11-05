"""Shared access to application-lifetime tool context state."""

from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    from ...components import ApplicationContext


class _ToolContextMixin:
    """Share storage access while leaving each step's state format independent."""

    TOOL_CONTEXTS_KEY: Final[str] = "tool_contexts"

    if TYPE_CHECKING:
        app_context: "ApplicationContext | None"
        kwargs: dict[str, Any]

    def _tool_context_store(self, tool_context_id: str) -> dict:
        if self.app_context is not None:
            contexts = self.app_context.metadata.setdefault(self.TOOL_CONTEXTS_KEY, {})
        else:
            contexts = self.kwargs.setdefault(self.TOOL_CONTEXTS_KEY, {})
        return contexts.setdefault(tool_context_id, {})
