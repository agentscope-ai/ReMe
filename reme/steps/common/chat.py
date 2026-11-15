"""Workspace-aware streaming chat step for the ReMe web interface."""

from ..base_step import BaseStep
from ...components import R
from ...schema import collect_reply


@R.register("chat_step")
class ChatStep(BaseStep):
    """Stream a workspace conversation with read-only-by-default tools and an empty prompt unless configured."""

    READ_ONLY_TOOLS = ["search", "list", "read", "read_image", "frontmatter_read", "stat", "traverse"]

    async def execute(self):
        assert self.context is not None
        query = str(self.context.get("query") or "").strip()
        if not query:
            self.context.response.success = False
            self.context.response.answer = "Skipped: empty query"
            return self.context.response
        if self.agent_wrapper is None:
            raise RuntimeError("chat_step requires an agent_wrapper")

        wrapper_kwargs = {
            "system_prompt": str(self.context.get("system_prompt") or ""),
            "job_tools": self.kwargs.get("job_tools", self.READ_ONLY_TOOLS),
            "builtin_tools": [],
        }
        if session_id := self.context.get("session_id"):
            wrapper_kwargs["resume"] = str(session_id)

        result = await collect_reply(
            self.agent_wrapper.reply_stream(query, **wrapper_kwargs),
            on_chunk=self.context.add_stream_chunk,
        )
        self.context.response.success = result["status"] == "completed" and self.context.stream_error is None
        self.context.response.answer = result["result"]
        self.context.response.metadata.update(result)
        return self.context.response
