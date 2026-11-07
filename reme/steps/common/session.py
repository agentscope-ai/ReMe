"""Backend-neutral session jobs; unsupported wrappers fail explicitly."""

from ..base_step import BaseStep
from ...components import R


class _SessionStep(BaseStep):
    """Resolve the configured wrapper and return a typed session result as JSON."""

    operation = ""

    async def execute(self):
        assert self.context is not None
        if self.agent_wrapper is None:
            raise RuntimeError("Session jobs require an agent_wrapper")
        session_id = self.context.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id is required")
        kwargs = {}
        if self.operation == "append_session":
            kwargs = {
                "content": self.context.get("content"),
                "message_id": self.context.get("message_id"),
                "timeout": float(self.context.get("wait_timeout", 30)),
            }
        try:
            result = await getattr(self.agent_wrapper, self.operation)(session_id, **kwargs)
        except NotImplementedError:
            self.context.response.metadata["code"] = "session_not_supported"
            raise
        self.context.response.answer = result.model_dump(mode="json")
        return self.context.response


@R.register("session_read_step")
class SessionReadStep(_SessionStep):
    """Read ordered session history."""

    operation = "read_session"


@R.register("session_status_step")
class SessionStatusStep(_SessionStep):
    """Inspect session activity without waiting behind generation."""

    operation = "get_session_status"


@R.register("session_append_step")
class SessionAppendStep(_SessionStep):
    """Append a proactive assistant message after the current turn finishes."""

    operation = "append_session"
