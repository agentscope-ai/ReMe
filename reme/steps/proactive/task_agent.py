"""Execute a task by collecting and interpreting evidence with an agent."""

from .base_pa_step import BasePaStep
from ...components import R


@R.register("task_agent_step")
class TaskAgentStep(BasePaStep):
    """Answer a concrete query in a fresh session using progressive resource reading."""

    async def execute(self):
        query = self.context.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if self.app_context is None or self.agent_wrapper is None:
            raise RuntimeError("task_agent requires an ApplicationContext and agent_wrapper")
        memory_path = await self._ensure_memory()
        reply = await self.agent_wrapper.reply(
            query.strip(),
            system_prompt=self.prompt_format(
                "system_prompt",
                now=self._now().isoformat(),
                workspace_dir=self.workspace_path,
                memory_path=memory_path,
            ),
            job_tools=list(self.kwargs.get("job_tools") or []),
        )
        if reply["status"] != "completed":
            raise RuntimeError(reply.get("error") or f"Task agent {reply['status']}")
        self.context.response.answer = reply["result"]
        self.context.response.metadata.update(
            session_id=reply["session_id"],
            status=reply["status"],
            usage=reply.get("usage"),
        )
        return self.context.response
