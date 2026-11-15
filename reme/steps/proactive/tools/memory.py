"""Revision-checked writes to the calling Agent's user-owned memory."""

import asyncio
from hashlib import sha256
from ..agent.base import BaseAgentStep
from ...file_io import get_path_lock
from ....components import R
from ....schema.agent_runtime import AgentMemoryRequest, validate_agent_id
from ....utils.async_utils import complete_in_thread


@R.register("agent_memory_step")
class AgentMemoryStep(BaseAgentStep):
    """Keep native command tools away from memory replacement and revision checks."""

    async def execute(self):
        request = AgentMemoryRequest.model_validate(self.parameters())
        runtime = self.agent_runtime
        caller = validate_agent_id(self.context.get("_agent_id"))
        agent_id = request.agent_id or caller
        if caller not in {"chat", agent_id}:
            raise PermissionError("task Agents may access only their own memory")
        if agent_id != "chat":
            await runtime.definition(agent_id)
        path = runtime.agents.path(agent_id) / "MEMORY.md"
        self.require_paths(path)

        def access():
            current = runtime.files.read(path)
            revision = sha256(current.encode("utf-8")).hexdigest()
            if request.action == "write":
                if revision != request.expected_revision:
                    raise ValueError(
                        "Agent memory changed; read it again and merge the latest user edits before saving",
                    )
                runtime.files.write(path, request.content)
                current = request.content
                revision = sha256(current.encode("utf-8")).hexdigest()
            return {
                "path": path.relative_to(runtime.workspace_path).as_posix(),
                "revision": revision,
                **({"content": current} if request.action == "read" else {}),
            }

        async with await get_path_lock(path):
            if request.action == "read":
                result = await asyncio.to_thread(access)
            else:
                result = await complete_in_thread(access)
        self.context.response.answer = result
        return self.context.response
