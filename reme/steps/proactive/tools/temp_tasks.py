"""Main-only temporary task creation and explicit action dispatch."""

import asyncio

from ...base_step import BaseStep, Ref
from ...file_io._path import _check_path_permission
from ....components import PaRuntime, R


@R.register("manage_temp_proactive_agent_step")
class ManageTempProactiveAgentStep(BaseStep):
    """Create or manage one-shot tasks through one Main-bound tool."""

    pa: PaRuntime = Ref(PaRuntime, "pa")

    def validate_access(self):
        """Require Main's binding and access to the paths needed by the action."""
        if self.context.get("pa_caller") != "chat":
            raise ValueError("Temporary task management is bound to Main")
        paths = [self.pa.temp_tasks.path]
        if self.context.get("action") != "list":
            paths.append(self.pa.temp_tasks.root_path)
        for path in paths:
            if not _check_path_permission(self.workspace_path, path, self.context.get("_allowed_paths")):
                raise PermissionError("no permission to manage temporary tasks")

    async def execute(self):
        self.validate_access()
        action = self.context.get("action")
        task_id = self.context.get("task_id")
        request_id = self.context.get("request_id")
        options = {name: self.context.get(name) for name in ("query", "run_at", "delay_minutes")}
        if action == "list":
            if any(value is not None for value in (task_id, request_id, *options.values())):
                raise ValueError("list does not accept task or scheduling arguments")
            async with self.pa.temp_tasks.lock:
                tasks = await asyncio.to_thread(self.pa.temp_tasks.load)
            self.context.response.answer = {
                "items": [task.model_dump(mode="json") for task in tasks],
                "total": len(tasks),
            }
            return self.context.response
        if action == "create":
            if task_id is not None:
                raise ValueError("create does not accept task_id")
            task = await self.pa.temp_tasks.create(**options, request_id=request_id)
        else:
            if action not in {"update", "pause", "resume", "cancel"}:
                raise ValueError("action must be list, create, update, pause, resume or cancel")
            if not isinstance(task_id, str) or not task_id.strip():
                raise ValueError("task_id is required for existing tasks")
            if request_id is not None:
                raise ValueError("Only create accepts request_id")
            task = await self.pa.temp_tasks.manage(task_id, action, **options)
        self.context.response.answer = task.model_dump(mode="json")
        return self.context.response
