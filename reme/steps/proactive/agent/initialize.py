"""Create a task's default requirements through its registered execution Step."""

import asyncio

from ...prompt import PromptHandler
from ....enumeration import ComponentEnum
from ....utils.async_utils import complete_in_thread


async def initialize_task(runtime, app_context, task_description=None):
    """Render adjacent YAML without constructing or retaining an execution Step."""
    if await asyncio.to_thread(runtime.task_path.is_file):
        return runtime.task_path
    if task_description is None:
        config = app_context.app_config.jobs.get(runtime.run_job)
        if config is None or not config.steps:
            raise ValueError(f"Missing execution Job for task initialization: {runtime.run_job}")
        cls = app_context.registry.get(ComponentEnum.STEP, config.steps[0].backend)
        if cls is None:
            raise ValueError("Task execution Step is not registered")
        prompts = PromptHandler(language=app_context.app_config.language)
        for parent in reversed(cls.__mro__):
            prompts.load_prompt_by_class(parent)
        task_description = prompts.get_prompt("default_task")
    if not task_description.strip():
        raise ValueError("Task description must not be blank")
    await complete_in_thread(runtime.files.write, runtime.task_path, task_description, create_only=True)
    return runtime.task_path
