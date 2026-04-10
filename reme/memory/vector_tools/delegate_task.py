"""Hands-off tool to delegate memory tasks to specific agents"""

from loguru import logger

from .base_memory_tool import BaseMemoryTool
from ..vector_based import BaseMemoryAgent
from ...core.enumeration import MemoryType
from ...core.schema import ToolCall


class DelegateTask(BaseMemoryTool):
    """Tool to delegate memory tasks to appropriate memory agents"""

    def __init__(self, memory_agents: list[BaseMemoryAgent] = None, **kwargs):
        kwargs["enable_multiple"] = True
        kwargs["sub_ops"] = memory_agents or []
        super().__init__(**kwargs)
        self.sub_ops: list[BaseMemoryAgent] = [a for a in self.sub_ops if isinstance(a, BaseMemoryAgent)]
        assert all(a.memory_type is not None for a in self.sub_ops)

    @property
    def memory_agent_dict(self) -> dict[MemoryType, BaseMemoryAgent]:
        """Map memory types to their corresponding agents"""
        return {a.memory_type: a for a in self.sub_ops}

    def _build_multiple_tool_call(self) -> ToolCall:
        """Build and return the multiple tool call schema"""
        return ToolCall(
            **{
                "description": "Delegate tasks to appropriate agents.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tasks": {
                            "type": "array",
                            "description": "List of tasks to delegate to specific memory agents",
                            "items": {
                                "type": "object",
                                "description": "A task item",
                                "properties": {
                                    "memory_target": {
                                        "type": "string",
                                        "description": "The memory_target identifier to "
                                        "delegate to the corresponding agent",
                                    },
                                },
                                "required": ["memory_target"],
                            },
                        },
                    },
                    "required": ["tasks"],
                },
            },
        )

    async def execute(self):
        # Deduplicate and validate tasks
        tasks = self.context.get("tasks", [])
        memory_target_tasks = sorted(set(task["memory_target"] for task in tasks))

        # Submit memory_target_tasks to agents
        agent_list: list[BaseMemoryAgent] = []
        for i, memory_target in enumerate(memory_target_tasks):
            if memory_target not in self.memory_target_type_mapping:
                logger.warning(f"Memory target {memory_target} not found in memory_target_type_mapping")
                continue

            memory_type = self.memory_target_type_mapping[memory_target]
            agent = self.memory_agent_dict[memory_type].copy()
            agent_list.append(agent)

            logger.info(f"Task {i}: {memory_type.value} agent for {memory_target}")
            task_kwargs = {"memory_target": memory_target}
            for k in ["query", "messages", "description", "history_node"]:
                if k in self.context:
                    task_kwargs[k] = self.context[k]
            self.submit_async_task(agent.call, service_context=self.service_context, **task_kwargs)
        await self.join_async_tasks()

        # Collect results
        results = []
        for agent in agent_list:
            results.append(f"Task: {agent.memory_target}\n{agent.response.answer}")

        logger.info(f"Completed {len(results)} memory_target(s)")
        return {
            "answer": "\n\n".join(results),
            "agents": agent_list,
        }
