"""Hands-off operation for distributing memory summarization tasks to appropriate agents.

This module provides the HandsOffOp class for distributing memory summarization tasks
to the appropriate memory agents (identity, personal, procedural, tool) based on memory_type,
and executing them in parallel.
"""

import json
from typing import List, Dict

from loguru import logger

from .base_memory_tool_op import BaseMemoryToolOp
from .. import C
from ..agent import BaseMemoryAgentOp
from ..enumeration import MemoryType


@C.register_op()
class HandsOffOp(BaseMemoryToolOp):
    """Operation for distributing memory summarization tasks to appropriate agents.
    
    This tool receives memory_type and memory_target parameters, then distributes
    the summarization tasks to the corresponding memory agents and executes them in parallel.
    """

    def build_input_schema(self) -> dict:
        """Build input schema for single memory task distribution.

        Returns:
            dict: Input schema for distributing a single memory task.
        """
        return {
            "memory_type": {
                "type": "string",
                "description": "memory_type",
                "enum": [
                    MemoryType.IDENTITY.value,
                    MemoryType.PERSONAL.value,
                    MemoryType.PROCEDURAL.value,
                    MemoryType.TOOL.value,
                ],
                "required": True,
            },
            "memory_target": {
                "type": "string",
                "description": "memory_target",
                "required": True,
            },
        }

    def build_multiple_input_schema(self) -> dict:
        """Build input schema for multiple memory task distribution.

        Returns:
            dict: Input schema for distributing multiple memory tasks.
        """
        return {
            "memory_tasks": {
                "type": "array",
                "description": self.get_prompt("memory_tasks"),
                "required": True,
                "items": {
                    "type": "object",
                    "properties": {
                        "memory_type": {
                            "type": "string",
                            "description": "memory_type",
                            "enum": [
                                MemoryType.IDENTITY.value,
                                MemoryType.PERSONAL.value,
                                MemoryType.PROCEDURAL.value,
                                MemoryType.TOOL.value,
                            ],
                        },
                        "memory_target": {
                            "type": "string",
                            "description": "memory_target",
                        },
                    },
                    "required": ["memory_type", "memory_target"],
                },
            },
        }

    async def _build_agent_op_dict(self) -> Dict[MemoryType, BaseMemoryAgentOp]:
        """Build a dictionary mapping memory types to their corresponding agent operations.
        
        Returns:
            Dict[MemoryType, BaseMemoryAgentOp]: Dictionary mapping memory types to agent ops.
        """

        agent_op_dict: Dict[MemoryType, BaseMemoryAgentOp] = {}
        for op in self.ops.values():
            if isinstance(op, BaseMemoryAgentOp) and op.memory_type is not None:
                agent_op_dict[op.memory_type] = op

        return agent_op_dict

    async def async_execute(self):
        """Execute the hands-off operation.

        Distributes memory summarization tasks to appropriate agents based on memory_type
        and executes them in parallel. Supports both single and multiple modes.
        """
        # Build agent operation dictionary
        agent_op_dict = await self._build_agent_op_dict()

        # Collect tasks to execute
        tasks: List[Dict] = []
        if self.enable_multiple:
            memory_tasks: List[dict] = self.context.get("memory_tasks", [])
            for task in memory_tasks:
                memory_type_str = task.get("memory_type", "")
                memory_target = task.get("memory_target", "")
                if memory_type_str:
                    tasks.append({
                        "memory_type": MemoryType(memory_type_str),
                        "memory_target": memory_target,
                    })
        else:
            memory_type_str = self.context.get("memory_type", "")
            memory_target = self.context.get("memory_target", "")
            if memory_type_str:
                tasks.append({
                    "memory_type": MemoryType(memory_type_str),
                    "memory_target": memory_target,
                })

        if not tasks:
            self.set_output("No valid memory tasks to execute.")
            return

        # Submit tasks to corresponding agents in parallel
        op_list = []
        for i, task in enumerate(tasks):
            memory_type: MemoryType = task["memory_type"]
            memory_target: str = task["memory_target"]

            if memory_type not in agent_op_dict:
                logger.warning(f"No agent found for memory_type={memory_type.value}")
                continue

            # Copy the agent op and prepare for execution
            op_copy = agent_op_dict[memory_type].copy()
            op_list.append({
                "op": op_copy,
                "memory_type": memory_type,
                "memory_target": memory_target,
            })

            # Submit async task
            logger.info(f"Task {i}: Submitting {memory_type.value} agent for target={memory_target}")
            self.submit_async_task(
                op_copy.async_call,
                workspace_id=self.context.get("workspace_id", "default"),
                memory_target=memory_target,
                query=self.context.get("query", ""),
                messages=self.context.get("messages", []),
                ref_memory_id=self.context.get("ref_memory_id", ""),
            )

        # Wait for all tasks to complete
        await self.join_async_task()

        # Collect results
        results = []
        for i, op_info in enumerate(op_list):
            op = op_info["op"]
            memory_type = op_info["memory_type"]
            memory_target = op_info["memory_target"]

            result_str = str(op.output)
            results.append({
                "memory_type": memory_type.value,
                "memory_target": memory_target,
                "result": result_str[:200] + ("..." if len(result_str) > 200 else ""),
            })
            logger.info(f"Task {i}: Completed {memory_type.value} agent for target={memory_target}")

        # Format output
        results_str = json.dumps(results, ensure_ascii=False, indent=2)
        self.set_output(f"Successfully executed {len(results)} memory summarization tasks:\n{results_str}")
