"""Failure extraction operation for task memory generation.

This module provides operations to extract task memories from failed trajectories,
identifying mistakes, pitfalls, and lessons learned from failures.
"""

from typing import List

from loguru import logger

from ....core.enumeration import MemoryType, Role
from ....core.op import BaseOp
from ....core.schema.memory_node import MemoryNode
from ....core.schema.message import Message, Trajectory
from ..utils import (
    get_trajectory_context,
    merge_messages_content,
    parse_json_experience_response,
)


class FailureExtraction(BaseOp):
    """Extract task memories from failed trajectories.

    This operation analyzes failed trajectories (or their segments) to extract
    lessons learned, common mistakes, and anti-patterns that should be avoided
    in similar future tasks.
    """

    async def execute(self):
        """Extract task memories from failed trajectories"""
        failure_trajectories: List[Trajectory] = self.context.get("failure_trajectories", [])

        if not failure_trajectories:
            logger.info("No failure trajectories found for extraction")
            return

        logger.info(f"Extracting task memories from {len(failure_trajectories)} failed trajectories")

        failure_task_memories = []

        # Process trajectories
        for trajectory in failure_trajectories:
            if "segments" in trajectory.metadata:
                # Process segmented step sequences
                for segment in trajectory.metadata["segments"]:
                    task_memories = await self._extract_failure_task_memory_from_steps(segment, trajectory)
                    failure_task_memories.extend(task_memories)
            else:
                # Process entire trajectory
                task_memories = await self._extract_failure_task_memory_from_steps(trajectory.messages, trajectory)
                failure_task_memories.extend(task_memories)

        logger.info(f"Extracted {len(failure_task_memories)} failure task memories")

        # Add task memories to context
        self.context.failure_task_memories = failure_task_memories

    async def _extract_failure_task_memory_from_steps(
        self,
        steps: List[Message],
        trajectory: Trajectory,
    ) -> List[MemoryNode]:
        """Extract task memory from failed step sequences"""
        step_content = merge_messages_content(steps)
        context = get_trajectory_context(trajectory, steps)

        prompt = self.prompt_format(
            prompt_name="failure_step_task_memory_prompt",
            query=trajectory.metadata.get("query", ""),
            step_sequence=step_content,
            context=context,
            outcome="failed",
        )

        def parse_task_memories(message: Message) -> List[MemoryNode]:
            task_memories_data = parse_json_experience_response(message.content)
            task_memories = []

            for tm_data in task_memories_data:
                task_memory = MemoryNode(
                    memory_type=MemoryType.PROCEDURAL,
                    when_to_use=tm_data.get("when_to_use", tm_data.get("condition", "")),
                    content=tm_data.get("experience", ""),
                    author=getattr(self.llm, "model_name", "system"),
                    metadata=tm_data,
                )
                task_memories.append(task_memory)

            return task_memories

        return await self.llm.chat(
            messages=[Message(role=Role.USER, content=prompt)],
            callback_fn=parse_task_memories,
        )
