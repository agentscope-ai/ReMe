"""Success extraction operation for task memory generation.

This module provides operations to extract task memories from successful
trajectories, identifying patterns and strategies that lead to success.
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


class SuccessExtraction(BaseOp):
    """Extract task memories from successful trajectories.

    This operation analyzes successful trajectories (or their segments) to
    extract reusable patterns, strategies, and best practices that can be
    applied to similar future tasks.
    """

    async def execute(self):
        """Extract task memories from successful trajectories"""
        success_trajectories: List[Trajectory] = self.context.success_trajectories

        if not success_trajectories:
            logger.info("No success trajectories found for extraction")
            return

        logger.info(f"Extracting task memories from {len(success_trajectories)} successful trajectories")

        success_task_memories = []

        # Process trajectories
        for trajectory in success_trajectories:
            if "segments" in trajectory.metadata:
                # Process segmented step sequences
                for segment in trajectory.metadata["segments"]:
                    task_memories = await self._extract_success_task_memory_from_steps(segment, trajectory)
                    success_task_memories.extend(task_memories)
            else:
                # Process entire trajectory
                task_memories = await self._extract_success_task_memory_from_steps(trajectory.messages, trajectory)
                success_task_memories.extend(task_memories)

        logger.info(f"Extracted {len(success_task_memories)} success task memories")

        # Add task memories to context
        self.context.success_task_memories = success_task_memories

    async def _extract_success_task_memory_from_steps(
        self,
        steps: List[Message],
        trajectory: Trajectory,
    ) -> List[MemoryNode]:
        """Extract task memory from successful step sequences"""
        step_content = merge_messages_content(steps)
        context = get_trajectory_context(trajectory, steps)

        prompt = self.prompt_format(
            prompt_name="success_step_task_memory_prompt",
            query=trajectory.metadata.get("query", ""),
            step_sequence=step_content,
            context=context,
            outcome="successful",
        )

        def parse_task_memories(message: Message) -> list[MemoryNode]:
            task_memories_data = parse_json_experience_response(message.content)  # extract content
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
