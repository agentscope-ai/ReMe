"""Trajectory segmentation operation for task memory generation.

This module provides operations to segment trajectories into meaningful step
sequences that can be used for more granular memory extraction.
"""

import json
import re
from typing import List

from loguru import logger

from ....core.enumeration import Role
from ....core.op import BaseOp
from ....core.schema.message import Message, Trajectory


class TrajectorySegmentation(BaseOp):
    """Segment trajectories into meaningful step sequences.

    This operation uses LLM to identify natural breakpoints in trajectories,
    allowing for more granular analysis and memory extraction from specific
    segments rather than entire trajectories.
    """

    async def execute(self):
        """Segment trajectories into meaningful steps"""
        # Get trajectories from context
        all_trajectories: List[Trajectory] = self.context.get("all_trajectories", [])
        success_trajectories: List[Trajectory] = self.context.get("success_trajectories", [])
        failure_trajectories: List[Trajectory] = self.context.get("failure_trajectories", [])

        if not all_trajectories:
            logger.warning("No trajectories found in context")
            return

        # Determine which trajectories to segment
        target_trajectories = self._get_target_trajectories(
            all_trajectories,
            success_trajectories,
            failure_trajectories,
        )

        # Add segmentation info to trajectories
        segmented_count = 0
        for trajectory in target_trajectories:
            segments = await self._llm_segment_trajectory(trajectory)
            trajectory.metadata["segments"] = segments
            segmented_count += 1

        logger.info(f"Segmented {segmented_count} trajectories")

        # Update context with segmented trajectories

    def _get_target_trajectories(
        self,
        all_trajectories: List[Trajectory],
        success_trajectories: List[Trajectory],
        failure_trajectories: List[Trajectory],
    ) -> List[Trajectory]:
        """Determine which trajectories to segment based on configuration"""
        segment_target = self.context.get("segment_target", "all")

        if segment_target == "success":
            return success_trajectories
        elif segment_target == "failure":
            return failure_trajectories
        else:
            return all_trajectories

    async def _llm_segment_trajectory(self, trajectory: Trajectory) -> List[List[Message]]:
        """Use LLM for trajectory segmentation"""
        trajectory_content = self._format_trajectory_content(trajectory)

        prompt = self.prompt_format(
            prompt_name="step_segmentation_prompt",
            query=trajectory.metadata.get("query", ""),
            trajectory_content=trajectory_content,
            total_steps=len(trajectory.messages),
        )

        def parse_segmentation(message: Message) -> List[List[Message]]:
            content = message.content
            segment_points = self._parse_segmentation_response(content)

            # Segment trajectory based on segmentation points
            segments = []
            start_idx = 0

            for end_idx in segment_points:
                if start_idx < end_idx <= len(trajectory.messages):
                    segments.append(trajectory.messages[start_idx:end_idx])
                    start_idx = end_idx

            # Add remaining steps
            if start_idx < len(trajectory.messages):
                segments.append(trajectory.messages[start_idx:])

            return segments if segments else [trajectory.messages]

        return await self.llm.chat(
            messages=[Message(role=Role.USER, content=prompt)],
            callback_fn=parse_segmentation,
            default_value=[trajectory.messages],
        )

    @staticmethod
    def _format_trajectory_content(trajectory: Trajectory) -> str:
        """Format trajectory content for LLM processing"""
        content = ""
        for i, step in enumerate(trajectory.messages):
            content += f"Step {i + 1} ({step.role.value}):\n{step.content}\n\n"
        return content

    @staticmethod
    def _parse_segmentation_response(response: str) -> List[int]:
        """Parse segmentation response from LLM"""
        segment_points = []

        # Try to extract JSON format
        json_pattern = r"```json\s*([\s\S]*?)\s*```"
        json_blocks = re.findall(json_pattern, response)

        if json_blocks:
            try:
                parsed = json.loads(json_blocks[0])
                if isinstance(parsed, dict) and "segment_points" in parsed:
                    segment_points = parsed["segment_points"]
                elif isinstance(parsed, list):
                    segment_points = parsed
            except json.JSONDecodeError:
                pass

        # Fallback: extract numbers
        if not segment_points:
            numbers = re.findall(r"\b\d+\b", response)
            segment_points = [int(num) for num in numbers if int(num) > 0]

        return sorted(list(set(segment_points)))
