"""Trajectory preprocessing operation for task memory generation.

This module provides operations to preprocess and classify trajectories
into success and failure categories based on score thresholds.
"""

from typing import Dict, List

from loguru import logger

from ....core.op import BaseOp
from ....core.schema.message import Trajectory


class TrajectoryPreprocess(BaseOp):
    """Preprocess trajectories: validate and classify by success/failure.

    This operation classifies trajectories into success and failure categories
    based on score thresholds, preparing them for downstream memory extraction
    operations.
    """

    async def execute(self):
        """Preprocess trajectories: validate and classify"""
        trajectories: list = self.context.get("trajectories", [])
        trajectories: List[Trajectory] = [Trajectory(**x) if isinstance(x, dict) else x for x in trajectories]

        # Classify trajectories
        classified = self._classify_trajectories(trajectories)
        logger.info(
            f"Classified trajectories - Success: {len(classified['success'])}, "
            f"Failure: {len(classified['failure'])}, All: {len(classified['all'])}",
        )

        # Set context for downstream operators
        self.context.success_trajectories = classified["success"]
        self.context.failure_trajectories = classified["failure"]
        self.context.all_trajectories = classified["all"]

    def _classify_trajectories(self, trajectories: List[Trajectory]) -> Dict[str, List[Trajectory]]:
        """Classify trajectories based on score threshold"""
        success_trajectories = []
        failure_trajectories = []

        success_threshold = self.context.get("success_threshold", 1.0)

        for traj in trajectories:
            is_success = traj.score >= success_threshold

            if is_success:
                success_trajectories.append(traj)
            else:
                failure_trajectories.append(traj)

        return {
            "success": success_trajectories,
            "failure": failure_trajectories,
            "all": trajectories,
        }
