"""Comparative extraction operation for task memory generation.

This module provides operations to extract comparative task memories by comparing
different trajectories with varying scores or success/failure outcomes.
"""

from typing import List, Tuple, Optional

from loguru import logger

from ....core.enumeration import MemoryType, Role
from ....core.op import BaseOp
from ....core.schema.memory_node import MemoryNode
from ....core.schema.message import Message, Trajectory
from ..utils import (
    merge_messages_content,
    parse_json_experience_response,
)


class ComparativeExtraction(BaseOp):
    """Extract comparative task memories by comparing different scoring trajectories.

    This operation performs two types of comparisons:
    1. Soft comparison: Compares highest vs lowest scoring trajectories
    2. Hard comparison: Compares similar success vs failure step sequences

    The extracted memories help identify what makes some trajectories more successful
    than others.
    """

    async def execute(self):
        """Extract comparative task memories by comparing different scoring trajectories"""
        all_trajectories: List[Trajectory] = self.context.get("all_trajectories", [])
        success_trajectories: List[Trajectory] = self.context.get("success_trajectories", [])
        failure_trajectories: List[Trajectory] = self.context.get("failure_trajectories", [])

        comparative_task_memories = []

        # Soft comparison: highest score vs lowest score
        if self.context.get("enable_soft_comparison", True) and len(all_trajectories) >= 2:
            highest_traj, lowest_traj = self._find_highest_lowest_scoring_trajectories(all_trajectories)
            if highest_traj and lowest_traj and highest_traj.score > lowest_traj.score:
                logger.info(
                    f"Extracting soft comparative task memories: "
                    f"highest ({highest_traj.score:.2f}) vs lowest ({lowest_traj.score:.2f})",
                )
                soft_task_memories = await self._extract_soft_comparative_task_memory(highest_traj, lowest_traj)
                comparative_task_memories.extend(soft_task_memories)

        # Hard comparison: success vs failure (if similarity search is enabled)
        if self.context.get("enable_similarity_comparison", True) and success_trajectories and failure_trajectories:
            similar_pairs = self._find_similar_step_sequences(success_trajectories, failure_trajectories)
            logger.info(f"Found {len(similar_pairs)} similar pairs for hard comparison")

            for success_steps, failure_steps, similarity_score in similar_pairs:
                hard_task_memories = await self._extract_hard_comparative_task_memory(
                    success_steps,
                    failure_steps,
                    similarity_score,
                )
                comparative_task_memories.extend(hard_task_memories)

        logger.info(f"Extracted {len(comparative_task_memories)} comparative task memories")

        # Add task memories to context
        self.context.comparative_task_memories = comparative_task_memories

    @staticmethod
    def _find_highest_lowest_scoring_trajectories(trajectories: List[Trajectory]) -> Tuple[
        Optional[Trajectory],
        Optional[Trajectory],
    ]:
        """Find the highest and lowest scoring trajectories"""
        if len(trajectories) < 2:
            return None, None

        # Filter trajectories with valid scores
        valid_trajectories = [traj for traj in trajectories if traj.score is not None]

        if len(valid_trajectories) < 2:
            logger.warning("Not enough trajectories with valid scores for comparison")
            return None, None

        # Sort by score
        sorted_trajectories = sorted(valid_trajectories, key=lambda x: x.score, reverse=True)

        highest_traj = sorted_trajectories[0]
        lowest_traj = sorted_trajectories[-1]

        return highest_traj, lowest_traj

    @staticmethod
    def _get_trajectory_score(trajectory: Trajectory) -> Optional[float]:
        """Get trajectory score"""
        return trajectory.score

    async def _extract_soft_comparative_task_memory(
        self,
        higher_traj: Trajectory,
        lower_traj: Trajectory,
    ) -> List[MemoryNode]:
        """Extract soft comparative task memory (high score vs low score)"""
        higher_steps = self._get_trajectory_steps(higher_traj)
        lower_steps = self._get_trajectory_steps(lower_traj)
        higher_score = self._get_trajectory_score(higher_traj)
        lower_score = self._get_trajectory_score(lower_traj)

        prompt = self.prompt_format(
            prompt_name="soft_comparative_step_task_memory_prompt",
            higher_steps=merge_messages_content(higher_steps),
            lower_steps=merge_messages_content(lower_steps),
            higher_score=f"{higher_score:.2f}",
            lower_score=f"{lower_score:.2f}",
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

    async def _extract_hard_comparative_task_memory(
        self,
        success_steps: List[Message],
        failure_steps: List[Message],
        similarity_score: float,
    ) -> List[MemoryNode]:
        """Extract hard comparative task memory (success vs failure)"""
        prompt = self.prompt_format(
            prompt_name="hard_comparative_step_task_memory_prompt",
            success_steps=merge_messages_content(success_steps),
            failure_steps=merge_messages_content(failure_steps),
            similarity_score=similarity_score,
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

    @staticmethod
    def _get_trajectory_steps(trajectory: Trajectory) -> List[Message]:
        """Get trajectory steps, prioritizing segmented steps"""
        if hasattr(trajectory, "segments") and trajectory.segments:
            # If there are segments, merge all segments
            all_steps = []
            for segment in trajectory.segments:
                all_steps.extend(segment)
            return all_steps
        else:
            return trajectory.messages

    def _find_similar_step_sequences(
        self,
        success_trajectories: List[Trajectory],
        failure_trajectories: List[Trajectory],
    ) -> List[Tuple[List[Message], List[Message], float]]:
        """Find similar step sequences for comparison"""
        try:
            similar_pairs = []

            # Get step sequences
            success_step_sequences = []
            for traj in success_trajectories:
                if hasattr(traj.metadata, "segments") and traj.metadata["segments"]:
                    success_step_sequences.extend(traj.metadata["segments"])
                else:
                    success_step_sequences.append(traj.messages)

            failure_step_sequences = []
            for traj in failure_trajectories:
                if hasattr(traj.metadata, "segments") and traj.metadata["segments"]:
                    failure_step_sequences.extend(traj.metadata["segments"])
                else:
                    failure_step_sequences.append(traj.messages)

            # Limit comparison count to avoid computational overload
            max_sequences = self.context.get("max_similarity_sequences", 5)
            success_step_sequences = success_step_sequences[:max_sequences]
            failure_step_sequences = failure_step_sequences[:max_sequences]

            if not success_step_sequences or not failure_step_sequences:
                return []

            # Generate text representation for embedding
            success_texts = [merge_messages_content(seq) for seq in success_step_sequences]
            failure_texts = [merge_messages_content(seq) for seq in failure_step_sequences]

            # Get embedding vectors
            if (
                hasattr(self, "vector_store")
                and self.vector_store
                and hasattr(
                    self.vector_store,
                    "embedding_model",
                )
            ):
                success_embeddings = self.vector_store.embedding_model.get_embeddings(success_texts)
                failure_embeddings = self.vector_store.embedding_model.get_embeddings(failure_texts)

                # Calculate similarity and find most similar pairs
                similarity_threshold = self.context.get("similarity_threshold", 0.5)

                for i, s_emb in enumerate(success_embeddings):
                    for j, f_emb in enumerate(failure_embeddings):
                        similarity = self._calculate_cosine_similarity(s_emb, f_emb)

                        if similarity > similarity_threshold:
                            similar_pairs.append(
                                (
                                    success_step_sequences[i],
                                    failure_step_sequences[j],
                                    similarity,
                                ),
                            )

                # Return top most similar pairs
                max_pairs = self.context.get("max_similarity_pairs", 3)
                return sorted(similar_pairs, key=lambda x: x[2], reverse=True)[:max_pairs]

        except Exception as e:
            logger.error(f"Error finding similar step sequences: {e}")

        return []

    @staticmethod
    def _calculate_cosine_similarity(embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity"""
        import numpy as np

        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)
