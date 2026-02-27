"""Memory reranking operation module.

This module provides functionality to rerank and filter retrieved memories
using LLM-based reranking and score-based filtering to select the most relevant
memories for the current task.
"""

import json
import re
from typing import List

from loguru import logger

from ....core.enumeration import Role
from ....core.op import BaseOp
from ....core.schema.message import Message
from ....core.schema.memory_node import MemoryNode


class RerankMemory(BaseOp):
    """Rerank and filter recalled experiences using LLM and score-based filtering.

    This operation takes recalled memories and applies multiple filtering and
    ranking strategies to select the most relevant memories for the current task.
    It supports LLM-based reranking and score-based filtering.
    """

    async def execute(self):
        """Execute the memory reranking operation.

        Applies LLM-based reranking (optional) and score-based filtering (optional)
        to rerank retrieved memories. Stores the reranked results
        in the context response metadata.
        """
        memory_list: List[MemoryNode] = self.context.response.metadata["memory_list"]
        retrieval_query: str = self.context.query
        enable_llm_rerank = self.context.get("enable_llm_rerank", False)
        enable_score_filter = self.context.get("enable_score_filter", False)
        min_score_threshold = self.context.get("min_score_threshold", 0.3)

        if not memory_list:
            logger.info("No recalled memory_list to rerank")
            return

        logger.info(f"Reranking {len(memory_list)} memories")

        # Step 1: LLM reranking (optional)
        if enable_llm_rerank:
            memory_list = await self._llm_rerank(retrieval_query, memory_list)
            logger.info(f"After LLM reranking: {len(memory_list)} memories")

        # Step 2: Score-based filtering (optional)
        if enable_score_filter:
            memory_list = self._score_based_filter(memory_list, min_score_threshold)
            logger.info(f"After score filtering: {len(memory_list)} memories")

        # Store results in context
        self.context.response.metadata["memory_list"] = memory_list

    async def _llm_rerank(self, query: str, candidates: List[MemoryNode]) -> List[MemoryNode]:
        """LLM-based reranking of candidate experiences.

        Args:
            query: The retrieval query used to rank candidates.
            candidates: List of memory candidates to rerank.

        Returns:
            List of memories reranked by relevance to the query.
        """
        if not candidates:
            return candidates

        # Format candidates for LLM evaluation
        candidates_text = self._format_candidates_for_rerank(candidates)

        prompt = self.prompt_format(
            prompt_name="memory_rerank_prompt",
            query=query,
            candidates=candidates_text,
            num_candidates=len(candidates),
        )

        response = await self.llm.chat(messages=[Message(role=Role.USER, content=prompt)])

        # Parse reranking results
        reranked_indices = self._parse_rerank_response(response.content)

        # Reorder candidates based on LLM ranking
        if reranked_indices:
            reranked_candidates = []
            for idx in reranked_indices:
                if 0 <= idx < len(candidates):
                    reranked_candidates.append(candidates[idx])

            # Add any remaining candidates that weren't explicitly ranked
            ranked_indices_set = set(reranked_indices)
            for i, candidate in enumerate(candidates):
                if i not in ranked_indices_set:
                    reranked_candidates.append(candidate)

            return reranked_candidates

        return candidates

    @staticmethod
    def _score_based_filter(memories: List[MemoryNode], min_score: float) -> List[MemoryNode]:
        """Filter memories based on quality scores.

        Args:
            memories: List of memories to filter.
            min_score: Minimum combined score threshold for filtering.

        Returns:
            List of memories that meet the minimum score threshold.
        """
        filtered_memories = []

        for memory in memories:
            # Get confidence score from metadata
            confidence = memory.metadata.get("confidence", 0.5)
            validation_score = memory.score or 0.5

            # Calculate combined score
            combined_score = (confidence + validation_score) / 2

            if combined_score >= min_score:
                filtered_memories.append(memory)
            else:
                logger.debug(f"Filtered out memory with score {combined_score:.2f}")

        logger.info(f"Score filtering: {len(filtered_memories)}/{len(memories)} memories retained")
        return filtered_memories

    @staticmethod
    def _format_candidates_for_rerank(candidates: List[MemoryNode]) -> str:
        """Format candidates for LLM reranking.

        Args:
            candidates: List of memory candidates to format.

        Returns:
            Formatted string representation of candidates for LLM evaluation.
        """
        formatted_candidates = []

        for i, candidate in enumerate(candidates):
            condition = candidate.when_to_use
            content = candidate.content

            candidate_text = f"Candidate {i}:\n"
            candidate_text += f"Condition: {condition}\n"
            candidate_text += f"Experience: {content}\n"

            formatted_candidates.append(candidate_text)

        return "\n---\n".join(formatted_candidates)

    @staticmethod
    def _parse_rerank_response(response: str) -> List[int]:
        """Parse LLM reranking response to extract ranked indices.

        Args:
            response: The LLM response containing ranked indices.

        Returns:
            List of indices representing the reranked order.
        """
        try:
            # Try to extract JSON format
            json_pattern = r"```json\s*([\s\S]*?)\s*```"
            json_blocks = re.findall(json_pattern, response)

            if json_blocks:
                parsed = json.loads(json_blocks[0])
                if isinstance(parsed, dict) and "ranked_indices" in parsed:
                    return parsed["ranked_indices"]
                elif isinstance(parsed, list):
                    return parsed

            # Try to extract numbers from text
            numbers = re.findall(r"\b\d+\b", response)
            return [int(num) for num in numbers if int(num) < 100]  # Reasonable upper bound

        except Exception as e:
            logger.error(f"Error parsing rerank response: {e}")
            return []
