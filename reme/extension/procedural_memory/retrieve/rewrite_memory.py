"""Memory rewriting operation module.

This module provides functionality to rewrite and format retrieved memories
into context messages that can be used by LLMs for task completion.
"""

import json
import re
from typing import List

from loguru import logger

from ....core.enumeration import Role
from ....core.op import BaseOp
from ....core.schema.message import Message
from ....core.schema.memory_node import MemoryNode


class RewriteMemory(BaseOp):
    """Generate and rewrite context messages from reranked experiences.

    This operation takes reranked memories and formats them into context messages
    that can be used by LLMs. It optionally uses LLM-based rewriting to make
    the context more relevant and actionable for the current task.
    """

    async def execute(self):
        """Execute the memory rewrite operation.

        Retrieves memories from context metadata, formats them, and optionally
        rewrites them using LLM to make them more relevant for the current query.
        Stores the rewritten context in the response answer field.
        """
        memory_list: List[MemoryNode] = self.context.response.metadata["memory_list"]
        query: str = self.context.query
        messages: List[Message] = [Message(**x) if isinstance(x, dict) else x for x in self.context.get("messages", [])]

        if not memory_list:
            logger.info("No reranked memories to rewrite")
            self.context.response.answer = ""
            return

        logger.info(f"Generating context from {len(memory_list)} memories")

        # Generate initial context message
        rewritten_memory = await self._generate_context_message(query, messages, memory_list)

        # Store results in context
        self.context.response.answer = rewritten_memory
        self.context.response.metadata["memory_list"] = [memory.model_dump() for memory in memory_list]

    async def _generate_context_message(self, query: str, messages: List[Message], memories: List[MemoryNode]) -> str:
        """Generate context message from retrieved memories.

        Args:
            query: The current query string.
            messages: List of conversation messages for context.
            memories: List of retrieved memories to format.

        Returns:
            Formatted context string, optionally rewritten by LLM.
        """
        if not memories:
            return ""

        try:
            logger.info("memories")
            # Format retrieved memories
            formatted_memories = self._format_memories_for_context(memories)

            if self.context.get("enable_llm_rewrite", False):
                context_content = await self._rewrite_context(query, formatted_memories, messages)
            else:
                context_content = formatted_memories

            return context_content

        except Exception as e:
            logger.error(f"Error generating context message: {e}")
            return self._format_memories_for_context(memories)

    async def _rewrite_context(self, query: str, context_content: str, messages: List[Message]) -> str:
        """LLM-based context rewriting to make experiences more relevant and actionable.

        Args:
            query: The current query string.
            context_content: The formatted context content to rewrite.
            messages: List of conversation messages for additional context.

        Returns:
            Rewritten context string optimized for the current task.
        """
        if not context_content:
            return context_content

        try:
            # Extract current context
            current_context = self._extract_context(messages)

            prompt = self.prompt_format(
                prompt_name="memory_rewrite_prompt",
                current_query=query,
                current_context=current_context,
                original_context=context_content,
            )

            response = await self.llm.chat(messages=[Message(role=Role.USER, content=prompt)])

            # Extract rewritten context
            rewritten_context = self._parse_json_response(response.content, "rewritten_context")

            if rewritten_context and rewritten_context.strip():
                logger.info("Context successfully rewritten for current task")
                return rewritten_context.strip()

            return context_content

        except Exception as e:
            logger.error(f"Error in context rewriting: {e}")
            return context_content

    @staticmethod
    def _format_memories_for_context(memories: List[MemoryNode]) -> str:
        """Format memories for context generation.

        Args:
            memories: List of memories to format.

        Returns:
            Formatted string containing all memories with their conditions and content.
        """
        formatted_memories = []

        for i, memory in enumerate(memories, 1):
            condition = memory.when_to_use
            memory_content = memory.content
            memory_text = f"Memory {i} :\n When to use: {condition}\n Content: {memory_content}\n"

            formatted_memories.append(memory_text)

        return "\n".join(formatted_memories)

    @staticmethod
    def _extract_context(messages: List[Message]) -> str:
        """Extract relevant context from messages.

        Args:
            messages: List of conversation messages.

        Returns:
            Formatted string containing recent conversation context.
        """
        if not messages:
            return ""

        context_parts = []

        # Add recent messages if available
        recent_messages = messages[-3:]  # Last 3 messages
        message_summaries = []
        for message in recent_messages:
            content = message.content[:300] + "..." if len(message.content) > 300 else message.content
            message_summaries.append(f"- {message.role.value}: {content}")

        if message_summaries:
            context_parts.append("Recent conversation:\n" + "\n".join(message_summaries))

        return "\n\n".join(context_parts)

    @staticmethod
    def _parse_json_response(response: str, key: str) -> str:
        """Parse JSON response to extract specific key.

        Args:
            response: The response string that may contain JSON.
            key: The key to extract from the JSON object.

        Returns:
            The value associated with the key, or the response string if parsing fails.
        """
        try:
            # Try to extract JSON blocks
            json_pattern = r"```json\s*([\s\S]*?)\s*```"
            json_blocks = re.findall(json_pattern, response)

            if json_blocks:
                parsed = json.loads(json_blocks[0])
                if isinstance(parsed, dict) and key in parsed:
                    return parsed[key]

            # Fallback: try to parse the entire response as JSON
            parsed = json.loads(response)
            if isinstance(parsed, dict) and key in parsed:
                return parsed[key]

        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON response for key '{key}', using raw response")
            # If JSON parsing fails, return the response as-is for fallback
            return response.strip()

        return ""
