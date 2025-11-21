"""
Context compression module for reducing token usage in conversation contexts using LLM.

This module provides functionality to compress conversation history by using a language
model to generate concise summaries of older messages while preserving recent messages.
This helps manage context window limits while maintaining conversation coherence.

The compression process:
1. Identifies messages that exceed token thresholds
2. Splits messages into groups if needed
3. Uses LLM to generate compressed summaries of older message groups
4. Stores original messages to files for potential retrieval
5. Appends compressed summaries to the system message while preserving recent messages
"""

import json
from pathlib import Path
from typing import List, Tuple
from uuid import uuid4

from flowllm.core.context import C
from flowllm.core.enumeration import Role
from flowllm.core.op import BaseAsyncOp
from flowllm.core.schema import Message
from loguru import logger

from reme_ai.utils import merge_messages_content
from reme_ai.utils.op_utils import extract_xml_tag_content


@C.register_op()
class ContextCompressOp(BaseAsyncOp):
    """
    Context compression operation that uses LLM to reduce token usage.

    When the total token count exceeds the threshold, this operation uses a language
    model to compress older messages into a concise summary while keeping recent
    messages intact. This preserves conversation context while reducing token usage.

    Attributes:
        file_path: Path to the operation file, used for configuration.

    Context Parameters:
        max_total_tokens (int): Maximum token count threshold for compression.
            Defaults to 20000. Does not include keep_recent_count messages or system messages.
        group_token_threshold (int, optional): Maximum token count per compression group.
            If None or 0, all messages are compressed in a single group.
        keep_recent_count (int): Number of recent messages to preserve without compression.
            Defaults to 2. Must be non-negative.
        chat_id (str): Unique identifier for the chat session, used for file naming.
            Defaults to a generated UUID if not provided.
    """

    file_path: str = __file__

    def get_store_path(self, name: str) -> Path:
        """Get the storage path for a given file name.

        Args:
            name: Name of the file to store.

        Returns:
            Path object representing the full path to the storage location.
        """
        return Path(self.context.store_dir) / name

    def _split_messages_by_token_threshold(self, messages: List[Message], token_threshold: int) -> List[List[Message]]:
        """Split messages into groups based on token threshold.

        Messages are grouped such that each group's token count does not exceed the threshold,
        except when a single message exceeds the threshold, in which case it forms its own group.

        Args:
            messages: List of messages to split
            token_threshold: Maximum token count for each group (may be exceeded by single messages)

        Returns:
            List of message groups, where each group attempts to stay within the token threshold
        """
        if not messages:
            return []

        groups = []
        current_group = []
        current_token_count = 0

        for msg in messages:
            msg_tokens = self.token_count([msg])

            # If single message exceeds threshold, put it in its own group
            if msg_tokens > token_threshold:
                if current_group:
                    groups.append(current_group)
                    current_group = []
                    current_token_count = 0
                groups.append([msg])
                continue

            # If adding this message would exceed threshold, start new group
            if current_token_count + msg_tokens > token_threshold and current_group:
                groups.append(current_group)
                current_group = [msg]
                current_token_count = msg_tokens
            else:
                current_group.append(msg)
                current_token_count += msg_tokens

        # Add the last group if it has messages
        if current_group:
            groups.append(current_group)

        logger.info(f"Split {len(messages)} messages into {len(groups)} groups with token threshold {token_threshold}")
        return groups

    async def _compress_messages_with_llm(self, messages_to_compress: List[Message]) -> str:
        """Compress a list of messages using LLM to generate a summary.

        This method formats the messages into a prompt, sends it to the LLM, and extracts
        the compressed state snapshot from the response. The LLM response is expected to
        contain XML tags for scratchpad and state_snapshot.

        Args:
            messages_to_compress: List of Message objects to compress into a summary.

        Returns:
            Compressed summary string extracted from the LLM response. Returns empty string
            if LLM returns None or if state_snapshot cannot be extracted.

        Note:
            If state_snapshot extraction fails, the full content is used as fallback.
        """
        prompt = self.prompt_format(
            "compress_context_prompt",
            messages_content=merge_messages_content(messages_to_compress),
        )

        def parse_compressed_result(message: Message) -> str:
            content = message.content.strip()

            scratchpad = extract_xml_tag_content(content, "scratchpad")
            state_snapshot = extract_xml_tag_content(content, "state_snapshot")
            logger.info(f"Parsed scratchpad: \n{scratchpad} \nstate_snapshot: \n{state_snapshot}")

            if state_snapshot is None:
                logger.warning("Failed to extract state_snapshot from LLM response, using full content as fallback")
                return content

            return state_snapshot

        # Call LLM to generate compressed summary
        result = await self.llm.achat(
            messages=[Message(role=Role.USER, content=prompt)],
            callback_fn=parse_compressed_result,
        )

        if result is None:
            logger.error("LLM returned None, using empty string as fallback")
            return ""

        return result

    async def _compress_with_groups(
        self,
        system_message: Message,
        message_groups: List[List[Message]],
    ) -> Tuple[dict, list]:
        """Compress multiple message groups and prepare them for storage.

        This method processes each message group, compresses it using LLM, and determines
        whether compression is beneficial. If compression reduces token count, the original
        messages are saved to files and compressed summaries are appended to the system
        message. Otherwise, original messages are preserved in the return list.

        Args:
            system_message: The system message to append compressed summaries to.
            message_groups: List of message groups, where each group is a list of Message
                objects to be compressed together.

        Returns:
            A tuple containing:
                - write_file_dict: Dictionary mapping file paths to JSON-serialized message
                  strings for messages that were successfully compressed.
                - return_messages: List of Message objects including the modified system
                  message with compressed summaries and any messages that couldn't be
                  compressed or didn't benefit from compression.
        """
        write_file_dict = {}
        return_messages = []
        chat_id: str = self.context.get("chat_id", uuid4().hex)

        # Create a copy of system_message to avoid modifying the original
        system_message_copy = Message(role=system_message.role, content=system_message.content)

        for g_idx, messages in enumerate(message_groups):
            group_original_tokens = self.token_count(messages)
            messages_str = json.dumps([x.simple_dump() for x in messages], ensure_ascii=False, indent=2)
            store_path = Path(self.context.get("store_dir", "")) / f"{chat_id}_{g_idx}.json"

            logger.info(f"Compress {g_idx}/{len(message_groups)} ({len(messages)}, {group_original_tokens} tokens)")
            group_summary = await self._compress_messages_with_llm(messages)

            if not group_summary:
                logger.warning(f"Group {g_idx} compression returned empty summary, using original messages.")
                return_messages.extend(messages)
                continue

            compress_content = (
                f"[Compressed conversation history - Part {g_idx}/{len(message_groups)}]\n{group_summary}\n\n"
                f"(Original {len(messages)} messages are stored in: {store_path.as_posix()})\n"
            )
            compressed_tokens = self.token_count([Message(content=compress_content)])

            if compressed_tokens >= group_original_tokens:
                logger.warning(
                    f"Group {g_idx} compression did not reduce tokens: "
                    f"{group_original_tokens} -> {compressed_tokens}. Using original messages.",
                )
                return_messages.extend(messages)
            else:
                system_message_copy.content += compress_content + "\n\n"
                write_file_dict[store_path.as_posix()] = messages_str
                logger.info(
                    f"Group {g_idx} compression successful: "
                    f"{group_original_tokens} -> {compressed_tokens} tokens "
                    f"(reduction: {group_original_tokens - compressed_tokens} tokens, "
                    f"{100 * (1 - compressed_tokens / group_original_tokens):.1f}%)",
                )

        return_messages = [system_message_copy] + return_messages
        return write_file_dict, return_messages

    async def async_execute(self):
        """
        Execute the context compression operation.

        The operation:
        1. Splits messages into system messages, messages to compress, and recent messages
        2. Calculates token count of messages to compress
        3. If below threshold, returns messages unchanged
        4. Otherwise, uses LLM to compress older messages by:
           - Saving original messages to file
           - Generating a concise summary of older messages
           - Appending compressed summaries to the system message
           - Preserving messages that couldn't be compressed or didn't benefit from compression
        """
        # Get configuration from context
        # Note: max_total_tokens does not include keep_recent_count messages or system messages
        max_total_tokens: int = self.context.get("max_total_tokens", 20000)
        group_token_threshold: int = self.context.get("group_token_threshold", None)
        keep_recent_count: int = self.context.get("keep_recent_count", 2)

        assert max_total_tokens > 0, "max_total_tokens must be positive"
        assert keep_recent_count >= 0, "keep_recent_count must be non-negative"

        # Convert context messages to Message objects
        messages = [Message(**x) for x in self.context.messages]

        # Extract system message (should be exactly one)
        system_message = [x for x in messages if x.role is Role.SYSTEM]
        assert len(system_message) <= 1, f"Expected at most one system message, got {len(system_message)}"

        if len(system_message) == 0:
            system_message = Message(role=Role.SYSTEM, content="")
        else:
            system_message = system_message[0]

        messages_without_system = [x for x in messages if x.role is not Role.SYSTEM]
        if keep_recent_count > 0:
            messages_to_compress = messages_without_system[:-keep_recent_count]
            recent_messages = messages_without_system[-keep_recent_count:]
        else:
            messages_to_compress = messages_without_system
            recent_messages = []

        # If nothing to compress after filtering, return original messages
        if not messages_to_compress:
            self.context.response.answer = self.context.messages
            logger.info("No messages to compress after filtering, returning original messages")
            return

        logger.info(f"{len(messages_to_compress)} messages remaining for compression check")

        # Calculate token count of messages to compress (only the content that will be compressed)
        compress_token_cnt: int = self.token_count(messages_to_compress)
        logger.info(f"Context compression check: token count={compress_token_cnt} threshold={max_total_tokens}")

        # If token count is within threshold, no compression needed
        if compress_token_cnt <= max_total_tokens:
            self.context.response.answer = self.context.messages
            logger.info(f"messages_to_compress ({compress_token_cnt}) is within threshold ({max_total_tokens})")
            return

        if group_token_threshold is not None and group_token_threshold > 0:
            message_groups = self._split_messages_by_token_threshold(messages_to_compress, group_token_threshold)
        else:
            message_groups = [messages_to_compress]

        write_file_dict, return_messages = await self._compress_with_groups(system_message, message_groups)

        # Store write_file_dict in context for potential batch writing
        if write_file_dict:
            self.context.write_file_dict = write_file_dict

        self.context.response.answer = [x.simple_dump() for x in (return_messages + recent_messages)]
        self.context.response.metadata["write_file_dict"] = write_file_dict

    async def async_default_execute(self, e: Exception = None, **_kwargs):
        """Handle execution errors by returning original messages.

        This method is called when an exception occurs during async_execute. It preserves
        the original messages and marks the operation as unsuccessful.

        Args:
            e: The exception that occurred during execution, if any.
            **_kwargs: Additional keyword arguments (unused but required by interface).
        """
        self.context.response.answer = self.context.messages
        self.context.response.success = False
        self.context.response.metadata["error"] = str(e)
