"""
Context compaction module for reducing token usage in conversation contexts.

This module provides functionality to compress large tool messages by storing
their full content in external files and keeping only previews in the context.
This helps manage context window limits while preserving important information.
"""

from pathlib import Path
from uuid import uuid4

from flowllm.core.context import C
from flowllm.core.enumeration import Role
from flowllm.core.op import BaseAsyncOp
from flowllm.core.schema import Message
from loguru import logger


@C.register_op()
class ContextCompactOp(BaseAsyncOp):
    """
    Context compaction operation that reduces token usage by compressing tool messages.

    When the total token count exceeds the threshold, this operation compresses large tool
    messages by truncating their content and storing the full content in external files.
    This helps manage context window limits while preserving recent tool messages.
    """

    async def async_execute(self):
        """
        Execute the context compaction operation.

        The operation:
        1. Calculates the total token count of all messages
        2. If below threshold, returns messages unchanged
        3. Otherwise, compresses large tool messages by:
           - Keeping only a preview of the content
           - Storing full content in external files
           - Preserving recent tool messages
        """
        # Get configuration from context
        max_total_tokens: int = self.context.get("max_total_tokens", 20000)
        max_tool_message_tokens: int = self.context.get("max_tool_message_tokens", 2000)
        preview_char_length: int = self.context.get("preview_char_length", 100)
        keep_recent_count: int = self.context.get("keep_recent_count", 1)
        store_dir: Path = Path(self.context.get("store_dir", ""))

        assert max_total_tokens > 0, "max_total_tokens must be greater than 0"
        assert max_tool_message_tokens > 0, "max_tool_message_tokens must be greater than 0"
        assert preview_char_length >= 0, "preview_char_length must be greater than 0"
        assert keep_recent_count >= 0, "keep_recent_count must be greater than 0"

        # Convert context messages to Message objects
        messages = [Message(**x) for x in self.context.messages]
        messages_to_compress = [x for x in messages if x.role is not Role.SYSTEM]
        if keep_recent_count > 0:
            messages_to_compress = messages_to_compress[:-keep_recent_count]

        # If nothing to compress after filtering, return original messages
        if not messages_to_compress:
            self.context.response.answer = self.context.messages
            logger.info("No messages to compress after filtering, returning original messages")
            return

        logger.info(f"{len(messages_to_compress)} messages remaining for compression check")

        # Calculate total token count
        compact_token_cnt: int = self.token_count(messages_to_compress)
        logger.info(f"Context compaction check: total token count={compact_token_cnt}, threshold={max_total_tokens}")

        # If token count is within threshold, no compaction needed
        if compact_token_cnt <= max_total_tokens:
            self.context.response.answer = self.context.messages
            logger.info(f"Token count ({compact_token_cnt}) is within ({max_total_tokens}), no compaction needed")
            return

        # Filter tool messages for processing
        tool_messages = [x for x in messages_to_compress if x.role is Role.TOOL]

        # Dictionary to store file paths and their full content (for potential batch writing)
        write_file_dict = {}

        # Process each tool message
        for tool_message in tool_messages:
            # Calculate token count for this specific tool message
            tool_token_cnt = self.token_count([tool_message])

            # Skip if token count is within threshold
            if tool_token_cnt <= max_tool_message_tokens:
                logger.info(
                    f"Skipping tool message (tool_call_id={tool_message.tool_call_id}): "
                    f"token count ({tool_token_cnt}) is within threshold ({max_tool_message_tokens})",
                )
                continue

            # Save original full content before modifying
            original_content = tool_message.content

            # Generate file name from tool_call_id or create a unique identifier
            file_name = tool_message.tool_call_id or uuid4().hex
            store_path = store_dir / f"{file_name}.txt"

            # Store the full content for batch writing
            write_file_dict[store_path.as_posix()] = original_content

            # Create compressed preview of the tool message content
            compact_result = original_content[:preview_char_length] + "..."

            # Log the compaction action
            logger.info(
                f"Compacting tool message (tool_call_id={tool_message.tool_call_id}): "
                f"token count={tool_token_cnt}, saving full content to {store_path}",
            )

            # Update tool message content with preview and file reference
            compact_result += f" (detailed result is stored in {store_path})"
            tool_message.content = compact_result

        # Store write_file_dict in context for potential batch writing
        if write_file_dict:
            self.context.write_file_dict = write_file_dict

        # Return the compacted messages as JSON
        self.context.response.answer = [x.simple_dump() for x in messages]
        self.context.response.metadata["write_file_dict"] = write_file_dict

        logger.info(f"Context compaction completed: {len(write_file_dict)} tool messages were compacted")

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
