"""
Context offload module for managing context window limits through compaction and compression.

This module provides a high-level operation that orchestrates context compaction and compression
to reduce token usage. It first attempts to compact tool messages, and if the compaction ratio
is not sufficient, it applies LLM-based compression to further reduce token count.

The offload process:
1. Compacts tool messages by storing full content in external files
2. Evaluates the compaction effectiveness by comparing token counts
3. If compaction ratio exceeds threshold, applies LLM-based compression
"""

from flowllm.core.context import C
from flowllm.core.op import BaseAsyncOp
from flowllm.core.schema import Message
from loguru import logger

from reme_ai.enumeration import ContextManageEnum


@C.register_op()
class ContextOffloadOp(BaseAsyncOp):
    """
    Context offload operation that orchestrates compaction and compression to reduce token usage.

    This operation combines context compaction and compression strategies to manage context
    window limits. It first applies compaction to tool messages, then evaluates the effectiveness.
    If the compaction ratio (compressed tokens / original tokens) exceeds a threshold, it
    applies additional LLM-based compression to further reduce token count.

    Context Parameters:
        context_manage_mode (ContextManageEnum): The context management mode to use.
            - COMPACT: Only applies context compaction to tool messages.
            - COMPRESS: Only applies LLM-based compression to messages.
            - AUTO: Applies compaction first, then compression if compaction ratio exceeds threshold.
            Defaults to AUTO.
        compact_ratio_threshold (float): Threshold for compaction ratio above which compression
            is applied. Only used in AUTO mode. Defaults to 0.75. If the ratio of compressed
            tokens to original tokens exceeds this value, compression will be triggered.
    """

    async def async_execute(self):
        """
        Execute the context offload operation.

        The operation behavior depends on the context_manage_mode:
        - COMPACT: Only applies context compaction to reduce token usage in tool messages.
        - COMPRESS: Only applies LLM-based compression to generate concise summaries.
        - AUTO: Applies compaction first, then compression if compaction ratio exceeds threshold.

        The compaction operation stores full tool message content in external files and
        keeps only previews in the context. The compression operation uses LLM to generate
        concise summaries of older messages.
        """
        from .context_compact_op import ContextCompactOp
        from .context_compress_op import ContextCompressOp

        # Get the context management mode from context, default to AUTO
        context_manage_mode = self.context.get("context_manage_mode", ContextManageEnum.AUTO)
        if isinstance(context_manage_mode, str):
            context_manage_mode = ContextManageEnum(context_manage_mode)

        context_compact_op = ContextCompactOp()
        context_compress_op = ContextCompressOp()

        if context_manage_mode == ContextManageEnum.COMPACT:
            # Only apply compaction
            logger.info("Context management mode: COMPACT")
            await context_compact_op.async_call(context=self.context)
        elif context_manage_mode == ContextManageEnum.COMPRESS:
            # Only apply compression
            logger.info("Context management mode: COMPRESS")
            await context_compress_op.async_call(context=self.context)
        elif context_manage_mode == ContextManageEnum.AUTO:
            # Apply compaction first, then compression if needed
            logger.info("Context management mode: AUTO")
            await context_compact_op.async_call(context=self.context)

            origin_messages = [Message(**x) for x in self.context.messages]
            origin_token_cnt = self.token_count(origin_messages)

            result_messages = [Message(**x) for x in self.context.response.answer]
            answer_token_cnt = self.token_count(result_messages)

            compact_ratio = answer_token_cnt / origin_token_cnt

            compact_ratio_threshold: float = self.context.get("compact_ratio_threshold", 0.75)
            if compact_ratio > compact_ratio_threshold:
                logger.info(f"Compact ratio {compact_ratio:.2f} > {compact_ratio_threshold:.2f}, compress answer")
                await context_compress_op.async_call(context=self.context)
        else:
            raise ValueError(f"Unknown context management mode: {context_manage_mode}")

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
