"""Meta-ReMe compression recovery adapted to AgentScope's native context pipeline."""

import asyncio
import logging
from pathlib import Path
from uuid import uuid4

from agentscope.exception import ToolJSONDecodeError
from agentscope.message import TextBlock
import jsonschema
import openai
from pydantic import ValidationError

from ..evaluation.artifacts import contained, write_json
from .bounded import BoundedReActAgent

logger = logging.getLogger(__name__)


class RecoverableCompressionAgent(BoundedReActAgent):
    """Skip recoverable failures until emergency offload is needed.

    Override the native implementation so compression middleware, successful-change
    counters, finalization and cancellation remain owned by their existing layers.
    """

    def __init__(self, *args, fallback_directory=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.fallback_directory = Path(fallback_directory) if fallback_directory is not None else None

    async def _compress_context_impl(self, context_config=None, instructions=None):
        """Keep the old trigger-derived emergency boundary without silent truncation."""
        config = context_config or self.context_config
        model_input = await self._prepare_model_input()
        context_ratio = await self.model.count_tokens(**model_input) / self.model.context_size
        if context_ratio < config.trigger_ratio:
            return
        # The SDK's default truncation would hide the failure and discard old
        # evidence before Home can decide whether emergency offload is needed.
        config = config.model_copy(update={"compression_fallback_to_truncation": False})
        try:
            await super()._compress_context_impl(context_config=config, instructions=instructions)
        except Exception as exc:
            if not self._is_recoverable_compression_error(exc):
                raise
            emergency_ratio = (config.trigger_ratio + 1.0) / 2.0
            if context_ratio < emergency_ratio:
                logger.warning(
                    "Context compression failed at %.1f%%, below the %.1f%% emergency threshold; skipping: %s: %s",
                    context_ratio * 100,
                    emergency_ratio * 100,
                    type(exc).__name__,
                    exc,
                )
                return
            logger.warning(
                "Context compression failed at %.1f%%; using local offload at the %.1f%% emergency threshold: %s: %s",
                context_ratio * 100,
                emergency_ratio * 100,
                type(exc).__name__,
                exc,
            )
            await self._fallback_compress_context(exc, context_ratio, config)

    @staticmethod
    def _is_recoverable_compression_error(exc):
        """Match dev_meta's model/output failures, excluding uncompressible prompts."""
        if isinstance(exc, RuntimeError):
            return not str(exc).startswith("The system prompt")
        return isinstance(exc, (ToolJSONDecodeError, jsonschema.ValidationError, ValidationError, openai.OpenAIError))

    async def _fallback_compress_context(self, cause, context_ratio, config):
        """Persist older messages before replacing state; never call a second model."""
        if self.fallback_directory is None:
            raise RuntimeError("Optimizer fallback_directory is required for local context compression") from cause
        model_input = await self._prepare_model_input()
        old_messages, recent_messages = await self._split_context_for_compression(
            config.reserve_ratio * self.model.context_size,
            model_input.get("tools", []),
        )
        if not old_messages:
            raise RuntimeError(
                "Local fallback compression cannot remove any context while preserving the configured reserve",
            ) from cause

        async def apply_change():
            contained(self.fallback_directory.parent, self.fallback_directory, directory=True)
            path = contained(self.fallback_directory, f"context-{uuid4().hex}.json")
            write_json(
                path,
                {
                    "session_id": self.state.session_id,
                    "context_ratio": context_ratio,
                    "trigger_ratio": config.trigger_ratio,
                    "emergency_ratio": (config.trigger_ratio + 1.0) / 2.0,
                    "failure": {"type": type(cause).__name__, "message": str(cause)},
                    "messages": [message.model_dump(mode="json", exclude_none=False) for message in old_messages],
                },
            )
            notice = (
                "<system-reminder>Model-based context compression failed near the context limit. "
                f"Older context was offloaded without semantic summarization to '{path}'. "
                "The recent context is preserved; use Read on that file if older experiments, decisions, "
                "validation results, or tool outputs are needed.</system-reminder>"
            )
            await self._clear_unreserved_read_cache(recent_messages)
            # Recent SDK versions also carry current-reply usage through compression.
            usage = self._get_reply_usage()
            summary = self.state.summary
            self.state.summary = (
                "\n".join(part for part in (summary, notice) if part)
                if isinstance(summary, str)
                else [*summary, TextBlock(text=notice)]
            )
            self.state.context = recent_messages
            if usage is not None and self._get_reply_usage() is None:
                self.state.append_context(self.name, [])
                self.state.context[-1].usage = usage
            logger.warning("Local compression offloaded %d message(s) to %s", len(old_messages), path)

        task = asyncio.create_task(apply_change())
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            await task
            raise
