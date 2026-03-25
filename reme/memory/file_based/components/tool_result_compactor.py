"""Tool Result Compactor: truncate large tool results and save full content to files."""

import os
import uuid
from datetime import datetime, timedelta
from pathlib import Path

from agentscope.message import Msg

from ..utils import truncate_text_output, DEFAULT_MAX_BYTES, TRUNCATION_NOTICE_MARKER
from ....core.op import BaseOp
from ....core.utils import get_logger

logger = get_logger()


class ToolResultCompactor(BaseOp):
    """Truncate large tool_result outputs and save full content to files."""

    def __init__(
            self,
            tool_result_dir: str | Path,
            retention_days: int = 3,
            old_max_bytes: int = 3000,
            recent_max_bytes: int = DEFAULT_MAX_BYTES,
            encoding: str = "utf-8",
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.tool_result_dir = Path(tool_result_dir)
        self.retention_days = retention_days
        self.old_max_bytes = old_max_bytes
        self.recent_max_bytes = recent_max_bytes
        self.encoding = encoding

        self.tool_result_dir.mkdir(parents=True, exist_ok=True)

    def _save_and_truncate(
            self,
            content: str,
            max_bytes: int,
            allow_retruncate: bool = True,
    ) -> str:
        """Save full content to file and return truncated version with file reference.

        Args:
            content: The content to potentially truncate
            max_bytes: Maximum bytes allowed
            allow_retruncate: If True and content already truncated, do re-truncation.
                             If False and content already truncated, return as-is.
        """
        if not content:
            return content

        if TRUNCATION_NOTICE_MARKER in content:
            if not allow_retruncate:
                return content
            origin_context, extra = content.split(TRUNCATION_NOTICE_MARKER, 1)
            origin_context_encode = origin_context.encode(self.encoding)
            if len(origin_context_encode) <= max_bytes:
                return content
            else:
                origin_context_retruncate = origin_context_encode[:max_bytes].decode(self.encoding, errors="ignore")
                content = origin_context_retruncate + "..." + TRUNCATION_NOTICE_MARKER + extra
                return content
        else:
            content_encode = content.encode(self.encoding)
            if len(content_encode) <= max_bytes:
                return content
            else:
                # Save full content to file
                file_path = self.tool_result_dir / f"{uuid.uuid4().hex}.txt"
                file_path.write_text(content, encoding="utf-8")

                total_lines = content.count("\n") + 1
                content = truncate_text_output(content, 1, total_lines, max_bytes, file_path=str(file_path))
                return content

    def _process_output(
            self,
            output: str | list[dict],
            max_bytes: int,
            allow_retruncate: bool = True,
    ) -> str | list[dict]:
        """Process tool result output, truncating if necessary."""
        if isinstance(output, str):
            return self._save_and_truncate(output, max_bytes, allow_retruncate)

        if isinstance(output, list):
            return [
                (
                    {
                        **b,
                        "text": self._save_and_truncate(
                            b.get("text", ""), max_bytes, allow_retruncate
                        ),
                    }
                    if isinstance(b, dict) and b.get("type") == "text"
                    else b
                )
                for b in output
            ]
        return output

    async def execute(self) -> list[Msg]:
        """Process all messages, truncating large tool results."""
        messages: list[Msg] = self.context.get("messages", [])
        if not messages:
            return messages

        # Calculate recent_n: count consecutive messages with tool_result from the end
        recent_n = 0
        for msg in reversed(messages):
            if not isinstance(msg.content, list):
                break
            has_tool_result = any(
                isinstance(b, dict) and b.get("type") == "tool_result"
                for b in msg.content
            )
            if not has_tool_result:
                break
            recent_n += 1
        recent_n = max(recent_n, 1)

        # Split messages into old and recent parts
        split_index = max(0, len(messages) - recent_n)

        for idx, msg in enumerate(messages):
            if not isinstance(msg.content, list):
                continue

            # Determine max_bytes and allow_retruncate based on message position
            is_recent = idx >= split_index
            max_bytes = self.recent_max_bytes if is_recent else self.old_max_bytes
            allow_retruncate = not is_recent  # recent messages: no re-truncate, old messages: allow re-truncate

            for block in msg.content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    output = block.get("output")
                    if output:
                        block["output"] = self._process_output(
                            output, max_bytes, allow_retruncate
                        )

        return messages

    def cleanup_expired_files(self) -> int:
        """Clean up files older than retention_days."""
        if not self.tool_result_dir.exists():
            return 0

        cutoff = datetime.now() - timedelta(days=self.retention_days)
        deleted = 0

        for fp in self.tool_result_dir.glob("*.txt"):
            try:
                stat_info = os.stat(fp)
                created_at = datetime.fromtimestamp(stat_info.st_birthtime)
                if created_at < cutoff:
                    fp.unlink()
                    deleted += 1
            except Exception as e:
                logger.warning("Failed to process %s: %s", fp, e)

        if deleted:
            logger.info("Cleaned up %d expired files", deleted)
        return deleted
