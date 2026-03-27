"""Tool Result Compactor: truncate large tool results and save full content to files."""

import os
import sys
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
        recent_n: int = 1,
        encoding: str = "utf-8",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tool_result_dir = Path(tool_result_dir)
        self.retention_days = retention_days
        self.old_max_bytes = old_max_bytes
        self.recent_max_bytes = recent_max_bytes
        self.recent_n = recent_n
        self.encoding = encoding
        self.tool_result_dir.mkdir(parents=True, exist_ok=True)

    def _truncate(self, content: str, max_bytes: int) -> str:
        if not content:
            return content

        try:
            if TRUNCATION_NOTICE_MARKER in content:
                return truncate_text_output(content, max_bytes=max_bytes, encoding=self.encoding)

            if len(content.encode(self.encoding)) <= max_bytes + 100:
                return content

            saved_path: str | None = None
            fp = self.tool_result_dir / f"{uuid.uuid4().hex}.txt"
            fp.write_text(content, encoding=self.encoding)
            saved_path = str(fp)

            return truncate_text_output(
                content,
                1,
                content.count("\n") + 1,
                max_bytes,
                file_path=saved_path,
                encoding=self.encoding,
            )
        except Exception as e:
            logger.warning("Failed to truncate content, returning original: %s", e)
            return content

    def _compact(self, output: str | list[dict], max_bytes: int) -> str | list[dict]:
        """Truncate output to max_bytes, saving full content to file if needed."""

        if isinstance(output, str):
            return self._truncate(output, max_bytes)
        if isinstance(output, list):
            for b in output:
                if isinstance(b, dict) and b.get("type") == "text":
                    b["text"] = self._truncate(b.get("text", ""), max_bytes)
        return output

    async def execute(self) -> list[Msg]:
        """Process all messages, truncating large tool results."""
        messages: list[Msg] = self.context.get("messages", [])
        if not messages:
            return messages

        recent_n = 0
        for msg in reversed(messages):
            if not isinstance(msg.content, list) or not any(
                isinstance(b, dict) and b.get("type") == "tool_result" for b in msg.content
            ):
                break
            recent_n += 1
        split_index = max(0, len(messages) - max(recent_n, self.recent_n))

        skills_tool_ids = set()
        try:
            for msg in messages:
                if not isinstance(msg.content, list):
                    continue

                for block in msg.content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        tool_id = block.get("id", "")
                        if not tool_id:
                            continue

                        if (
                            block.get("name", "").lower() == "read_file"
                            and "skill.md" in (block.get("raw_input") or "").lower()
                        ):
                            skills_tool_ids.add(tool_id)
        except Exception as e:
            logger.warning("Failed to detect skill tool ids: %s", e)
        logger.info(f"skills_tool_ids: {skills_tool_ids}")

        for idx, msg in enumerate(messages):
            if not isinstance(msg.content, list):
                continue
            is_recent = idx >= split_index
            max_bytes = self.recent_max_bytes if is_recent else self.old_max_bytes
            for block in msg.content:
                if isinstance(block, dict) and block.get("type") == "tool_result" and block.get("output"):
                    tool_use_id = block.get("id", "")
                    if tool_use_id in skills_tool_ids:
                        effective_max_bytes = self.recent_max_bytes
                    else:
                        effective_max_bytes = max_bytes
                    block["output"] = self._compact(block["output"], effective_max_bytes)

        return messages

    def cleanup_expired_files(self) -> int:
        """Clean up files older than retention_days.

        Returns:
            Number of files successfully deleted.
        """
        if not self.tool_result_dir.exists():
            return 0

        cutoff = datetime.now() - timedelta(days=self.retention_days)
        deleted = failed = 0

        for fp in self.tool_result_dir.glob("*.txt"):
            try:
                stat = os.stat(fp)
                if sys.platform == "win32":
                    ts = stat.st_ctime  # creation time on Windows
                else:
                    ts = getattr(stat, "st_birthtime", stat.st_mtime)  # macOS/BSD; Linux fallback to mtime
                if datetime.fromtimestamp(ts) < cutoff:
                    fp.unlink()
                    deleted += 1
            except FileNotFoundError:
                pass  # deleted by another process between glob and stat/unlink
            except Exception as e:
                failed += 1
                logger.warning("Failed to delete %s: %s", fp, e)

        if deleted or failed:
            logger.info("Cleaned up %d expired files (%d failed)", deleted, failed)
        return deleted
