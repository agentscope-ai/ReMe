"""Tool Result Compactor: truncate large tool results and save full content to files."""

import uuid
from datetime import datetime, timedelta
from pathlib import Path

from agentscope.message import Msg

from ....core.op import BaseOp
from ....core.utils import get_std_logger
from ....core.utils import truncate_text, is_truncated

logger = get_std_logger()


class ToolResultCompactor(BaseOp):
    """Truncate large tool_result outputs and save full content to files."""

    def __init__(
        self,
        tool_result_dir: str | Path,
        tool_result_threshold: int,
        retention_days: int = 7,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tool_result_dir = Path(tool_result_dir)
        self.tool_result_threshold = tool_result_threshold
        self.retention_days = retention_days

    def _save_and_truncate(self, content: str, tool_name: str) -> str:
        """Save full content to file and return truncated version with file reference."""
        if not content or is_truncated(content) or len(content) <= self.tool_result_threshold:
            return content

        # Save full content
        self.tool_result_dir.mkdir(parents=True, exist_ok=True)
        file_path = self.tool_result_dir / f"{uuid.uuid4().hex}.txt"
        created_at = datetime.now().isoformat()

        file_path.write_text(
            f"# tool_name: {tool_name}\n# created_at: {created_at}\n# ---\n{content}",
            encoding="utf-8",
        )
        logger.debug("Saved tool result to %s (len=%d)", file_path, len(content))

        # Return truncated with file reference
        return f"{truncate_text(content, self.tool_result_threshold)}\n\n[Full content saved to: {file_path}]"

    def _process_output(self, output: str | list[dict], tool_name: str) -> str | list[dict]:
        """Process tool result output, truncating if necessary."""
        if isinstance(output, str):
            return self._save_and_truncate(output, tool_name)

        if isinstance(output, list):
            return [
                (
                    {**b, "text": self._save_and_truncate(b.get("text", ""), tool_name)}
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

        for msg in messages:
            if not isinstance(msg.content, list):
                continue

            for block in msg.content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    output = block.get("output")
                    if output:
                        block["output"] = self._process_output(output, block.get("name", "unknown"))

        return messages

    def cleanup_expired_files(self) -> int:
        """Clean up files older than retention_days."""
        if not self.tool_result_dir.exists():
            return 0

        cutoff = datetime.now() - timedelta(days=self.retention_days)
        deleted = 0

        for fp in self.tool_result_dir.glob("*.txt"):
            try:
                for line in fp.read_text(encoding="utf-8").splitlines()[:3]:
                    if line.startswith("# created_at:"):
                        if datetime.fromisoformat(line.split(":", 1)[1].strip()) < cutoff:
                            fp.unlink()
                            deleted += 1
                        break
            except Exception as e:
                logger.warning("Failed to process %s: %s", fp, e)

        if deleted:
            logger.info("Cleaned up %d expired files", deleted)
        return deleted
