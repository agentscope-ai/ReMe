"""Tool Result Compactor: truncate large tool results and save full content to files."""

import uuid
from datetime import datetime, timedelta
from pathlib import Path

from agentscope.message import Msg

from ....core.op import BaseOp
from ....core.utils import get_logger
from ....core.utils import truncate_text_head, TRUNCATION_MARKER_START

logger = get_logger()

MAX_LINE_LENGTH = 10000


def _split_long_lines(text: str, max_len: int = MAX_LINE_LENGTH) -> str:
    """Split lines that exceed max_len by inserting newlines."""
    lines = text.split("\n")
    result = []
    for line in lines:
        if len(line) <= max_len:
            result.append(line)
        else:
            # Split line into chunks of max_len
            for i in range(0, len(line), max_len):
                result.append(line[i : i + max_len])
    return "\n".join(result)


class ToolResultCompactor(BaseOp):
    """Truncate large tool_result outputs and save full content to files."""

    def __init__(
        self,
        tool_result_dir: str | Path,
        retention_days: int = 7,
        recent_n: int = 1,
        old_threshold: int = 500,
        recent_threshold: int = 30000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tool_result_dir = Path(tool_result_dir)
        self.retention_days = retention_days
        self.recent_n = recent_n
        self.old_threshold = old_threshold
        self.recent_threshold = recent_threshold

    def _save_and_truncate(self, content: str, tool_name: str, threshold: int) -> str:
        """Save full content to file and return truncated version with file reference."""
        if not content:
            return content

        # Check if content was previously truncated
        if TRUNCATION_MARKER_START in content:
            parts = content.split(TRUNCATION_MARKER_START, 1)
            if len(parts[0]) <= threshold:
                return content
            return f"{truncate_text_head(parts[0], threshold)}{parts[1]}"

        # Not truncated before
        if len(content) <= threshold:
            return content

        # Save full content with long lines split
        self.tool_result_dir.mkdir(parents=True, exist_ok=True)
        file_path = self.tool_result_dir / f"{uuid.uuid4().hex}.txt"
        created_at = datetime.now().isoformat()

        processed_content = _split_long_lines(content)
        file_path.write_text(
            f"# tool_name: {tool_name}\n# created_at: {created_at}\n# ---\n{processed_content}",
            encoding="utf-8",
        )
        logger.debug("Saved tool result to %s (len=%d)", file_path, len(content))

        # Return truncated with file reference
        return f"{truncate_text_head(content, threshold)}\n\n[Full content saved to: {file_path}]"

    def _process_output(self, output: str | list[dict], tool_name: str, threshold: int) -> str | list[dict]:
        """Process tool result output, truncating if necessary."""
        if isinstance(output, str):
            return self._save_and_truncate(output, tool_name, threshold)

        if isinstance(output, list):
            return [
                (
                    {**b, "text": self._save_and_truncate(b.get("text", ""), tool_name, threshold)}
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

        # Split messages into old and recent parts
        split_index = max(0, len(messages) - self.recent_n)

        for idx, msg in enumerate(messages):
            if not isinstance(msg.content, list):
                continue

            # Determine threshold based on message position
            threshold = self.recent_threshold if idx >= split_index else self.old_threshold

            for block in msg.content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    output = block.get("output")
                    if output:
                        block["output"] = self._process_output(output, block.get("name", "unknown"), threshold)

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
