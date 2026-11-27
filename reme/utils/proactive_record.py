"""Daily record paths, Markdown parsing, and version tracking for proactive tasks."""

from datetime import date
from pathlib import Path
import re
from stat import S_ISREG

import yaml

from .async_utils import complete_in_thread
from .files.path_lock import get_path_lock
from .files.workspace_files import WorkspaceFiles

_ROUND_SUFFIX = re.compile(r"[0-9]{6}(?:_[0-9]+)?\.md")
_FRONTMATTER = re.compile(r"\A(?:[ \t]*\r?\n)*---[ \t]*\r?\n(.*?)^---[ \t]*(?=\r?$)", re.MULTILINE | re.DOTALL)


def is_round_record(path, agent_name):
    """Check whether the filename matches this agent's round record format."""
    prefix = f"{agent_name}_"
    name = path.name
    return name.startswith(prefix) and _ROUND_SUFFIX.fullmatch(name[len(prefix) :]) is not None


def is_date_directory(path):
    """Recognize the application's ISO date folder names."""
    try:
        return date.fromisoformat(path.name).isoformat() == path.name
    except ValueError:
        return False


def _split_frontmatter(text):
    text = text.removeprefix("\ufeff")
    if match := _FRONTMATTER.match(text):
        return match[1], text[match.end() :]
    return None, text


def split_record(text):
    """Read YAML metadata while keeping the exact Markdown suffix for state updates."""
    header, body = _split_frontmatter(text)
    if header is not None:
        metadata = yaml.safe_load(header)
        if isinstance(metadata, dict):
            return metadata, body
    return {}, text.removeprefix("\ufeff")


class RoundRecord:
    """Track the record's version until the input receipt is durable."""

    def __init__(self, files: WorkspaceFiles):
        self.files = files
        self.path: Path | None = None
        self.written = None

    def _version(self, path):
        try:
            stat = self.files.checked(path).stat(follow_symlinks=False)
        except FileNotFoundError:
            return None
        if not S_ISREG(stat.st_mode):
            raise ValueError(f"Round record must be a regular file: {path}")
        return stat.st_ino, stat.st_mtime_ns, stat.st_size

    async def prepare(self, path: Path):
        """Ensure the model starts with an unused record path."""
        async with await get_path_lock(path):
            if await complete_in_thread(self._version, path) is not None:
                raise FileExistsError(path)
            self.path = path

    def capture(self):
        """Capture the version visible after the model turn for validation."""
        try:
            self.written = self._version(self.path)
        except (OSError, ValueError):
            # Validation reports invalid or unreadable destinations.
            self.written = None

    def check_version(self, path: Path):
        """Reject edits made before commit, including a file created after a skip."""
        if path != self.path or self._version(path) != self.written:
            raise RuntimeError(f"Agent output changed outside this round: {path}")
