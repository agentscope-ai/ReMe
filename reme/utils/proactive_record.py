"""Daily record paths and Markdown parsing for proactive tasks."""

from datetime import date
import re

import yaml

_ROUND_SUFFIX = re.compile(r"[0-9]{6}(?:_[0-9]+)?\.md")
_FRONTMATTER = re.compile(r"\A---[ \t]*\r?\n(.*?)^---[ \t]*(?=\r?$)", re.MULTILINE | re.DOTALL)


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


def round_paths(directory, agent_name):
    """Discover this agent's round records in dated directories."""
    for day in directory.glob("????-??-??"):
        if day.is_symlink() or not day.is_dir() or not is_date_directory(day):
            continue
        yield from (path for path in day.glob(f"{agent_name}_*.md") if is_round_record(path, agent_name))


def split_record(text):
    """Read YAML metadata while keeping the exact Markdown suffix for state updates."""
    text = text.removeprefix("\ufeff")
    if match := _FRONTMATTER.match(text):
        metadata = yaml.safe_load(match[1])
        if isinstance(metadata, dict):
            return metadata, text[match.end() :]
    return {}, text
