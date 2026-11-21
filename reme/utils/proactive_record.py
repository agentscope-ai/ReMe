"""Paths and Markdown decisions shared by v2 task steps and the outbox."""

from datetime import date
import re

import yaml

from ..schema.proactive_agent import ProactiveRecordDecision

_ROUND_TIME = re.compile(r"([0-9]{6})(?:_([0-9]+))?\.md")
_FRONTMATTER = re.compile(r"\A---[ \t]*\r?\n(.*?)^---[ \t]*(?=\r?$)", re.MULTILINE | re.DOTALL)


def round_time(path, agent_name):
    """Return the filename's time and numeric collision suffix for this agent."""
    prefix = f"{agent_name}_"
    if path.name.startswith(prefix) and (match := _ROUND_TIME.fullmatch(path.name[len(prefix) :])):
        return match[1], int(match[2] or 0)
    return None


def is_date_directory(path):
    """Recognize the application's ISO date folder names."""
    try:
        return date.fromisoformat(path.name).isoformat() == path.name
    except ValueError:
        return False


def round_paths(directory, agent_name):
    """Yield this agent's record paths in reverse round order."""
    for day in sorted(directory.glob("????-??-??"), reverse=True):
        if day.is_symlink() or not day.is_dir() or not is_date_directory(day):
            continue
        paths = (
            (stamp, path)
            for path in day.glob(f"{agent_name}_*.md")
            if (stamp := round_time(path, agent_name)) is not None
        )
        for _, path in sorted(paths, reverse=True):
            yield path


def split_record(text):
    """Read YAML metadata while keeping the exact Markdown suffix for state updates."""
    text = text.removeprefix("\ufeff")
    if match := _FRONTMATTER.match(text):
        metadata = yaml.safe_load(match[1])
        if isinstance(metadata, dict):
            return metadata, text[match.end() :]
    return {}, text


def record_decision(metadata, body):
    """Validate that the daily note and its reminder decision form a complete output."""
    decision = ProactiveRecordDecision.model_validate(metadata)
    if decision.has_content != bool(body.strip()):
        raise ValueError("has_content must match whether the Markdown body has content")
    return decision
