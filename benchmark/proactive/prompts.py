"""Freeze the current three-agent prompts without historical workflow variants."""

from pathlib import Path

import yaml


def freeze(repo: Path) -> dict:
    """Capture the current localized prompts for all three agent roles."""
    folder = repo / "reme/steps/proactive"
    files = {
        "pa": "agents/main",
        "proactive": "agents/proactive",
        "cron": "agents/cron",
        "temporary": "agents/temporary",
        "memory": "agents/base",
    }
    return {"current": {role: yaml.safe_load((folder / f"{name}.yaml").read_text()) for role, name in files.items()}}
