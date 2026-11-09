"""Freeze the current three-agent prompts without historical workflow variants."""

from pathlib import Path

import yaml


def freeze(repo: Path) -> dict:
    """Capture the current localized prompts for all three agent roles."""
    folder = repo / "reme/steps/proactive"
    files = {
        "pa": "pa_chat",
        "proactive": "proactive_agent",
        "cron": "cron_agent",
        "temporary": "temp_proactive",
        "memory": "base_pa_step",
    }
    return {"current": {role: yaml.safe_load((folder / f"{name}.yaml").read_text()) for role, name in files.items()}}
