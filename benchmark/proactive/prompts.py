"""Freeze the current three-agent prompts without historical workflow variants."""

from pathlib import Path

import yaml


def freeze(repo: Path) -> dict:
    folder = repo / "reme/steps/proactive"
    files = {"pa": "pa_chat", "proactive": "proactive_agent", "task": "task_agent", "memory": "base_pa_step"}
    return {"current": {role: yaml.safe_load((folder / f"{name}.yaml").read_text()) for role, name in files.items()}}
