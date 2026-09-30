"""Content-free, endpoint-scoped plugin diagnostics shared across Codex processes."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

from reme_config import data_dir, lock

LOG_BYTES = 256 * 1024


def digest(value: str) -> str:
    """Return a deterministic filename-safe identifier."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:32]


def queue_root(config: dict) -> Path:
    """Keep queued conversations tied to the service that originally captured them."""
    return data_dir() / "queue" / digest(config["endpoint"])


def endpoint_dir(config: dict) -> Path:
    """Keep operational state separate from conversation retry data."""
    return data_dir() / "state" / digest(config["endpoint"])


def log_status(event: str, error: BaseException | None = None, *, config: dict | None = None, turns: int = 0) -> None:
    """Bound diagnostics without logging prompts, memory, credentials, or exception messages."""
    try:
        with lock(data_dir() / "log.lock") as acquired:
            if not acquired:
                return
            path = data_dir() / "hooks.log"
            if path.exists() and path.stat().st_size >= LOG_BYTES:
                path.replace(path.with_suffix(".log.1"))
            row = {
                "time": time.time(),
                "event": event,
                "error": type(error).__name__ if error else "",
                "endpoint": digest(config["endpoint"]) if config else "",
                "turns": turns,
            }
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
    except OSError:
        pass


def recent_activity(config: dict, limit: int = 20) -> list[dict]:
    """Read only the bounded tail, ignoring torn rows and other services' activity."""
    rows = []
    for name in ("hooks.log.1", "hooks.log"):
        path = data_dir() / name
        try:
            with path.open("rb") as handle:
                size = path.stat().st_size
                handle.seek(max(0, size - LOG_BYTES))
                if size > LOG_BYTES:
                    handle.readline()
                lines = handle.read(LOG_BYTES).splitlines()
        except FileNotFoundError:
            continue
        for line in lines:
            try:
                row = json.loads(line)
            except (ValueError, UnicodeError):
                continue
            if isinstance(row, dict) and row.get("endpoint") == digest(config["endpoint"]):
                rows.append({key: row.get(key) for key in ("time", "event", "error", "turns")})
    return rows[-limit:]


def local_status(config: dict) -> dict:
    """Report delivery state without opening queued conversation files."""
    current = queue_root(config)
    sessions = set()
    pending = other = 0
    for path in (data_dir() / "queue").glob("*/*/*.json"):
        if path.with_suffix(".done").exists():
            continue
        if path.parent.parent == current:
            pending += 1
            sessions.add(path.parent.name)
        else:
            other += 1
    state = endpoint_dir(config)
    try:
        latest = json.loads((state / "dream.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        latest = None
    if latest and latest.get("status") == "running":
        with lock(state / "dream.lock") as acquired:
            if acquired:
                latest = {**latest, "status": "interrupted"}
    try:
        scheduler = json.loads((data_dir() / "scheduler.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        scheduler = {}
    with lock(data_dir() / "scheduler.lock") as acquired:
        if acquired:
            scheduler = {"phase": "stopped", "next_run_at": None}
        elif (
            scheduler.get("endpoint") != digest(config["endpoint"])
            or scheduler.get("enabled") != config["autoDreamEnabled"]
            or scheduler.get("cron") != config["dreamCron"]
            or scheduler.get("timezone") != config["timezone"]
        ):
            scheduler = {"phase": "updating", "next_run_at": None}
    return {
        "auto_recall": {"enabled": config["autoRecall"]},
        "auto_memory": {
            "enabled": config["autoMemoryEnabled"],
            "interval": config["autoMemoryInterval"],
            "queued_turns": pending,
            "queued_sessions": len(sessions),
            "other_endpoint_queued_turns": other,
        },
        "auto_dream": {
            "scheduler": "codex_mcp",
            "enabled": config["autoDreamEnabled"],
            "cron": config["dreamCron"],
            "timezone": config["timezone"],
            "phase": scheduler.get("phase", "starting"),
            "next_run_at": scheduler.get("next_run_at"),
            "last_run": latest,
        },
        "recent_activity": recent_activity(config),
        "hook_trust": "Check the Codex Hooks settings; MCP health does not report host hook trust.",
    }
