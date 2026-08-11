#!/usr/bin/env python3
"""Checkpoint-resume support for the reme_eval suite.

Completion source of truth:
    - outputs/reme/<persona>/<task_id>/history/*-log.jsonl  (per-task logs,
      flushed incrementally, survive mid-run kills)
    - outputs/reme/<persona>/run/*-log.jsonl                (run-level logs,
      may be truncated if the process was killed before flush)
    lines: "Task finished task_id=<id> status=<STATUS>"
    A task counts as COMPLETED when its latest terminal status is one of
    SUCCESS / MAX_TURNS / TIMEOUT. ERROR or never-started tasks stay pending.

    "Latest" is decided by EVENT TIME, not by file category or read order:
    each record's "timestamp" (epoch seconds, or "timestamp_iso" as fallback)
    is compared across per-task and run-level logs alike, with the timestamp
    embedded in the log file name as a last-resort fallback. This keeps an
    old run-level SUCCESS from overriding a newer per-task ERROR when the
    re-run died before the new run-level log captured the task.

Commands:
    remaining <persona> [--json]
        Print task_ids still to run, in data/<persona>/episode.yaml order
        (one per line; --json prints {"completed": [...], "remaining": [...]}).

    cleanup <persona> [--dry-run]
        Surgically remove residual memory artifacts of tasks that are about
        to be RE-RUN (i.e. pending tasks that left partial state because a
        previous run was interrupted). This prevents answer leakage: an
        interrupted task's conversation may already have been distilled into
        daily notes during graceful shutdown, and re-running the task with
        that memory injected would inflate scores.

        Removed artifacts (only for pending tasks with residual state):
          - daily/<date>/<note>.md whose frontmatter session_id matches
            pibench_<task_id>_*, plus a refresh of ONLY the daily index of
            the affected date(s) (daily/<date>.md), matched by the full
            workspace-relative note path, never by bare file name
          - digest notes with matching session_id
          - session/dialog/pibench_<task_id>_*.jsonl
          - mem_session/**.jsonl files containing pibench_<task_id>_
        When the ReMe package is importable, the daily index refresh reuses
        ReMe's own rebuild logic (reme.steps.file_io._daily_index.
        refresh_day_index); otherwise index lines are dropped by exact
        wikilink path match. Either way, indexes of other dates are never
        touched. The ReMe watcher (init_changes_step) detects the deleted
        daily notes on next bridge startup and removes them from the BM25
        index itself.

        Completed tasks' memories are NEVER touched by this command.

Design note (resume vs memory-wipe conflict):
    A full memory wipe is a suite-level action of fresh mode (run_all.sh
    without --resume) and happens before any service starts. Resume mode
    never wipes; it only performs the surgical cleanup above. The two modes
    are mutually exclusive, so a resumed run can never lose the cross-session
    memory accumulated by completed tasks.
"""

import asyncio
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import yaml

try:  # Reuse ReMe's daily-index rebuild when running inside the ReMe venv.
    from reme.steps.file_io._daily_index import refresh_day_index
except ImportError:  # pragma: no cover - depends on runtime venv
    refresh_day_index = None

SUITE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("REME_EVAL_DATA_DIR", SUITE_DIR / "data")).resolve()
OUTPUTS_DIR = Path(os.environ.get("REME_EVAL_OUTPUTS_DIR", SUITE_DIR / "outputs")) / "reme"
WORKSPACE_ROOT = Path(
    os.environ.get("REME_WORKSPACE_ROOT", SUITE_DIR / "reme_workspace"),
).resolve()

COMPLETED_STATUSES = {"SUCCESS", "MAX_TURNS", "TIMEOUT"}
TASK_FINISHED_RE = re.compile(r"Task finished task_id=(\S+) status=(\S+)")
SESSION_ID_RE = re.compile(r"^session_id:\s*(\S+)", re.MULTILINE)
NOTE_COUNT_RE = re.compile(r"(description:\s*)\d+(\s*note\(s\) today)")
LOG_FILE_TS_RE = re.compile(r"^(\d{8}_\d{6})-log\.jsonl$")
TIME_FORMAT = "%Y%m%d_%H%M%S"


def log(msg: str) -> None:
    """Print a status message to stderr."""
    print(msg, file=sys.stderr)


def episode_task_order(persona: str) -> list[str]:
    """Return the ordered task ids from the persona's episode.yaml."""
    episode_path = DATA_DIR / persona / "episode.yaml"
    with open(episode_path, "r", encoding="utf-8") as f:
        episode = yaml.safe_load(f)
    return [task["task_id"] for task in episode.get("tasks", [])]


def _event_time(record: dict, file_ts: str) -> float:
    """Best-effort event time (epoch seconds) of one log record.

    Prefers the record's own timestamp fields; falls back to the timestamp
    embedded in the log file name so that even stripped records keep a
    meaningful order. Returns 0.0 when nothing is parseable.
    """
    timestamp = record.get("timestamp")
    if isinstance(timestamp, (int, float)) and not isinstance(timestamp, bool):
        return float(timestamp)
    iso = record.get("timestamp_iso")
    if isinstance(iso, str):
        try:
            return datetime.fromisoformat(iso).timestamp()
        except ValueError:
            pass
    if file_ts:
        try:
            return datetime.strptime(file_ts, TIME_FORMAT).timestamp()
        except ValueError:
            pass
    return 0.0


def latest_task_statuses(persona: str) -> dict[str, str]:
    """Scan per-task and run-level logs; the newest EVENT TIME wins per task.

    Every "Task finished" record across both log categories is keyed by
    (event_time, file timestamp, file order, line number); the record with
    the highest key decides the task's status. File category and read order
    alone can never override a newer record from the other category.
    """
    persona_dir = OUTPUTS_DIR / persona
    if not persona_dir.is_dir():
        return {}

    log_files = sorted(persona_dir.glob("*/history/*-log.jsonl"))
    log_files += sorted(persona_dir.glob("run/*-log.jsonl"))

    best: dict[str, tuple[tuple, str]] = {}
    for file_order, log_file in enumerate(log_files):
        ts_match = LOG_FILE_TS_RE.match(log_file.name)
        file_ts = ts_match.group(1) if ts_match else ""
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                for line_no, line in enumerate(f):
                    if "Task finished" not in line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    match = TASK_FINISHED_RE.search(str(record.get("message", "")))
                    if not match:
                        continue
                    task_id, status = match.group(1), match.group(2)
                    sort_key = (_event_time(record, file_ts), file_ts, file_order, line_no)
                    current = best.get(task_id)
                    if current is None or sort_key > current[0]:
                        best[task_id] = (sort_key, status)
        except OSError:
            continue
    return {task_id: status for task_id, (_, status) in best.items()}


def split_tasks(persona: str) -> tuple[list[str], list[str]]:
    """Split the episode task order into completed and remaining tasks."""
    order = episode_task_order(persona)
    statuses = latest_task_statuses(persona)
    completed = [t for t in order if statuses.get(t) in COMPLETED_STATUSES]
    remaining = [t for t in order if t not in set(completed)]
    return completed, remaining


def _daily_note_session_id(note_path: Path) -> str:
    try:
        text = note_path.read_text(encoding="utf-8")
    except OSError:
        return ""
    match = SESSION_ID_RE.search(text)
    return match.group(1) if match else ""


class _WorkspaceFileStoreShim:
    """Structural stand-in for ReMe's file store; only workspace_path is read."""

    def __init__(self, workspace_path: Path):
        self.workspace_path = workspace_path


def _refresh_daily_indexes(
    workspace: Path,
    removed_by_date: dict[str, set[str]],
    removed: list[str],
) -> None:
    """Rebuild the daily index of each affected date via ReMe's own logic."""
    for date in sorted(removed_by_date):
        result = asyncio.run(
            refresh_day_index(_WorkspaceFileStoreShim(workspace), date, "daily"),
        )
        if result.get("error"):
            log(f"[resume] WARNING: daily index refresh failed for {date}: {result['error']}")
            continue
        removed.append(f"daily/{date}.md (refreshed, {len(removed_by_date[date])} note(s) removed)")


def _strip_index_lines(
    workspace: Path,
    removed_by_date: dict[str, set[str]],
    removed: list[str],
    dry_run: bool,
) -> None:
    """Fallback index edit: drop lines that reference removed notes by full
    workspace-relative wikilink path, and fix the note count. Only the index
    files of affected dates are touched."""
    for date in sorted(removed_by_date):
        index_path = workspace / "daily" / f"{date}.md"
        if not index_path.is_file():
            continue
        wikilinks = [f"[[{rel_path}]]" for rel_path in sorted(removed_by_date[date])]
        lines = index_path.read_text(encoding="utf-8").splitlines()
        kept = [line for line in lines if not any(link in line for link in wikilinks)]
        if len(kept) == len(lines):
            continue
        note_count = sum(1 for line in kept if line.startswith("- [[daily/"))
        kept = [NOTE_COUNT_RE.sub(rf"\g<1>{note_count}\2", line) for line in kept]
        removed.append(f"{index_path.relative_to(workspace)} (rewritten)")
        if not dry_run:
            index_path.write_text("\n".join(kept) + "\n", encoding="utf-8")


def cleanup_partial_memory(persona: str, remaining: list[str], dry_run: bool = False) -> list[str]:
    """Remove partial memory artifacts of remaining tasks so they can be re-run cleanly."""
    workspace = WORKSPACE_ROOT / persona
    removed: list[str] = []
    if not workspace.is_dir() or not remaining:
        return removed

    prefixes = tuple(f"pibench_{task_id}_" for task_id in remaining)

    def act(path: Path, label: str) -> None:
        removed.append(label)
        if not dry_run:
            path.unlink()

    # 1) daily / digest notes distilled from interrupted sessions. For daily
    # notes, remember the full workspace-relative path grouped by date so only
    # the affected daily indexes are refreshed below.
    removed_by_date: dict[str, set[str]] = {}
    for section in ("daily", "digest"):
        section_root = workspace / section
        if not section_root.is_dir():
            continue
        for note_path in section_root.rglob("*.md"):
            if note_path.parent == section_root:
                continue  # index files handled below
            session_id = _daily_note_session_id(note_path)
            if session_id.startswith(prefixes):
                rel_path = note_path.relative_to(workspace).as_posix()
                act(note_path, rel_path)
                if section == "daily":
                    removed_by_date.setdefault(note_path.parent.name, set()).add(rel_path)

    # 2) daily index files: refresh only the dates that lost notes, matching
    # notes by their full wikilink path instead of their bare file name.
    if removed_by_date:
        if dry_run:
            for date in sorted(removed_by_date):
                removed.append(f"daily/{date}.md (would refresh index)")
        elif refresh_day_index is not None:
            _refresh_daily_indexes(workspace, removed_by_date, removed)
        else:
            _strip_index_lines(workspace, removed_by_date, removed, dry_run)

    # 3) raw dialog logs of interrupted sessions
    dialog_dir = workspace / "session" / "dialog"
    if dialog_dir.is_dir():
        for task_id in remaining:
            for dialog_path in dialog_dir.glob(f"pibench_{task_id}_*.jsonl"):
                act(dialog_path, str(dialog_path.relative_to(workspace)))

    # 4) agent-scope session states that contain interrupted-task sessions
    mem_session_dir = workspace / "mem_session"
    if mem_session_dir.is_dir():
        for session_path in mem_session_dir.rglob("*.jsonl"):
            try:
                content = session_path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if any(prefix in content for prefix in prefixes):
                act(session_path, str(session_path.relative_to(workspace)))

    return removed


def main() -> int:
    """CLI entrypoint: run 'remaining' or 'cleanup' action for a persona."""
    args = sys.argv[1:]
    if len(args) < 2 or args[0] not in {"remaining", "cleanup"}:
        print(__doc__, file=sys.stderr)
        return 2

    command, persona = args[0], args[1]
    completed, remaining = split_tasks(persona)

    if command == "remaining":
        if "--json" in args:
            print(json.dumps({"completed": completed, "remaining": remaining}))
        else:
            for task_id in remaining:
                print(task_id)
        log(
            f"[resume] {persona}: completed={len(completed)} "
            f"({', '.join(completed) if completed else '-'}) remaining={len(remaining)}",
        )
        return 0

    dry_run = "--dry-run" in args
    removed = cleanup_partial_memory(persona, remaining, dry_run=dry_run)
    if removed:
        verb = "would remove" if dry_run else "removed"
        log(f"[resume] {persona}: {verb} {len(removed)} partial-memory artifact(s):")
        for item in removed:
            log(f"  - {item}")
    else:
        log(f"[resume] {persona}: no partial-memory artifacts to clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
