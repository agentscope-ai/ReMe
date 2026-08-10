#!/usr/bin/env python3
"""Checkpoint-resume support for the reme_eval suite.

Completion source of truth:
    - outputs/reme/<persona>/<task_id>/history/*-log.jsonl  (per-task, primary:
      flushed incrementally, survives mid-run kills)
    - outputs/reme/<persona>/run/*-log.jsonl                (run-level, may be
      truncated if the process was killed before flush)
    lines: "Task finished task_id=<id> status=<STATUS>"
    A task counts as COMPLETED when its latest terminal status is one of
    SUCCESS / MAX_TURNS / TIMEOUT. ERROR or never-started tasks stay pending.

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
            pibench_<task_id>_* (plus the bullet in the daily index file,
            with the "N note(s) today" count fixed)
          - digest notes with matching session_id
          - session/dialog/pibench_<task_id>_*.jsonl
          - mem_session/**.jsonl files containing pibench_<task_id>_
        The ReMe watcher (init_changes_step) detects the deleted daily notes
        on next bridge startup and removes them from the BM25 index itself.

        Completed tasks' memories are NEVER touched by this command.

Design note (resume vs memory-wipe conflict):
    A full memory wipe is a suite-level action of fresh mode (run_all.sh
    without --resume) and happens before any service starts. Resume mode
    never wipes; it only performs the surgical cleanup above. The two modes
    are mutually exclusive, so a resumed run can never lose the cross-session
    memory accumulated by completed tasks.
"""

import json
import os
import re
import sys
from pathlib import Path

import yaml

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


def log(msg: str) -> None:
    """Print a status message to stderr."""
    print(msg, file=sys.stderr)


def episode_task_order(persona: str) -> list[str]:
    """Return the ordered task ids from the persona's episode.yaml."""
    episode_path = DATA_DIR / persona / "episode.yaml"
    with open(episode_path, "r", encoding="utf-8") as f:
        episode = yaml.safe_load(f)
    return [task["task_id"] for task in episode.get("tasks", [])]


def latest_task_statuses(persona: str) -> dict[str, str]:
    """Scan per-task history logs and run-level logs; later records win."""
    statuses: dict[str, str] = {}
    persona_dir = OUTPUTS_DIR / persona
    if not persona_dir.is_dir():
        return statuses

    log_files = sorted(persona_dir.glob("*/history/*-log.jsonl"))
    log_files += sorted(persona_dir.glob("run/*-log.jsonl"))

    for log_file in log_files:
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                for line in f:
                    if "Task finished" not in line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    match = TASK_FINISHED_RE.search(record.get("message", ""))
                    if match:
                        statuses[match.group(1)] = match.group(2)
        except OSError:
            continue
    return statuses


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

    # 1) daily / digest notes distilled from interrupted sessions
    removed_note_names: set[str] = set()
    for section in ("daily", "digest"):
        section_root = workspace / section
        if not section_root.is_dir():
            continue
        for note_path in section_root.rglob("*.md"):
            if note_path.parent == section_root:
                continue  # index files handled below
            session_id = _daily_note_session_id(note_path)
            if session_id.startswith(prefixes):
                act(note_path, str(note_path.relative_to(workspace)))
                removed_note_names.add(note_path.name)

    # 2) daily index files: drop bullets for removed notes, fix note count
    daily_root = workspace / "daily"
    if daily_root.is_dir() and removed_note_names:
        for index_path in daily_root.glob("*.md"):
            lines = index_path.read_text(encoding="utf-8").splitlines()
            kept = [line for line in lines if not any(note_name in line for note_name in removed_note_names)]
            if len(kept) == len(lines):
                continue
            note_count = sum(1 for line in kept if line.startswith("- [[daily/"))
            kept = [NOTE_COUNT_RE.sub(rf"\g<1>{note_count}\2", line) for line in kept]
            removed.append(f"{index_path.relative_to(workspace)} (rewritten)")
            if not dry_run:
                index_path.write_text("\n".join(kept) + "\n", encoding="utf-8")

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
