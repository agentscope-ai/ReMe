#!/usr/bin/env python3
"""Convert reme_eval run outputs into eval-compatible trace logs.

outputs/{model_id}/{user_id}/{task_id}/history/{ts}-messages.jsonl
  ->  ~/.nanobot/trace_logs/{model_id}/{user_id}/{task_id}/{ts}/turn_N.json

The bridge additionally writes {ts}-tools.jsonl sidecar files next to the
message histories: one JSON object per executed tool call with fields
{turn, name, arguments, result}. Each messages run is paired with the
temporally closest sidecar, and the records are merged into the generated
turn files under the "tool_steps" key, which is one of the tool-history
formats π-Bench's collect_tool_history() understands. Without this step,
tools_evaluation scripts would see no tool evidence at all.

Usage: python fix_trace_logs.py [user_id ...]   (no args = all users)
"""

import json
import re
import sys
from datetime import datetime
from pathlib import Path

SUITE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = SUITE_DIR / "outputs"
TRACE_LOGS_DIR = Path.home() / ".nanobot" / "trace_logs"

MESSAGES_FILE_RE = re.compile(r"^(\d{8}_\d{6})-messages\.jsonl$")
TOOLS_FILE_RE = re.compile(r"^(\d{8}_\d{6})-tools\.jsonl$")
TIME_FORMAT = "%Y%m%d_%H%M%S"
# A tool sidecar belongs to the messages run that started at most this many
# seconds earlier (the bridge stamps the sidecar when the task's first user
# message arrives, shortly after the runner opened the messages file).
MAX_PAIR_DELTA_SECONDS = 6 * 3600


def _to_epoch(timestamp: str) -> float:
    """Parse a YYYYMMDD_HHMMSS timestamp into epoch seconds."""
    try:
        return datetime.strptime(timestamp, TIME_FORMAT).timestamp()
    except ValueError:
        return 0.0


def load_tool_records(tools_file: Path) -> dict:
    """Group sidecar tool records by turn number."""
    by_turn: dict = {}
    try:
        with open(tools_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(record, dict) or not record.get("name"):
                    continue
                turn = int(record.get("turn") or 0)
                by_turn.setdefault(turn, []).append(
                    {
                        "name": record["name"],
                        "arguments": record.get("arguments", {}),
                        "result": record.get("result", ""),
                    },
                )
    except OSError as exc:
        print(f"  WARNING: cannot read tool sidecar {tools_file}: {exc}")
    return by_turn


def pair_tool_sidecars(message_runs: list, tool_runs: list) -> dict:
    """Pair each messages run with the temporally closest unused tool sidecar.

    Fresh runs produce exactly one messages file and one sidecar per task;
    re-runs append matching pairs, so sorted greedy nearest-timestamp
    matching is stable. Sidecars farther away than MAX_PAIR_DELTA_SECONDS
    (e.g. leftovers of a crashed bridge) stay unpaired.
    """
    pairing: dict = {}
    unused = list(tool_runs)
    for msg_ts, _ in message_runs:
        best_delta = None
        best_item = None
        for tool_ts, tool_path in unused:
            delta = abs(_to_epoch(tool_ts) - _to_epoch(msg_ts))
            if best_delta is None or delta < best_delta:
                best_delta = delta
                best_item = (tool_ts, tool_path)
        if best_delta is not None and best_item is not None and best_delta <= MAX_PAIR_DELTA_SECONDS:
            pairing[msg_ts] = best_item[1]
            unused.remove(best_item)
    return pairing


def build_turns(messages: list) -> list:
    """Split the flat message list into per-turn [user, assistant] groups."""
    turns = []
    i = 0
    while i < len(messages):
        turn_msgs = []
        if messages[i]["role"] == "user":
            turn_msgs.append({"role": "user", "content": messages[i]["message"]})
            i += 1
        if i < len(messages) and messages[i]["role"] == "assistant":
            turn_msgs.append({"role": "assistant", "content": messages[i]["message"]})
            i += 1
        if not turn_msgs:
            i += 1  # defensive: never spin on unexpected roles
            continue
        turns.append(turn_msgs)
    return turns


def convert_task(model_id: str, user_id: str, task_dir: Path) -> None:
    """Convert one task's history dir into trace turn files with tool_steps."""
    history_dir = task_dir / "history"
    if not history_dir.is_dir():
        return

    message_runs = []
    tool_runs = []
    for msg_file in history_dir.glob("*-messages.jsonl"):
        match = MESSAGES_FILE_RE.match(msg_file.name)
        if match:
            message_runs.append((match.group(1), msg_file))
    for tools_file in history_dir.glob("*-tools.jsonl"):
        match = TOOLS_FILE_RE.match(tools_file.name)
        if match:
            tool_runs.append((match.group(1), tools_file))
    if not message_runs:
        return

    message_runs.sort(key=lambda item: item[0])
    tool_runs.sort(key=lambda item: item[0])
    pairing = pair_tool_sidecars(message_runs, tool_runs)

    print(f"\n{model_id}/{user_id}/{task_dir.name}")
    for timestamp, msg_file in message_runs:
        trace_dir = TRACE_LOGS_DIR / model_id / user_id / task_dir.name / timestamp
        trace_dir.mkdir(parents=True, exist_ok=True)

        messages = []
        with open(msg_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                msg = json.loads(line)
                if msg.get("role") == "user" and msg.get("message") == "/new":
                    continue
                messages.append(msg)

        tools_file = pairing.get(timestamp)
        tools_by_turn = load_tool_records(tools_file) if tools_file else {}
        if tools_file is not None:
            print(f"  {timestamp}: paired tool sidecar {tools_file.name}")

        turns = build_turns(messages)
        for turn_idx, turn_msgs in enumerate(turns, start=1):
            turn_data = {"messages": turn_msgs}
            tool_steps = tools_by_turn.get(turn_idx)
            if tool_steps:
                turn_data["tool_steps"] = tool_steps
            turn_file = trace_dir / f"turn_{turn_idx}.json"
            with open(turn_file, "w", encoding="utf-8") as f:
                json.dump(turn_data, f, indent=2, ensure_ascii=False)
        tool_total = sum(len(steps) for steps in tools_by_turn.values())
        print(f"  {timestamp}: {len(turns)} turns, {tool_total} tool step(s) -> {trace_dir}")


def convert_outputs(user_filter=None):
    """Convert message history JSONL files into per-turn trace JSON files."""
    if not OUTPUTS_DIR.exists():
        print(f"outputs dir not found: {OUTPUTS_DIR}")
        return

    for model_dir in sorted(OUTPUTS_DIR.iterdir()):
        if not model_dir.is_dir():
            continue
        model_id = model_dir.name

        for user_dir in sorted(model_dir.iterdir()):
            if not user_dir.is_dir():
                continue
            user_id = user_dir.name
            if user_filter and user_id not in user_filter:
                continue

            for task_dir in sorted(user_dir.iterdir()):
                if task_dir.is_dir():
                    convert_task(model_id, user_id, task_dir)


if __name__ == "__main__":
    convert_outputs(set(sys.argv[1:]) or None)
    print("\ndone")
