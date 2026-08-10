#!/usr/bin/env python3
"""Convert reme_eval run outputs into eval-compatible trace logs.

outputs/{model_id}/{user_id}/{task_id}/history/{ts}-messages.jsonl
  ->  ~/.nanobot/trace_logs/{model_id}/{user_id}/{task_id}/{ts}/turn_N.json

Usage: python fix_trace_logs.py [user_id ...]   (no args = all users)
"""

import json
import re
import sys
from pathlib import Path

SUITE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = SUITE_DIR / "outputs"
TRACE_LOGS_DIR = Path.home() / ".nanobot" / "trace_logs"


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
                if not task_dir.is_dir():
                    continue
                task_id = task_dir.name
                history_dir = task_dir / "history"
                if not history_dir.exists():
                    continue

                pattern = re.compile(r"^(\d{8}_\d{6})-messages\.jsonl$")
                timestamp_files = {}
                for msg_file in history_dir.glob("*-messages.jsonl"):
                    match = pattern.match(msg_file.name)
                    if match:
                        timestamp_files[match.group(1)] = msg_file
                if not timestamp_files:
                    continue

                print(f"\n{model_id}/{user_id}/{task_id}")
                for timestamp, msg_file in sorted(timestamp_files.items()):
                    trace_dir = TRACE_LOGS_DIR / model_id / user_id / task_id / timestamp
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

                    turn_idx, i, turn_count = 1, 0, 0
                    while i < len(messages):
                        turn_msgs = []
                        if messages[i]["role"] == "user":
                            turn_msgs.append(
                                {"role": "user", "content": messages[i]["message"]},
                            )
                            i += 1
                        if i < len(messages) and messages[i]["role"] == "assistant":
                            turn_msgs.append(
                                {"role": "assistant", "content": messages[i]["message"]},
                            )
                            i += 1
                        if turn_msgs:
                            turn_file = trace_dir / f"turn_{turn_idx}.json"
                            with open(turn_file, "w", encoding="utf-8") as f:
                                json.dump(
                                    {"messages": turn_msgs},
                                    f,
                                    indent=2,
                                    ensure_ascii=False,
                                )
                            turn_idx += 1
                            turn_count += 1
                    print(f"  {timestamp}: {turn_count} turns -> {trace_dir}")


if __name__ == "__main__":
    convert_outputs(set(sys.argv[1:]) or None)
    print("\ndone")
