#!/usr/bin/env python3
"""Stop hook: warn if there are still events with status: active."""

import os
import re
import sys
from pathlib import Path

vault_path = os.environ.get("VAULT_PATH")
if not vault_path:
    sys.exit(0)

events_dir = Path(vault_path) / "events"
if not events_dir.is_dir():
    sys.exit(0)

active = 0
fm_re = re.compile(r"^---\s*$(.*?)^---\s*$", re.MULTILINE | re.DOTALL)
status_re = re.compile(r"^status:\s*active\s*$", re.MULTILINE)

for md in events_dir.rglob("*.md"):
    try:
        text = md.read_text(encoding="utf-8")
    except Exception:
        continue
    fm_match = fm_re.search(text)
    if fm_match and status_re.search(fm_match.group(1)):
        active += 1

if active > 0:
    sys.stderr.write(
        f"⚠️ 流程合规：还有 {active} 个 active event 未 distill。"
        f"建议调用 `/reme-distill` 把这些 active events 提炼到 topic 后再结束会话。\n",
    )
