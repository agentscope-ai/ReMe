"""Explicit, offline migration of a PA reminder journal to required message kinds."""

import argparse
import json
import os
from pathlib import Path
from uuid import uuid4

from reme.schema.pa_hint import PaHintState
from reme.schema.reminder import ReminderRecord


def migrate(path: Path) -> int:
    """Validate every row before atomically enriching old rows; preserve a backup."""
    if path.is_symlink() or not path.is_file():
        raise ValueError("Reminder journal must be a regular file, not a symlink")
    original = path.read_bytes()
    state_path = path.with_name("hint_state.json")
    state = PaHintState.model_validate_json(state_path.read_bytes()) if state_path.exists() else PaHintState()
    periods = {draw.round_id: period for period, draw in state.draws.items() if draw.hint_index is not None}
    rows, changed = [], 0
    for line in original.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "kind" not in row:
            row["kind"] = periods.get(row["id"], "normal")
            changed += 1
        ReminderRecord.model_validate(row)
        rows.append(row)
    if not changed:
        return 0
    backup = path.with_name(f"{path.name}.before-kinds-{uuid4().hex}.bak")
    temporary = path.with_name(f".pa-{uuid4().hex}.tmp")
    try:
        with backup.open("xb") as handle:
            os.chmod(backup, 0o600)
            handle.write(original)
            handle.flush()
            os.fsync(handle.fileno())
        with temporary.open("x", encoding="utf-8") as handle:
            os.chmod(temporary, 0o600)
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if path.read_bytes() != original:
            raise RuntimeError("Reminder journal changed during migration; stop the service first")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return changed


def main() -> None:
    """Run only while the service owning this journal is stopped."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("journal", type=Path, help="Path to {metadata_dir}/pa/chat_inbox.jsonl; stop ReMe first")
    args = parser.parse_args()
    print(f"Migrated {migrate(args.journal)} journal rows; originals are retained in a .bak file")


if __name__ == "__main__":
    main()
