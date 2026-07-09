#!/usr/bin/env python3
"""Remove generated LongMemEval files while keeping source inputs.

For each ``datasets/longmemeval/<idx>`` workspace, this keeps only:
  - query.json
  - answer.json
  - session/

All other files or directories in the sample root are considered generated
artifacts and can be removed. AppleDouble files whose names start with ``._``
are also removed recursively, including under ``session/``. The script is
dry-run by default; pass ``--apply`` to actually delete.

Examples:
    python benchmark/longmemeval/clean_sample_outputs.py
    python benchmark/longmemeval/clean_sample_outputs.py --apply
    python benchmark/longmemeval/clean_sample_outputs.py --start 36 --end 79 --apply
"""

import argparse
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "datasets" / "longmemeval"
KEEP = {"query.json", "answer.json", "session"}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", type=int, default=0, help="first numeric sample id to clean, inclusive (default 0)")
    p.add_argument("--end", type=int, default=499, help="last numeric sample id to clean, inclusive (default 499)")
    p.add_argument("--limit", type=int, default=0, help="only clean the first N selected samples (0 = all)")
    p.add_argument("--apply", action="store_true", help="actually delete files; default is dry-run")
    return p.parse_args()


def sample_ids() -> list[str]:
    """List all numeric sample IDs."""
    ids = [p.name for p in DATA.iterdir() if p.is_dir() and p.name.isdigit()]
    return sorted(ids, key=int)


def delete_path(path: Path) -> None:
    """Delete a file, symlink, or directory."""
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def is_under(path: Path, parent: Path) -> bool:
    """Return True when ``path`` is inside ``parent``."""
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def main() -> int:
    """Main entry point."""
    args = parse_args()
    if args.end < args.start:
        raise ValueError(f"--end ({args.end}) must be >= --start ({args.start})")

    ids = [idx for idx in sample_ids() if args.start <= int(idx) <= args.end]
    if args.limit:
        ids = ids[: args.limit]

    targets: list[Path] = []
    target_set: set[Path] = set()
    for idx in ids:
        sample_dir = DATA / idx
        for path in sorted(sample_dir.iterdir(), key=lambda p: p.name):
            if path.name not in KEEP:
                targets.append(path)
                target_set.add(path)
        for path in sorted(sample_dir.rglob("._*")):
            if path not in target_set and not any(is_under(path, target) for target in targets):
                targets.append(path)
                target_set.add(path)

    mode = "DELETE" if args.apply else "DRY-RUN"
    print(
        f"{mode} LongMemEval generated artifacts: samples={len(ids)} "
        f"targets={len(targets)} range={args.start}..{args.end}",
        flush=True,
    )
    for path in targets:
        print(path)
        if args.apply:
            delete_path(path)

    if not args.apply:
        print("No files deleted. Re-run with --apply to delete these paths.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
