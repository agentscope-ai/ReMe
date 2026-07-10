#!/usr/bin/env python3
"""Remove generated LongMemEval files while keeping source inputs.

For each ``datasets/longmemeval/<idx>`` workspace, this keeps only:
  - query.json
  - answer.json
  - session/

All other files or directories in the sample root are considered generated
artifacts and can be removed. AppleDouble files whose names start with ``._``
are also removed recursively, including under ``session/``. The script is
dry-run by default; pass ``--apply`` to actually delete. To delete only specific
root-level generated files, pass one or more ``--filename`` values.

Examples:
    python benchmark/longmemeval/clean_sample_outputs.py
    python benchmark/longmemeval/clean_sample_outputs.py --apply
    python benchmark/longmemeval/clean_sample_outputs.py --start 36 --end 79 --apply
    python benchmark/longmemeval/clean_sample_outputs.py --filename check_golden.json --apply
    python benchmark/longmemeval/clean_sample_outputs.py --filename session_review.json --apply
"""

import argparse
import shutil
import time
from collections.abc import Iterator
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
    p.add_argument("--progress-every", type=int, default=25, help="print progress every N samples when applying")
    p.add_argument(
        "--filename",
        action="append",
        default=[],
        help="delete only this root-level file or directory name; can be passed multiple times",
    )
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


def iter_sample_targets(sample_dir: Path, filenames: set[str] | None = None) -> Iterator[Path]:
    """Yield generated artifacts for one sample.

    Root-level generated directories are yielded as a whole, so there is no
    need to recurse into them. AppleDouble files are only searched inside the
    kept ``session/`` directory.
    """
    if filenames:
        for name in sorted(filenames):
            path = sample_dir / name
            if path.exists():
                yield path
        return

    for path in sorted(sample_dir.iterdir(), key=lambda p: p.name):
        if path.name not in KEEP:
            yield path

    session_dir = sample_dir / "session"
    if session_dir.is_dir():
        yield from session_dir.rglob("._*")


def main() -> int:
    """Main entry point."""
    args = parse_args()
    if args.end < args.start:
        raise ValueError(f"--end ({args.end}) must be >= --start ({args.start})")
    filenames = {name.strip() for name in args.filename if name.strip()}
    invalid_filenames = [name for name in filenames if Path(name).name != name]
    if invalid_filenames:
        raise ValueError(f"--filename only accepts root-level names, got: {invalid_filenames}")

    ids = [idx for idx in sample_ids() if args.start <= int(idx) <= args.end]
    if args.limit:
        ids = ids[: args.limit]

    total_targets = 0
    deleted = 0
    started_at = time.time()
    for ordinal, idx in enumerate(ids, start=1):
        sample_dir = DATA / idx
        sample_started_at = time.time()
        targets = list(iter_sample_targets(sample_dir, filenames=filenames))
        total_targets += len(targets)
        print(f"[sample {ordinal}/{len(ids)}] {idx} targets={len(targets)}", flush=True)
        for path in targets:
            if args.apply:
                target_started_at = time.time()
                print(f"[delete] {path}", flush=True)
                delete_path(path)
                deleted += 1
                print(f"[deleted] {path} elapsed={time.time() - target_started_at:.1f}s", flush=True)
            else:
                print(f"[would-delete] {path}")
        if args.apply and args.progress_every > 0 and (int(idx) + 1) % args.progress_every == 0:
            elapsed = time.time() - started_at
            print(
                f"[progress] processed={ordinal}/{len(ids)} through={idx} " f"deleted={deleted} elapsed={elapsed:.1f}s",
                flush=True,
            )
        print(f"[sample-done] {idx} elapsed={time.time() - sample_started_at:.1f}s", flush=True)

    mode = "DELETE" if args.apply else "DRY-RUN"
    print(
        f"{mode} LongMemEval generated artifacts: samples={len(ids)} "
        f"targets={total_targets} deleted={deleted if args.apply else 0} range={args.start}..{args.end}",
        flush=True,
    )

    if not args.apply:
        print("No files deleted. Re-run with --apply to delete these paths.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
