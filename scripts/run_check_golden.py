#!/usr/bin/env python3
"""Drive `reme start config=jinli_lme job=check_golden` across all LongMemEval samples.

For every workspace under ``datasets/longmemeval/<idx>`` this launches the
``check_golden`` job with ``LME_WORKSPACE_DIR`` pointed at that sample. Runs are
capped at ``--concurrency`` (default 3) and each launch is staggered by
``--stagger`` seconds relative to the previous one (the 2nd starts 1s later, the
3rd 2s later, ... ) so they do not all hit the LLM API at the same instant.

Samples that already produced ``check_golden.json`` are skipped unless ``--force``
is given, so the run is resumable. Each sample's stdout/stderr goes to
``logs/check_golden/<idx>.log``.

Examples:
    python scripts/run_check_golden.py                 # all 500, concurrency 3
    python scripts/run_check_golden.py --dry-run       # just list what would run
    python scripts/run_check_golden.py --limit 5       # first 5 samples only
    python scripts/run_check_golden.py --concurrency 3 --stagger 1
"""

import argparse
import asyncio
import os
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "datasets" / "longmemeval"
LOGDIR = REPO / "logs" / "check_golden"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--concurrency", type=int, default=4, help="max samples running at once (default 3)")
    p.add_argument("--stagger", type=float, default=1.0, help="seconds between consecutive launches (default 1)")
    p.add_argument("--limit", type=int, default=0, help="only process the first N samples (0 = all)")
    p.add_argument("--force", action="store_true", help="rerun even if check_golden.json already exists")
    p.add_argument("--dry-run", action="store_true", help="list what would run, launch nothing")
    return p.parse_args()


def sample_ids() -> list[str]:
    """List all sample IDs."""
    ids = [p.name for p in DATA.iterdir() if p.is_dir() and p.name.isdigit()]
    return sorted(ids, key=int)


async def run_one(idx: str, sem: asyncio.Semaphore, force: bool, counters: dict) -> None:
    """Run one sample."""
    out = DATA / idx / "check_golden.json"
    if not force and out.exists():
        counters["skip"] += 1
        print(f"[skip] {idx} (check_golden.json exists)", flush=True)
        return

    async with sem:
        log = LOGDIR / f"{idx}.log"
        env = dict(os.environ, LME_WORKSPACE_DIR=f"datasets/longmemeval/{idx}")
        started = time.strftime("%H:%M:%S")
        print(f"[start {started}] {idx}", flush=True)
        with log.open("w", encoding="utf-8") as f:
            proc = await asyncio.create_subprocess_exec(
                "reme",
                "start",
                "config=jinli_lme",
                "job=check_golden",
                cwd=str(REPO),
                env=env,
                stdout=f,
                stderr=asyncio.subprocess.STDOUT,
            )
            rc = await proc.wait()
        ok = rc == 0 and out.exists()
        counters["done" if ok else "fail"] += 1
        tag = "done" if ok else "fail"
        print(f"[{tag}] {idx} rc={rc} ({counters['done']} done / {counters['fail']} fail)", flush=True)


async def main() -> int:
    """Run the script."""
    args = parse_args()
    LOGDIR.mkdir(parents=True, exist_ok=True)

    ids = sample_ids()
    if args.limit:
        ids = ids[: args.limit]

    pending = [i for i in ids if args.force or not (DATA / i / "check_golden.json").exists()]
    print(
        f"samples total={len(ids)} pending={len(pending)} " f"concurrency={args.concurrency} stagger={args.stagger}s",
        flush=True,
    )

    if args.dry_run:
        for i in pending:
            print(f"[would-run] {i}")
        return 0

    sem = asyncio.Semaphore(args.concurrency)
    counters = {"done": 0, "fail": 0, "skip": 0}
    tasks: list[asyncio.Task] = []
    for n, idx in enumerate(ids):
        if n and args.stagger > 0:
            await asyncio.sleep(args.stagger)  # stagger each launch relative to the previous
        tasks.append(asyncio.create_task(run_one(idx, sem, args.force, counters)))

    await asyncio.gather(*tasks, return_exceptions=True)
    print(f"ALL FINISHED done={counters['done']} fail={counters['fail']} skip={counters['skip']}", flush=True)
    return 0 if counters["fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
