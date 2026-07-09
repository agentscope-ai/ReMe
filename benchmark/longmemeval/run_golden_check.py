#!/usr/bin/env python3
"""Run LongMemEval ``golden_check`` concurrently across samples.

For every workspace under ``datasets/longmemeval/<idx>`` in the selected numeric
range, this launches:

    reme start config=jinli_lme job=golden_check

with ``LME_WORKSPACE_DIR`` pointed at that sample. Multiple samples can run at
once, capped by ``--concurrency``. The ``golden_check`` job itself waits for
``session_review.json`` when configured with ``wait_for_paths_step`` in
``jinli_lme.yaml``. Each sample's stdout/stderr goes to
``logs/golden_check/<idx>.log``.

By default the script processes samples 0..499 inclusive and reruns every sample
in that range. Pass ``--resume`` to skip samples whose ``check_golden.json``
already exists.

Examples:
    python benchmark/longmemeval/run_golden_check.py
    python benchmark/longmemeval/run_golden_check.py --start 187 --end 499
    python benchmark/longmemeval/run_golden_check.py --concurrency 8 --stagger 1
    python benchmark/longmemeval/run_golden_check.py --resume
    python benchmark/longmemeval/run_golden_check.py --limit 5 --dry-run
"""

import argparse
import asyncio
import os
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "datasets" / "longmemeval"
LOGDIR = REPO / "logs" / "golden_check"
OUTPUT_FILENAME = "check_golden.json"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", type=int, default=0, help="first numeric sample id to process, inclusive (default 0)")
    p.add_argument("--end", type=int, default=499, help="last numeric sample id to process, inclusive (default 499)")
    p.add_argument("--limit", type=int, default=0, help="only process the first N selected samples (0 = all)")
    p.add_argument("--concurrency", type=int, default=3, help="max samples running at once (default 3)")
    p.add_argument("--stagger", type=float, default=1.0, help="seconds between consecutive launches (default 1)")
    p.add_argument(
        "--resume",
        action="store_true",
        help=f"skip samples whose {OUTPUT_FILENAME} already exists",
    )
    p.add_argument("--dry-run", action="store_true", help="list what would run, launch nothing")
    return p.parse_args()


def sample_ids() -> list[str]:
    """List all sample IDs (numeric workspace dirs), numerically sorted."""
    ids = [p.name for p in DATA.iterdir() if p.is_dir() and p.name.isdigit()]
    return sorted(ids, key=int)


def output_exists(idx: str) -> bool:
    """Return True when the sample already has a golden-check artifact."""
    return (DATA / idx / OUTPUT_FILENAME).exists()


async def run_one(idx: str, sem: asyncio.Semaphore, resume: bool, counters: dict) -> None:
    """Run ``golden_check`` for one sample."""
    if resume and output_exists(idx):
        counters["skip"] += 1
        print(f"[skip] {idx} ({OUTPUT_FILENAME} exists)", flush=True)
        return

    async with sem:
        log = LOGDIR / f"{idx}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, LME_WORKSPACE_DIR=f"datasets/longmemeval/{idx}")

        started = time.strftime("%H:%M:%S")
        print(f"[start {started}] {idx}", flush=True)
        with log.open("w", encoding="utf-8") as f:
            proc = await asyncio.create_subprocess_exec(
                "reme",
                "start",
                "config=jinli_lme",
                "job=golden_check",
                cwd=str(REPO),
                env=env,
                stdout=f,
                stderr=asyncio.subprocess.STDOUT,
            )
            rc = await proc.wait()

        ok = rc == 0 and output_exists(idx)
        counters["done" if ok else "fail"] += 1
        tag = "done" if ok else "fail"
        print(
            f"[{tag}] {idx} rc={rc} log={log} ({counters['done']} done / {counters['fail']} fail)",
            flush=True,
        )


async def main() -> int:
    """Run the concurrent driver."""
    args = parse_args()
    if args.end < args.start:
        raise ValueError(f"--end ({args.end}) must be >= --start ({args.start})")
    if args.concurrency < 1:
        raise ValueError("--concurrency must be >= 1")

    LOGDIR.mkdir(parents=True, exist_ok=True)

    ids = [i for i in sample_ids() if args.start <= int(i) <= args.end]
    if args.limit:
        ids = ids[: args.limit]

    pending = [i for i in ids if not (args.resume and output_exists(i))]
    print(
        f"job=golden_check samples total={len(ids)} pending={len(pending)} "
        f"range={args.start}..{args.end} resume={args.resume} "
        f"concurrency={args.concurrency} stagger={args.stagger}s",
        flush=True,
    )

    if args.dry_run:
        for idx in pending:
            print(f"[would-run] {idx}")
        return 0

    sem = asyncio.Semaphore(args.concurrency)
    counters = {"done": 0, "fail": 0, "skip": 0}
    tasks: list[asyncio.Task] = []
    for n, idx in enumerate(ids):
        if n and args.stagger > 0:
            await asyncio.sleep(args.stagger)
        tasks.append(asyncio.create_task(run_one(idx, sem, args.resume, counters)))

    await asyncio.gather(*tasks)
    print(
        f"ALL FINISHED done={counters['done']} fail={counters['fail']} skip={counters['skip']}",
        flush=True,
    )
    return 0 if counters["fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
