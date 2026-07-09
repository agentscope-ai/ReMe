#!/usr/bin/env python3
"""Run LongMemEval ``session_review`` concurrently across samples.

For every workspace under ``datasets/longmemeval/<idx>`` in the selected numeric
range, this launches:

    reme start config=jinli_lme job=session_review

with ``LME_WORKSPACE_DIR`` pointed at that sample. Multiple samples can run at
once, capped by ``--concurrency``. Request submission inside ``session_review``
is globally throttled across these processes by the step itself. Each sample's
stdout/stderr goes to ``logs/session_review/<idx>.log``.

By default the script processes samples 0..499 inclusive and reruns every sample
in that range. Pass ``--resume`` to skip samples whose ``session_review.json``
already exists.

Examples:
    python benchmark/longmemeval/run_session_review.py
    python benchmark/longmemeval/run_session_review.py --start 187 --end 499
    python benchmark/longmemeval/run_session_review.py --concurrency 3
    python benchmark/longmemeval/run_session_review.py --resume
    python benchmark/longmemeval/run_session_review.py --limit 5 --dry-run
"""

import argparse
import asyncio
import os
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "datasets" / "longmemeval"
LOGDIR = REPO / "logs" / "session_review"
OUTPUT_FILENAME = "session_review.json"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", type=int, default=0, help="first numeric sample id to process, inclusive (default 0)")
    p.add_argument("--end", type=int, default=499, help="last numeric sample id to process, inclusive (default 499)")
    p.add_argument("--limit", type=int, default=0, help="only process the first N selected samples (0 = all)")
    p.add_argument("--concurrency", type=int, default=3, help="max samples running at once (default 3)")
    p.add_argument("--stagger", type=float, default=1.0, help="seconds between worker launches (default 1)")
    p.add_argument(
        "--resume",
        action="store_true",
        help=f"skip samples whose {OUTPUT_FILENAME} already exists",
    )
    p.add_argument("--dry-run", action="store_true", help="list what would run, launch nothing")
    p.add_argument("--stop-on-fail", action="store_true", help="stop immediately after the first failed sample")
    return p.parse_args()


def sample_ids() -> list[str]:
    """List all sample IDs (numeric workspace dirs), numerically sorted."""
    ids = [p.name for p in DATA.iterdir() if p.is_dir() and p.name.isdigit()]
    return sorted(ids, key=int)


def output_exists(idx: str) -> bool:
    """Return True when the sample already has a session review artifact."""
    return (DATA / idx / OUTPUT_FILENAME).exists()


async def run_one(idx: str, active: set[str]) -> bool:
    """Run ``session_review`` for one sample. Returns True on success."""
    log = LOGDIR / f"{idx}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, LME_WORKSPACE_DIR=f"datasets/longmemeval/{idx}")

    started = time.strftime("%H:%M:%S")
    print(f"[start {started}] {idx}", flush=True)
    active.add(idx)
    try:
        with log.open("w", encoding="utf-8") as f:
            proc = await asyncio.create_subprocess_exec(
                "reme",
                "start",
                "config=jinli_lme",
                "job=session_review",
                cwd=str(REPO),
                env=env,
                stdout=f,
                stderr=asyncio.subprocess.STDOUT,
            )
            rc = await proc.wait()
    finally:
        active.discard(idx)

    ok = rc == 0 and output_exists(idx)
    tag = "done" if ok else "fail"
    print(f"[{tag}] {idx} rc={rc} log={log}", flush=True)
    return ok


async def worker(
    name: int,
    queue: asyncio.Queue[str],
    args: argparse.Namespace,
    counters: dict[str, int],
    active: set[str],
    stop: asyncio.Event,
) -> None:
    """Run samples from ``queue`` until exhausted or fail-fast is triggered."""
    if name and args.stagger > 0:
        await asyncio.sleep(args.stagger * name)

    while not stop.is_set():
        try:
            idx = queue.get_nowait()
        except asyncio.QueueEmpty:
            return

        try:
            if args.resume and output_exists(idx):
                counters["skip"] += 1
                print(f"[skip] {idx} ({OUTPUT_FILENAME} exists)", flush=True)
                continue

            if await run_one(idx, active):
                counters["done"] += 1
            else:
                counters["fail"] += 1
                if args.stop_on_fail:
                    stop.set()
        finally:
            queue.task_done()


async def main() -> int:
    """Run the concurrent driver."""
    args = parse_args()
    if args.end < args.start:
        raise ValueError(f"--end ({args.end}) must be >= --start ({args.start})")
    if args.concurrency < 1:
        raise ValueError("--concurrency must be >= 1")
    if args.stagger < 0:
        raise ValueError("--stagger must be >= 0")

    LOGDIR.mkdir(parents=True, exist_ok=True)

    ids = [i for i in sample_ids() if args.start <= int(i) <= args.end]
    if args.limit:
        ids = ids[: args.limit]

    pending = [i for i in ids if not (args.resume and output_exists(i))]
    print(
        f"job=session_review samples total={len(ids)} pending={len(pending)} "
        f"range={args.start}..{args.end} resume={args.resume} "
        f"concurrency={args.concurrency} stagger={args.stagger}s",
        flush=True,
    )

    if args.dry_run:
        for idx in pending:
            print(f"[would-run] {idx}")
        return 0

    counters: dict[str, int] = {"done": 0, "fail": 0, "skip": 0}
    active: set[str] = set()
    stop = asyncio.Event()
    queue: asyncio.Queue[str] = asyncio.Queue()
    for idx in ids:
        queue.put_nowait(idx)

    workers = [
        asyncio.create_task(worker(n, queue, args, counters, active, stop))
        for n in range(min(args.concurrency, len(ids)))
    ]
    await asyncio.gather(*workers)

    print(
        f"ALL FINISHED done={counters['done']} fail={counters['fail']} skip={counters['skip']}",
        flush=True,
    )
    return 0 if counters["fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
