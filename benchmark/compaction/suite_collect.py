"""Freeze the fourteen-task oracle panel, then collect real read-only Hermes trajectories."""

import argparse
import asyncio
import hashlib
from pathlib import Path
import shutil

from benchmark.compaction.cases import SOURCE_SHA256, write
from benchmark.compaction.suite import freeze


async def run(args):
    """Run isolated finite subprocesses; preserve every planned task and failure record."""
    repo = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    if output.exists() or output == repo or repo in output.parents:
        raise ValueError("Use a new directory outside repository")
    payment = args.payment_trajectory.read_bytes()
    if hashlib.sha256(payment).hexdigest() != SOURCE_SHA256:
        raise ValueError("Expected the pinned original payment source")
    manifest = freeze(args.hermes.resolve(), output)
    shutil.copy2(args.payment_trajectory, output / "payment/trajectory.full.json")
    write(output / "payment/collection.json", {"completed": True, "reused": True, "trajectory_sha256": SOURCE_SHA256})
    for name in ("suite.py", "suite_collect.py", "hermes_worker.py", "README.md"):
        shutil.copy2(Path(__file__).parent / name, output / name)
    semaphore = asyncio.Semaphore(args.concurrency)
    statuses = []

    async def collect(case):
        if case["id"] == "payment":
            return
        root = output / case["id"]
        async with semaphore:
            with (root / "worker.log").open("w") as log:
                process = await asyncio.create_subprocess_exec(
                    str(args.hermes_python),
                    "-m",
                    "benchmark.compaction.hermes_worker",
                    "--root",
                    str(root),
                    "--hermes",
                    str(args.hermes.resolve()),
                    "--env-file",
                    str(repo / ".env"),
                    cwd=repo,
                    stdout=log,
                    stderr=log,
                )
                try:
                    code = await asyncio.wait_for(process.wait(), 360)
                except BaseException:
                    if process.returncode is None:
                        process.terminate()
                        try:
                            await asyncio.wait_for(process.wait(), 10)
                        except asyncio.TimeoutError:
                            process.kill()
                            await process.wait()
                    raise
            statuses.append({"id": case["id"], "completed": code == 0, "returncode": code})
            write(output / "collection-status.json", statuses)
            print(case["id"], "collected" if code == 0 else "FAILED", flush=True)

    async with asyncio.TaskGroup() as tasks:
        for case in manifest["cases"]:
            tasks.create_task(collect(case))
    if any(not s["completed"] for s in statuses):
        raise RuntimeError("Frozen collection failures remain; inspect retained logs")


def main():
    """Only explicit live collection can contact the configured provider."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hermes", type=Path, required=True)
    parser.add_argument("--hermes-python", type=Path, required=True)
    parser.add_argument("--payment-trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    if not args.live or args.concurrency < 1:
        parser.error("Use --live and positive concurrency")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
