#!/usr/bin/env python3
"""Drive the LongMemEval memory pipeline across all samples.

For every workspace under ``datasets/longmemeval/<idx>`` this launches one or more
``reme start config=jinli_lme job=<job>`` runs with ``LME_WORKSPACE_DIR`` pointed
at that sample. The pipeline jobs, in order, are:

  1. auto_memory   — distil every raw session into a daily note (``daily/*.md``)
  2. update_index  — clear the store and rebuild the index over ``daily/*.md``
  3. agentic_answer — read ``query.json`` and answer it, writing ``mem_answer.json``
  4. llm_judge      — judge ``mem_answer.json`` against ``answer.json``

Pick one with ``--job``, or ``--job all`` to run the full pipeline *serially per sample*.
Runs are capped at ``--concurrency`` (default 1 for ``--job auto_memory``, otherwise
3) samples at once and each launch is staggered by ``--stagger`` seconds so they
do not all hit the LLM API at once.

By default every selected job is rerun for every sample — each job's own clear
step (configured in jinli_lme.yaml) wipes stale output first, so a run is always
a clean rebuild. Pass ``--resume`` to instead skip samples whose output already
exists (``daily/`` for auto_memory, ``metadata/embedding_store/`` for
update_index, ``mem_answer.json`` for agentic_answer, ``mem_answer.json`` with
``llm_judge.judgement`` for llm_judge) and continue an interrupted batch. Each
sample's stdout/stderr goes to ``logs/agentic_answer/<job>/<idx>.log``.

After an agentic_answer run finishes, the driver aggregates every sample's query,
golden answer, predicted answer, LLM judgement and a best-effort tool-call trail
into one big JSON at ``logs/agentic_answer/aggregate.json``.

Examples:
    python benchmark/longmemeval/run_agentic_answer.py                       # agentic_answer, all 500, conc 3
    python benchmark/longmemeval/run_agentic_answer.py --job all             # full pipeline serially per sample
    python benchmark/longmemeval/run_agentic_answer.py --job auto_memory     # just step 1
    python benchmark/longmemeval/run_agentic_answer.py --job llm_judge       # just judge existing answers
    python benchmark/longmemeval/run_agentic_answer.py --limit 5 --dry-run   # list what would run
    python benchmark/longmemeval/run_agentic_answer.py --start 187           # samples 187..499
    python benchmark/longmemeval/run_agentic_answer.py --start 187 --end 499 # samples 187..499
    python benchmark/longmemeval/run_agentic_answer.py --job all --resume    # continue an interrupted batch
"""

import argparse
import asyncio
import json
import os
import re
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "datasets" / "longmemeval"
LOGDIR = REPO / "logs" / "agentic_answer"
AGGREGATE = LOGDIR / "aggregate.json"

# Pipeline jobs in execution order.
JOB_ORDER = ["auto_memory", "update_index", "agentic_answer", "llm_judge"]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--job",
        choices=[*JOB_ORDER, "all"],
        default="agentic_answer",
        help="which job to run per sample; 'all' runs the full pipeline serially (default: agentic_answer)",
    )
    p.add_argument("--concurrency", type=int, default=1, help="max samples running at once (default 3)")
    p.add_argument("--stagger", type=float, default=1.0, help="seconds between consecutive launches (default 1)")
    p.add_argument("--start", type=int, default=0, help="first numeric sample id to process, inclusive (default 0)")
    p.add_argument(
        "--end",
        type=int,
        default=0,
        help="last numeric sample id to process, inclusive (0 = no upper bound)",
    )
    p.add_argument("--limit", type=int, default=0, help="only process the first N samples (0 = all)")
    p.add_argument(
        "--resume",
        action="store_true",
        help="skip a sample when the job's output already exists (resume an interrupted run); "
        "by default every selected job is rerun so the config's clear step rebuilds cleanly",
    )
    p.add_argument("--dry-run", action="store_true", help="list what would run, launch nothing")
    p.add_argument("--no-aggregate", action="store_true", help="skip writing aggregate.json after answer/judge jobs")
    return p.parse_args()


def selected_jobs(job: str) -> list[str]:
    """Expand the --job choice into an ordered list of jobs."""
    return list(JOB_ORDER) if job == "all" else [job]


def sample_ids() -> list[str]:
    """List all sample IDs (numeric workspace dirs), numerically sorted."""
    ids = [p.name for p in DATA.iterdir() if p.is_dir() and p.name.isdigit()]
    return sorted(ids, key=int)


def job_done(idx: str, job: str) -> bool:
    """Return True when ``job``'s expected output already exists for sample ``idx``."""
    ws = DATA / idx
    if job == "auto_memory":
        daily = ws / "daily"
        return daily.is_dir() and any(daily.rglob("*.md"))
    if job == "update_index":
        store = ws / "metadata" / "embedding_store"
        return store.is_dir() and any(store.iterdir())
    if job == "agentic_answer":
        return (ws / "mem_answer.json").exists()
    if job == "llm_judge":
        judge = _load_json(ws / "mem_answer.json").get("llm_judge")
        return isinstance(judge, dict) and bool(str(judge.get("judgement") or "").strip())
    raise ValueError(f"unknown job: {job}")


async def run_job(idx: str, job: str, counters: dict) -> bool:
    """Run a single job for a single sample. Returns True on success."""
    log = LOGDIR / job / f"{idx}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, LME_WORKSPACE_DIR=f"datasets/longmemeval/{idx}")
    started = time.strftime("%H:%M:%S")
    print(f"[start {started}] {idx}/{job}", flush=True)
    with log.open("w", encoding="utf-8") as f:
        proc = await asyncio.create_subprocess_exec(
            "reme",
            "start",
            "config=jinli_lme",
            f"job={job}",
            cwd=str(REPO),
            env=env,
            stdout=f,
            stderr=asyncio.subprocess.STDOUT,
        )
        rc = await proc.wait()
    ok = rc == 0 and job_done(idx, job)
    counters["done" if ok else "fail"] += 1
    tag = "done" if ok else "fail"
    print(f"[{tag}] {idx}/{job} rc={rc} ({counters['done']} done / {counters['fail']} fail)", flush=True)
    return ok


async def run_one(idx: str, jobs: list[str], sem: asyncio.Semaphore, resume: bool, counters: dict) -> None:
    """Run the selected jobs for one sample, serially.

    By default every selected job is rerun (the job's own clear step wipes stale
    output first). With ``resume`` a job is skipped when its output already
    exists, so an interrupted batch can continue without redoing finished work.
    """
    async with sem:
        for job in jobs:
            if resume and job_done(idx, job):
                counters["skip"] += 1
                print(f"[skip] {idx}/{job} (output exists)", flush=True)
                continue
            ok = await run_job(idx, job, counters)
            if not ok:
                # Later jobs depend on earlier ones; don't waste a run on a broken workspace.
                print(f"[abort] {idx}: {job} failed, skipping remaining jobs", flush=True)
                break


# --------------------------------------------------------------------------- #
# Aggregation of agentic_answer results into one big JSON.
# --------------------------------------------------------------------------- #

# Match ``session_id=abc123`` headers and ``"...session_id": "abc123"`` fields in
# tool-result text, so we can list which sessions each search actually surfaced.
_SID_RE = re.compile(r'session_id["\s:=]+"?([A-Za-z0-9_\-]+)')


def _load_json(path: Path) -> dict:
    """Load a JSON object, returning {} on any error."""
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def parse_tool_calls(idx: str, session_id: str) -> list[dict]:
    """Best-effort: parse the agent trajectory into an ordered tool-call summary.

    Reads ``mem_session/agentscope/<session_id>.jsonl`` — the trajectory the
    agentic_answer run dumped — and pairs every ``tool_call`` (name + parsed
    args) with the ``session_id`` hits found in its ``tool_result``. Returns an
    empty list if the file is missing or unreadable (never raises).
    """
    if not session_id:
        return []
    path = DATA / idx / "mem_session" / "agentscope" / f"{session_id}.jsonl"
    if not path.exists():
        return []

    calls: dict[str, dict] = {}
    order: list[str] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            for c in msg.get("content") or []:
                if not isinstance(c, dict):
                    continue
                cid = c.get("id")
                if c.get("type") == "tool_call" and cid:
                    try:
                        args = json.loads(c.get("input") or "{}")
                    except (json.JSONDecodeError, TypeError):
                        args = c.get("input")
                    calls[cid] = {"name": c.get("name"), "args": args, "hit_session_ids": []}
                    order.append(cid)
                elif c.get("type") == "tool_result" and cid in calls:
                    text = ""
                    for o in c.get("output") or []:
                        if isinstance(o, dict) and isinstance(o.get("text"), str):
                            text += o["text"]
                    hits = list(dict.fromkeys(_SID_RE.findall(text)))
                    calls[cid]["hit_session_ids"] = hits
    except OSError:
        return []

    return [{"iter": i + 1, **calls[cid]} for i, cid in enumerate(order)]


def build_record(idx: str) -> dict:
    """Assemble one sample's aggregate record from its on-disk artifacts."""
    ws = DATA / idx
    query = _load_json(ws / "query.json")
    golden = _load_json(ws / "answer.json")
    mem = _load_json(ws / "mem_answer.json")

    pred = str(mem.get("answer") or "").strip()
    session_id = str(mem.get("session_id") or "")
    llm_judge = mem.get("llm_judge") if isinstance(mem.get("llm_judge"), dict) else {}
    tool_calls = parse_tool_calls(idx, session_id) if mem else []

    if not mem:
        status = "missing"
    elif not pred:
        status = "empty"
    elif "not provided" in pred.lower():
        status = "not_provided"
    else:
        status = "answered"

    return {
        "idx": idx,
        "question_id": query.get("question_id"),
        "question_type": query.get("question_type"),
        "question": query.get("question"),
        "question_date": query.get("question_date"),
        "golden_answer": golden.get("answer"),
        "golden_answer_session_ids": golden.get("answer_session_ids"),
        "pred_answer": pred,
        "session_id": session_id,
        "status": status,
        "llm_judge": llm_judge.get("judgement"),
        "llm_judge_raw": llm_judge.get("raw_judgement"),
        "num_tool_calls": len(tool_calls),
        "tool_calls": tool_calls,
    }


def write_aggregate(ids: list[str]) -> None:
    """Aggregate every sample's agentic_answer artifacts into one big JSON."""
    records = [build_record(idx) for idx in ids]
    finished = [r for r in records if r["status"] != "missing"]
    by_status: dict[str, int] = {}
    by_llm_judge: dict[str, int] = {}
    for r in records:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
        judgement = r.get("llm_judge") or "missing"
        by_llm_judge[judgement] = by_llm_judge.get(judgement, 0) + 1

    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total": len(records),
        "finished": len(finished),
        "by_status": by_status,
        "by_llm_judge": by_llm_judge,
        "samples": records,
    }
    AGGREGATE.parent.mkdir(parents=True, exist_ok=True)
    AGGREGATE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[aggregate] wrote {len(records)} samples ({len(finished)} finished) -> {AGGREGATE}", flush=True)


async def main() -> int:
    """Run the driver."""
    args = parse_args()
    LOGDIR.mkdir(parents=True, exist_ok=True)
    jobs = selected_jobs(args.job)

    ids = sample_ids()
    if args.end and args.end < args.start:
        raise ValueError(f"--end ({args.end}) must be >= --start ({args.start})")
    ids = [i for i in ids if int(i) >= args.start and (not args.end or int(i) <= args.end)]
    if args.limit:
        ids = ids[: args.limit]

    # Without --resume every job reruns; with --resume, jobs whose output exists are skipped.
    def todo_jobs(i: str) -> list[str]:
        return [j for j in jobs if not (args.resume and job_done(i, j))]

    pending = [i for i in ids if todo_jobs(i)]
    print(
        f"jobs={jobs} resume={args.resume} samples total={len(ids)} pending={len(pending)} "
        f"concurrency={args.concurrency} stagger={args.stagger}s",
        flush=True,
    )

    if args.dry_run:
        for i in pending:
            print(f"[would-run] {i}: {todo_jobs(i)}")
        return 0

    sem = asyncio.Semaphore(args.concurrency)
    counters = {"done": 0, "fail": 0, "skip": 0}
    tasks: list[asyncio.Task] = []
    for n, idx in enumerate(ids):
        if n and args.stagger > 0:
            await asyncio.sleep(args.stagger)  # stagger each launch relative to the previous
        tasks.append(asyncio.create_task(run_one(idx, jobs, sem, args.resume, counters)))

    await asyncio.gather(*tasks, return_exceptions=True)
    print(
        f"ALL FINISHED done={counters['done']} fail={counters['fail']} skip={counters['skip']}",
        flush=True,
    )

    if any(j in jobs for j in ("agentic_answer", "llm_judge")) and not args.no_aggregate:
        write_aggregate(ids)

    return 0 if counters["fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
