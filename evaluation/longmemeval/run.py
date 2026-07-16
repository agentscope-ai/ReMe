"""LongMemEval evaluation runner for ReMe.

Evaluates ReMe's long-term memory capability using the LongMemEval dataset.
Each item gets an isolated workspace; sessions are ingested in chronological order;
dream is triggered when sessions cross midnight (23:00); finally questions are
answered via search and judged by an LLM.

Usage:
    python evaluation/longmemeval/run.py
    python evaluation/longmemeval/run.py --config evaluation/longmemeval/config.yaml
    python evaluation/longmemeval/run.py -q                          # quiet: only eval-level logs
    python evaluation/longmemeval/run.py --log-level WARNING         # reduce eval runner logs
    python evaluation/longmemeval/run.py --reme-log-level WARNING    # reduce reme internal logs
    python evaluation/longmemeval/run.py --eval_only                 # query+judge only, reuse existing workspace
"""

import json
import logging
import os
import re
import shutil
import time
import threading
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Load .env from project root
_PROJECT_ROOT = Path(__file__).parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# Workspace root for evaluation items
_WORKSPACE_ROOT = _PROJECT_ROOT / "memory_workspaces/longmemeval-s"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"

logging.basicConfig(level=logging.INFO, format=_DEFAULT_LOG_FORMAT)
logger = logging.getLogger("longmemeval")

# Noisy library loggers silenced by default
_NOISY_LOGGERS = [
    "httpx", "httpcore", "openai", "uvicorn", "multipart",
    "asyncio", "watchfiles", "filelock",
]


def setup_logging(log_level: str, reme_log_level: str):
    """Configure logging for the eval runner and reme internals.

    Args:
        log_level: Level for the eval runner logger (DEBUG/INFO/WARNING/ERROR).
        reme_log_level: Level for reme's internal loguru logger.
    """
    numeric = getattr(logging, log_level.upper(), logging.INFO)
    # Eval runner logger
    logging.getLogger().setLevel(numeric)
    logger.setLevel(numeric)

    # Suppress noisy library loggers when above DEBUG
    if numeric > logging.DEBUG:
        for name in _NOISY_LOGGERS:
            lib_logger = logging.getLogger(name)
            lib_logger.setLevel(max(numeric, logging.WARNING))

    # Reme internal logger (loguru) — will be applied per-worker via _configure_worker
    os.environ["REME_LOG_LEVEL"] = reme_log_level.upper()


def _configure_worker(log_level: str, reme_log_level: str):
    """Set up logging inside a multiprocessing worker process.

    Must be called at the top of each worker because child processes inherit
    parent state but loguru sinks are NOT shared across fork/spawn.
    """
    numeric = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(level=numeric, format=_DEFAULT_LOG_FORMAT, force=True)
    logging.getLogger("longmemeval").setLevel(numeric)
    if numeric > logging.DEBUG:
        for name in _NOISY_LOGGERS:
            logging.getLogger(name).setLevel(max(numeric, logging.WARNING))

    # Re-initialize loguru for reme internals at the desired level
    from reme.utils import get_logger
    get_logger(level=reme_log_level.upper(), force_init=True)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
def load_eval_config(config_path: str | None = None) -> dict:
    """Load evaluation config yaml with env-var expansion."""
    if config_path is None:
        config_path = str(Path(__file__).parent / "config.yaml")
    with open(config_path, encoding="utf-8") as f:
        raw = f.read()

    # Expand ${VAR} and ${VAR:-default}
    def _expand(m):
        expr = m.group(1)
        if ":-" in expr:
            key, default = expr.split(":-", 1)
            return os.environ.get(key, default)
        return os.environ.get(expr, "")

    raw = re.sub(r"\$\{([^}]+)\}", _expand, raw)
    return yaml.safe_load(raw)


# ---------------------------------------------------------------------------
# Date utilities
# ---------------------------------------------------------------------------
def parse_haystack_date(date_str: str) -> datetime:
    """Parse LongMemEval date format: '2023/05/20 (Sat) 02:21' -> datetime."""
    m = re.match(r"(\d{4}/\d{2}/\d{2})\s+\(\w+\)\s+(\d{2}:\d{2})", date_str)
    if not m:
        raise ValueError(f"Cannot parse haystack date: {date_str!r}")
    return datetime.strptime(f"{m.group(1)} {m.group(2)}", "%Y/%m/%d %H:%M")


def to_iso(dt: datetime) -> str:
    """Convert datetime to ISO-8601 string precise to seconds."""
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


def should_trigger_dream(prev_dt: datetime, curr_dt: datetime, _trigger_hour: int = 23) -> bool:
    """Check if the time gap between two sessions crosses trigger_hour (e.g. 23:00)."""
    if prev_dt.date() == curr_dt.date():
        return False
    # There's at least one midnight crossing; check if trigger_hour is between them
    # Simple heuristic: if dates differ, dream should run for the previous day
    return True


def sessions_sorted_by_time(item: dict) -> list[tuple[int, datetime, str, list[dict]]]:
    """Return (original_index, parsed_datetime, session_id, messages) sorted by time."""
    entries = []
    for i, (date_str, sid, msgs) in enumerate(
        zip(item["haystack_dates"], item["haystack_session_ids"], item["haystack_sessions"]),
    ):
        dt = parse_haystack_date(date_str)
        entries.append((i, dt, sid, msgs))
    # Sort by time (ascending)
    entries.sort(key=lambda x: x[1])
    return entries


# ---------------------------------------------------------------------------
# Message formatting
# ---------------------------------------------------------------------------
def format_messages_for_reme(messages: list[dict], session_dt: datetime) -> list[dict]:
    """Convert LongMemEval messages to ReMe auto_memory format.

    Adds: name, created_at (ISO seconds). All messages in a session share the
    same created_at (the session timestamp).
    """
    formatted = []
    for msg in messages:
        role = msg["role"]
        formatted.append(
            {
                "name": role,
                "role": role,
                "content": msg["content"],
                "created_at": to_iso(session_dt),
            },
        )
    return formatted


# ---------------------------------------------------------------------------
# LLM-as-Judge (delegated to answer_judge_step via app.run_job)
# ---------------------------------------------------------------------------


async def judge_response_via_job(
    app,
    question: str,
    ground_truth: str,
    response: str,
    question_type: str,
) -> dict:
    """Use the answer_judge_step to evaluate a response against the golden answer."""
    judge_resp = await app.run_job(
        "answer_judge",
        query=question,
        agent_answer=response,
        golden_answer=ground_truth,
        question_type=question_type,
    )

    verdict = (judge_resp.answer or "").strip().lower()
    raw_answer = (judge_resp.metadata or {}).get("raw_answer_judgement", "")

    return {
        "verdict": verdict,
        "reason": raw_answer if verdict not in ("yes", "no") else "",
        "metric": "binary",
        "question_type": question_type,
    }


# ---------------------------------------------------------------------------
# Main evaluation pipeline
# ---------------------------------------------------------------------------
async def evaluate_item(item: dict, eval_config: dict, item_index: int, eval_only: bool = False) -> dict:
    """Evaluate a single LongMemEval item end-to-end.

    Args:
        item: The dataset item containing question, answer, sessions, etc.
        eval_config: The evaluation configuration dict.
        item_index: The index of this item in the dataset.
        eval_only: If True, skip ingestion (phases 1-3) and only run query+judge
            using the existing workspace. Useful for re-evaluating different query
            configurations without re-ingesting sessions.
    """
    from reme import Application
    from reme.config import resolve_app_config

    reme_cfg = eval_config["reme"]
    dream_trigger_hour = reme_cfg.get("dream_trigger_hour", 23)
    dream_scan_days = reme_cfg.get("dream_scan_days", 2)
    dream_max_units = reme_cfg.get("dream_max_units", 5)

    # Sort sessions by time
    sorted_sessions = sessions_sorted_by_time(item)

    # Filter out sessions that occur after question_date (if enabled)
    filter_future = eval_config["evaluation"].get("filter_future_sessions", True)
    if filter_future and item.get("question_date"):
        question_dt = parse_haystack_date(item["question_date"])
        total_before_filter = len(sorted_sessions)
        sorted_sessions = [(i, dt, sid, msgs) for i, dt, sid, msgs in sorted_sessions if dt <= question_dt]
        if len(sorted_sessions) < total_before_filter:
            logger.info(
                f"[Item {item_index}] Filtered sessions: {total_before_filter} -> {len(sorted_sessions)} "
                f"(removed {total_before_filter - len(sorted_sessions)} future sessions "
                f"after question_date={item['question_date']})",
            )

    logger.info(
        f"[Item {item_index}] question_id={item['question_id']} "
        f"type={item['question_type']} sessions={len(sorted_sessions)}"
        + (" [eval_only]" if eval_only else ""),
    )

    # Use fixed workspace directory (clean it for fresh evaluation)
    item_dir = _WORKSPACE_ROOT / f"item_{item_index}"
    workspace_dir = str(item_dir / ".reme")
    if eval_only:
        if not item_dir.exists() or not Path(workspace_dir).exists():
            logger.warning(
                f"[Item {item_index}] eval_only: workspace not found at {item_dir}, skipping"
            )
            _skip_judgment = {"verdict": "no", "reason": "workspace missing in eval_only mode", "metric": "binary", "question_type": item["question_type"]}
            return {
                "question_id": item["question_id"],
                "question_type": item["question_type"],
                "question": item["question"],
                "ground_truth": item["answer"],
                "agentic_response": "(workspace missing, skipped)",
                "agentic_judgment": dict(_skip_judgment),
                "prompted_response": "(workspace missing, skipped)",
                "prompted_judgment": dict(_skip_judgment),
                "prompted_input_tokens": 0,
                "prompted_output_tokens": 0,
                "sessions_ingested": 0,
                "dreams_triggered": 0,
            }
    else:
        if item_dir.exists():
            shutil.rmtree(item_dir)
            logger.info(f"[Item {item_index}] Cleaned existing workspace: {item_dir}")
        item_dir.mkdir(parents=True, exist_ok=True)

    cfg = resolve_app_config(
        config=reme_cfg["config"],
        workspace_dir=workspace_dir,
        log_to_console=eval_config["output"].get("log_to_console", True),
        log_to_file=eval_config["output"].get("log_to_file", False),
        enable_logo=False,
    )

    app = Application(**cfg)
    await app.start()

    try:
        dream_dates_triggered = set()
        dream_available = True  # Set to False if auto_dream job is not found

        if not eval_only:
            # ── Phase 1: Ingest sessions ──────────────────────────────
            prev_dt = None

            for idx, (_, session_dt, session_id, messages) in enumerate(sorted_sessions):
                # Check if dream should be triggered before this session
                if dream_available and prev_dt is not None and should_trigger_dream(prev_dt, session_dt, dream_trigger_hour):
                    dream_date = prev_dt.strftime("%Y-%m-%d")
                    if dream_date not in dream_dates_triggered:
                        logger.info(f"[Item {item_index}] Triggering dream for date={dream_date}")
                        try:
                            dream_resp = await app.run_job(
                                "auto_dream",
                                date=dream_date,
                                scan_days=dream_scan_days,
                                max_units=dream_max_units,
                            )
                            logger.info(
                                f"[Item {item_index}] Dream done: success={dream_resp.success} "
                                f"answer={dream_resp.answer[:100] if dream_resp.answer else ''}",
                            )
                        except Exception as e:
                            if "not found" in str(e).lower():
                                dream_available = False
                                logger.warning(f"[Item {item_index}] auto_dream job not found, skipping all dreams")
                            else:
                                logger.warning(f"[Item {item_index}] Dream failed for {dream_date}: {e}")
                        dream_dates_triggered.add(dream_date)
                        # Index update after dream to pick up new digest nodes
                        await app.run_job("index_update")

                # Format and ingest the session
                formatted_msgs = format_messages_for_reme(messages, session_dt)
                date_str = session_dt.strftime("%Y-%m-%d")

                logger.info(
                    f"[Item {item_index}] Ingesting session {idx+1}/{len(sorted_sessions)} "
                    f"id={session_id} date={date_str} msgs={len(formatted_msgs)}",
                )
                resp = await app.run_job(
                    "auto_memory",
                    messages=formatted_msgs,
                    session_id=session_id,
                    date=date_str,
                )
                if not resp.success:
                    logger.warning(
                        f"[Item {item_index}] auto_memory failed for session {session_id}: {resp.answer}",
                    )

                # Manual index update after each session
                await app.run_job("index_update")

                prev_dt = session_dt

            # ── Phase 2: Final dream for the last day ─────────────────
            if dream_available and prev_dt is not None:
                last_dream_date = prev_dt.strftime("%Y-%m-%d")
                if last_dream_date not in dream_dates_triggered:
                    logger.info(f"[Item {item_index}] Final dream for date={last_dream_date}")
                    try:
                        await app.run_job(
                            "auto_dream",
                            date=last_dream_date,
                            scan_days=dream_scan_days,
                            max_units=dream_max_units,
                        )
                    except Exception as e:
                        if "not found" in str(e).lower():
                            dream_available = False
                            logger.warning(f"[Item {item_index}] auto_dream job not found, skipping all dreams")
                        else:
                            logger.warning(f"[Item {item_index}] Final dream failed: {e}")
                    dream_dates_triggered.add(last_dream_date)
                    # Index update after final dream
                    await app.run_job("index_update")

            # ── Phase 3: Digest update ────────────────────────────────
            await app.run_job("digest_update")

        # ── Phase 4: Ask question via agentic_answer job (ReAct agent) ──
        question = item["question"]
        question_date_raw = item.get("question_date", "")
        question_dt = parse_haystack_date(question_date_raw) if question_date_raw else None
        query_time = to_iso(question_dt) if question_dt else ""
        logger.info(
            f"[Item {item_index}] Asking (agentic): {question[:80]}... query_time={query_time}",
        )

        query_resp = await app.run_job(
            "agentic_answer",
            query=question,
            query_time=query_time,
        )
        agentic_response = (query_resp.answer or "").strip()
        if not agentic_response:
            agentic_response = "(no answer generated)"

        logger.info(f"[Item {item_index}] Agentic response: {agentic_response[:200]}...")

        # ── Phase 4b: Prompted answer (direct LLM with search context) ──
        logger.info(
            f"[Item {item_index}] Asking (prompted): {question[:80]}...",
        )
        prompted_resp = await app.run_job(
            "context_answer",
            question=question,
            query_time=query_time,
        )
        prompted_response = (prompted_resp.answer or "").strip()
        prompted_input_tokens = (prompted_resp.metadata or {}).get("input_tokens", 0)
        prompted_output_tokens = (prompted_resp.metadata or {}).get("output_tokens", 0)

        logger.info(
            f"[Item {item_index}] Prompted tokens: input={prompted_input_tokens} output={prompted_output_tokens}",
        )

        if not prompted_response:
            prompted_response = "(no answer generated)"

        logger.info(f"[Item {item_index}] Prompted response: {prompted_response[:200]}...")

        # ── Phase 5: Judge both responses (via answer_judge_step) ──────────
        # Judge agentic response
        logger.info(f"[Item {item_index}] Judging agentic (binary, type={item['question_type']})...")
        agentic_judgment = await judge_response_via_job(
            app=app,
            question=question,
            ground_truth=item["answer"],
            response=agentic_response,
            question_type=item["question_type"],
        )
        logger.info(f"[Item {item_index}] agentic binary result: {agentic_judgment}")

        # Judge prompted response
        logger.info(f"[Item {item_index}] Judging prompted (binary, type={item['question_type']})...")
        prompted_judgment = await judge_response_via_job(
            app=app,
            question=question,
            ground_truth=item["answer"],
            response=prompted_response,
            question_type=item["question_type"],
        )
        logger.info(f"[Item {item_index}] prompted binary result: {prompted_judgment}")

    finally:
        await app.close()

    return {
        "question_id": item["question_id"],
        "question_type": item["question_type"],
        "question": question,
        "ground_truth": item["answer"],
        "agentic_response": agentic_response,
        "agentic_judgment": agentic_judgment,
        "prompted_response": prompted_response,
        "prompted_judgment": prompted_judgment,
        "prompted_input_tokens": prompted_input_tokens,
        "prompted_output_tokens": prompted_output_tokens,
        "sessions_ingested": len(sorted_sessions),
        "dreams_triggered": len(dream_dates_triggered),
    }


# ---------------------------------------------------------------------------
# Worker: runs a single item in its own process with its own event loop
# ---------------------------------------------------------------------------
def _evaluate_item_worker(task_input: tuple) -> dict:
    """Worker function for multiprocessing. Each process gets its own event loop."""
    item, eval_config, item_index, log_level, reme_log_level, eval_only = task_input
    import asyncio  # pylint: disable=import-outside-toplevel

    _configure_worker(log_level, reme_log_level)

    # Permanently suppress "Task exception was never retrieved" /
    # "Event loop is closed" noise from httpx AsyncClient GC cleanup.
    # These fire AFTER asyncio.run() closes the loop, during Python's
    # garbage collection of httpx connection-pool tasks — harmless.
    logging.getLogger("asyncio").setLevel(logging.CRITICAL)

    return asyncio.run(evaluate_item(item, eval_config, item_index, eval_only=eval_only))


def _indexed_worker(indexed_input: tuple) -> tuple:
    """Module-level wrapper for imap_unordered with index tracking."""
    idx, task_input = indexed_input
    return idx, _evaluate_item_worker(task_input)


def _resolve_num_workers(configured: int) -> int:
    """Resolve num_workers: 0=auto (cpu_count-2, min 1), 1=sequential, >1=parallel."""
    if configured == 0:
        return max(1, (os.cpu_count() or 4) - 2)
    return max(1, configured)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(config_path: str | None = None, log_level: str = "INFO", reme_log_level: str = "INFO", eval_only: bool = False):
    """Run the LongMemEval evaluation pipeline.

    Args:
        config_path: Path to the YAML config file.
        log_level: Log level for the eval runner.
        reme_log_level: Log level for reme internal logs.
        eval_only: If True, skip ingestion and only run query+judge using
            existing workspaces.
    """
    from multiprocessing import Pool  # pylint: disable=import-outside-toplevel

    setup_logging(log_level, reme_log_level)
    eval_config = load_eval_config(config_path)
    dataset_cfg = eval_config["dataset"]

    # Load dataset
    dataset_path = _PROJECT_ROOT / dataset_cfg["path"]
    logger.info(f"Loading dataset from {dataset_path}")
    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)

    start = dataset_cfg.get("start_index", 0)
    num_items = dataset_cfg.get("num_items", 1)
    raw_items = data[start : start + num_items]

    # Load confirmed cases if specified (overrides answers, filters items by question_id)
    confirmed_cases_path = dataset_cfg.get("confirmed_cases")
    if confirmed_cases_path:
        cc_full_path = _PROJECT_ROOT / confirmed_cases_path
        logger.info(f"Loading confirmed cases from {cc_full_path}")
        with open(cc_full_path, encoding="utf-8") as f:
            cc_data = json.load(f)
        cc_map = {item["question_id"]: item for item in cc_data}
        logger.info(f"Loaded {len(cc_map)} confirmed cases")

        # Filter by question_id and override answers with confirmed answers
        items_with_idx = []
        for orig_offset, item in enumerate(raw_items):
            qid = item["question_id"]
            if qid in cc_map:
                item = dict(item)  # shallow copy to avoid mutating original dataset
                item["answer"] = cc_map[qid]["answer"]
                items_with_idx.append((start + orig_offset, item))
        logger.info(
            f"Confirmed cases filter: {len(raw_items)} -> {len(items_with_idx)} items",
        )
    else:
        items_with_idx = [(start + i, item) for i, item in enumerate(raw_items)]

    # Filter by question_type if specified
    question_types = dataset_cfg.get("question_types") or []
    if question_types:
        before_filter = len(items_with_idx)
        items_with_idx = [(idx, item) for idx, item in items_with_idx if item.get("question_type") in question_types]
        logger.info(
            f"Filtered by question_types={question_types}: {before_filter} -> {len(items_with_idx)} items",
        )

    logger.info(f"Evaluating {len(items_with_idx)} item(s) starting from index {start}"
                + (" [eval_only: query+judge only]" if eval_only else ""))

    # Resolve parallelism
    num_workers = _resolve_num_workers(eval_config["evaluation"].get("num_workers", 1))
    logger.info(f"Using {num_workers} worker(s)")

    # Create output directory
    output_dir = _PROJECT_ROOT / eval_config["output"]["dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build task args — include log levels and eval_only flag (use original index for workspace lookup)
    task_args = [(item, eval_config, orig_idx, log_level, reme_log_level, eval_only) for orig_idx, item in items_with_idx]

    # Progress tracking (force print regardless of log level, every 10 minutes)
    total_items = len(task_args)
    completed_count = [0]  # use list for mutability in closure
    start_time = time.time()
    progress_lock = threading.Lock()

    def _print_progress(prefix: str = "PROGRESS"):
        elapsed = time.time() - start_time
        elapsed_min = elapsed / 60
        done = completed_count[0]
        pct = 100.0 * done / total_items if total_items else 0
        eta_str = "N/A"
        if done > 0:
            eta_sec = elapsed / done * (total_items - done)
            eta_str = f"{eta_sec/60:.1f}min"
        print(
            f"[{prefix}] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
            f"{done}/{total_items} ({pct:.1f}%) completed | "
            f"elapsed={elapsed_min:.1f}min | ETA={eta_str}",
            flush=True,
        )

    def _progress_timer():
        """Background thread: print progress every 10 minutes."""
        while not _timer_stop.is_set():
            _timer_stop.wait(600)  # 10 minutes
            if not _timer_stop.is_set():
                with progress_lock:
                    _print_progress()

    _timer_stop = threading.Event()
    timer_thread = threading.Thread(target=_progress_timer, daemon=True)
    timer_thread.start()

    # Run evaluation
    if num_workers == 1:
        # Sequential mode
        results = []
        for task_input in task_args:
            result = _evaluate_item_worker(task_input)
            results.append(result)
            with progress_lock:
                completed_count[0] += 1
    else:
        # Parallel mode — use imap_unordered for progress tracking
        results = [None] * total_items
        indexed_args = list(enumerate(task_args))

        with Pool(processes=num_workers) as pool:
            for idx, result in pool.imap_unordered(_indexed_worker, indexed_args):
                results[idx] = result
                with progress_lock:
                    completed_count[0] += 1

    # Stop progress timer
    _timer_stop.set()
    timer_thread.join(timeout=2)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"results_{dataset_cfg['variant']}_{timestamp}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Results saved to {output_file}")

    # Final progress
    _print_progress("FINAL")

    # Print concise summary: per-item binary results + aggregate stats
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    def _accumulate(results_key, judgment_key):
        correct = 0
        stats: dict = {}  # {question_type: {correct: int, total: int}}
        for r in results:
            qtype = r["question_type"]
            verdict = r.get(judgment_key, {}).get("verdict", "N/A")
            if qtype not in stats:
                stats[qtype] = {"correct": 0, "total": 0}
            stats[qtype]["total"] += 1
            if verdict == "yes":
                correct += 1
                stats[qtype]["correct"] += 1
        return correct, stats

    agentic_correct, agentic_type_stats = _accumulate("agentic_response", "agentic_judgment")
    prompted_correct, prompted_type_stats = _accumulate("prompted_response", "prompted_judgment")

    total = len(results)

    # Per-item verdict rows
    for r in results:
        a_verdict = r.get("agentic_judgment", {}).get("verdict", "N/A")
        p_verdict = r.get("prompted_judgment", {}).get("verdict", "N/A")
        print(f"  [{r['question_id']}] type={r['question_type']}  agentic={a_verdict}  prompted={p_verdict}")

    print("\n" + "-" * 60)
    print(f"  Items: {total}")

    # Agentic stats
    print("\n  ── Agentic (ReAct) ──")
    print(f"  Overall accuracy: {agentic_correct}/{total} ({100*agentic_correct/total:.1f}%)")
    print("  Per-type accuracy:")
    for qtype, stats in sorted(agentic_type_stats.items()):
        acc = 100 * stats["correct"] / stats["total"] if stats["total"] else 0
        print(f"    {qtype}: {stats['correct']}/{stats['total']} ({acc:.1f}%)")

    # Prompted stats
    print("\n  ── Prompted (direct LLM + think) ──")
    print(f"  Overall accuracy: {prompted_correct}/{total} ({100*prompted_correct/total:.1f}%)")
    print("  Per-type accuracy:")
    for qtype, stats in sorted(prompted_type_stats.items()):
        acc = 100 * stats["correct"] / stats["total"] if stats["total"] else 0
        print(f"    {qtype}: {stats['correct']}/{stats['total']} ({acc:.1f}%)")

    # Prompted token usage stats
    total_input_tokens = sum(r.get("prompted_input_tokens", 0) for r in results)
    total_output_tokens = sum(r.get("prompted_output_tokens", 0) for r in results)
    counted = sum(1 for r in results if r.get("prompted_input_tokens", 0) > 0)
    avg_input = total_input_tokens / counted if counted else 0
    avg_output = total_output_tokens / counted if counted else 0
    print(f"\n  ── Prompted Token Usage (avg over {counted} queries) ──")
    print(f"  Avg input tokens/query:  {avg_input:,.1f}")
    print(f"  Avg output tokens/query: {avg_output:,.1f}")
    print(f"  Total input tokens:      {total_input_tokens:,}")
    print(f"  Total output tokens:     {total_output_tokens:,}")

    print("=" * 60)
    total_elapsed = time.time() - start_time
    print(f"\n  Total time: {total_elapsed/60:.1f} min")
    print("\n" + "=" * 60)
    print("  [DONE] EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LongMemEval evaluation runner")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level for the eval runner (default: INFO)",
    )
    parser.add_argument(
        "--reme-log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level for reme internal logs — loguru (default: INFO)",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Shortcut for --log-level WARNING --reme-log-level WARNING",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Skip ingestion (phases 1-3). Reuse existing workspaces and only run query+judge.",
    )
    args = parser.parse_args()

    if args.quiet:
        args.log_level = "WARNING"
        args.reme_log_level = "WARNING"

    main(args.config, args.log_level, args.reme_log_level, eval_only=args.eval_only)
