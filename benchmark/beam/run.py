"""BEAM evaluation runner for ReMe.

Evaluates ReMe's memory capability using the BEAM dataset.
Each case gets an isolated workspace; chat.json batches are ingested as
sessions in chronological order; finally probing questions are answered
via both prompted (search + LLM) and agentic (ReAct) approaches, then
judged by BEAM's rubric-based LLM-as-judge.

Usage:
    python benchmark/beam/run.py
    python benchmark/beam/run.py --config benchmark/beam/config.yaml
    python benchmark/beam/run.py -q                          # quiet: only eval-level logs
    python benchmark/beam/run.py --log-level WARNING         # reduce eval runner logs
    python benchmark/beam/run.py --reme-log-level WARNING    # reduce reme internal logs
    python benchmark/beam/run.py --eval_only                 # query+judge only, reuse existing workspace
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

# Workspace root — read from config.yaml (dataset.workspace_root)
_WORKSPACE_ROOT_DEFAULT = "memory_workspaces/beam"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"

logging.basicConfig(level=logging.INFO, format=_DEFAULT_LOG_FORMAT)
logger = logging.getLogger("beam")

# Noisy library loggers silenced by default
_NOISY_LOGGERS = [
    "httpx",
    "httpcore",
    "openai",
    "uvicorn",
    "multipart",
    "asyncio",
    "watchfiles",
    "filelock",
]


def setup_logging(
    log_level: str,
    reme_log_level: str,
    log_dir: str | None = None,
):
    """Configure logging for the eval runner and reme internals.

    Args:
        log_level: Level for the eval runner logger (DEBUG/INFO/WARNING/ERROR).
        reme_log_level: Level for reme's internal loguru logger.
        log_dir: Per-run log directory (absolute path). None = no file logging.
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

    # Add file handler for eval runner if log_dir is specified
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_filepath = os.path.join(log_dir, "runner.log")
        file_handler = logging.FileHandler(log_filepath, encoding="utf-8")
        file_handler.setLevel(numeric)
        file_handler.setFormatter(logging.Formatter(_DEFAULT_LOG_FORMAT))
        logging.getLogger().addHandler(file_handler)
        logger.info(f"Eval runner log file: {log_filepath}")

    # Reme internal logger (loguru) — will be applied per-worker via _configure_worker
    os.environ["REME_LOG_LEVEL"] = reme_log_level.upper()
    if log_dir:
        os.environ["REME_LOG_DIR"] = log_dir


def _configure_worker(
    log_level: str,
    reme_log_level: str,
    log_dir: str | None = None,
):
    """Set up logging inside a multiprocessing worker process.

    Must be called at the top of each worker because child processes inherit
    parent state but loguru sinks are NOT shared across fork/spawn.
    """
    numeric = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(level=numeric, format=_DEFAULT_LOG_FORMAT, force=True)
    logging.getLogger("beam").setLevel(numeric)
    if numeric > logging.DEBUG:
        for name in _NOISY_LOGGERS:
            logging.getLogger(name).setLevel(max(numeric, logging.WARNING))

    # Add file handler for eval runner in worker process
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        pid = os.getpid()
        log_filepath = os.path.join(log_dir, f"worker-{pid}.log")
        file_handler = logging.FileHandler(log_filepath, encoding="utf-8")
        file_handler.setLevel(numeric)
        file_handler.setFormatter(logging.Formatter(_DEFAULT_LOG_FORMAT))
        logging.getLogger().addHandler(file_handler)

    # Re-initialize loguru for reme internals at the desired level
    from reme.utils import get_logger

    reme_log_dir = log_dir or "logs"
    get_logger(log_dir=reme_log_dir, level=reme_log_level.upper(), force_init=True)


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
# BEAM data loading
# ---------------------------------------------------------------------------
def parse_beam_time_anchor(time_str: str) -> datetime:
    """Parse BEAM time_anchor format: 'March-15-2024' -> datetime."""
    for fmt in ("%B-%d-%Y", "%b-%d-%Y"):
        try:
            return datetime.strptime(time_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse time_anchor: {time_str!r}")


def load_beam_chat(chat_path: Path, chat_size: str, case_id: str) -> list[dict]:
    """Load BEAM chat.json and convert to ReMe session format.

    Each batch becomes one session with all its turns flattened.
    Each turn resolves its own time_anchor independently; turns without
    an explicit time_anchor inherit from the most recent preceding turn.
    Returns list of sessions, each with:
      - session_id: str
      - date: str (YYYY-MM-DD)  — derived from the *first* turn's time
      - messages: list[dict] with name, role, content, created_at
    """
    with open(chat_path, encoding="utf-8") as f:
        batches = json.load(f)

    sessions = []
    for batch in batches:
        batch_num = batch["batch_number"]

        # Resolve batch-level fallback (used when no turn has a time_anchor)
        batch_anchor = batch.get("time_anchor")
        if not batch_anchor:
            batch_anchor = "January-1-2024"

        # Flatten all turns, resolving time_anchor per turn
        messages = []
        prev_dt = None  # carries forward from previous turn
        first_dt = None  # for session-level date

        for turn in batch["turns"]:
            # Find this turn's own time_anchor from its messages
            turn_anchor = None
            for msg in turn:
                if msg.get("time_anchor"):
                    turn_anchor = msg["time_anchor"]
                    break

            if turn_anchor:
                dt = parse_beam_time_anchor(turn_anchor)
            elif prev_dt is not None:
                dt = prev_dt  # inherit from previous turn
            else:
                dt = parse_beam_time_anchor(batch_anchor)

            if first_dt is None:
                first_dt = dt
            prev_dt = dt

            for msg in turn:
                role = msg["role"]
                messages.append(
                    {
                        "name": role,
                        "role": role,
                        "content": msg["content"],
                        "created_at": dt.strftime("%Y-%m-%dT%H:%M:%S"),
                    },
                )

        sessions.append(
            {
                "session_id": f"beam_{chat_size}_{case_id}_batch{batch_num}",
                "date": first_dt.strftime("%Y-%m-%d"),
                "messages": messages,
            },
        )

    return sessions


def get_available_cases(beam_root: Path, chat_size: str) -> list[str]:
    """Return sorted list of case IDs for a given chat size."""
    chats_dir = beam_root / "chats" / chat_size
    if not chats_dir.exists():
        return []
    return sorted(
        [d.name for d in chats_dir.iterdir() if d.is_dir()],
        key=int,
    )


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------
async def answer_question_prompted(app, question: str) -> tuple[str, dict]:
    """Answer a probing question using ReMe search + context_answer job.

    Returns (answer, metadata)
    """
    # Search ReMe memory
    search_resp = await app.run_job("search", query=question, limit=15)
    search_context = (search_resp.answer or "").strip()
    search_hit_count = (search_resp.metadata or {}).get("counts", {}).get("returned", 0)
    logger.info(f"  Search: {search_hit_count} hits")

    if not search_context:
        search_context = "(no search results found)"

    # Generate answer via context_answer job (BEAM original prompt)
    answer_resp = await app.run_job(
        "context_answer",
        retrieved_context=search_context,
        question=question,
    )
    answer = (answer_resp.answer or "").strip()

    return answer, {
        "search_hits": search_hit_count,
        "search_context_preview": search_context[:300],
    }


async def answer_question_agentic(app, question: str) -> tuple[str, dict]:
    """Answer a probing question using ReMe's agentic_answer job.

    Returns (answer, metadata)
    """
    query_resp = await app.run_job(
        "agentic_answer",
        query=question,
    )
    answer = (query_resp.answer or "").strip()

    return answer, {"mode": "agentic"}


# ---------------------------------------------------------------------------
# BEAM rubric-based LLM-as-Judge
# ---------------------------------------------------------------------------
async def judge_answer(
    app,
    question: str,
    llm_response: str,
    rubric: list[str],
    question_type: str = "",
) -> dict:
    """Judge an answer via the answer_judge job (beam_rubric_judge_step)."""
    judge_resp = await app.run_job(
        "answer_judge",
        llm_response=llm_response,
        rubric=rubric,
        probing_question=question,
        question_type=question_type,
    )
    result = {
        "llm_judge_score": (judge_resp.metadata or {}).get("llm_judge_score", 0.0),
        "llm_judge_responses": (judge_resp.metadata or {}).get("llm_judge_responses", []),
    }
    # Include event_ordering extra metrics if present
    eo = (judge_resp.metadata or {}).get("event_ordering")
    if eo:
        result["event_ordering"] = eo
    return result


# ---------------------------------------------------------------------------
# Main evaluation pipeline
# ---------------------------------------------------------------------------
async def evaluate_case(eval_config: dict, case_id: str, eval_only: bool = False) -> dict:
    """Evaluate a single BEAM case end-to-end.

    Args:
        eval_config: The evaluation configuration dict.
        case_id: The case directory name (e.g. "1").
        eval_only: If True, skip ingestion and only run query+judge
            using the existing workspace.

    Returns:
        A results dict with all questions, answers, and judgments.
    """
    from reme import Application
    from reme.config import resolve_app_config

    dataset_cfg = eval_config["dataset"]
    chat_size = dataset_cfg["chat_size"]
    beam_root = _PROJECT_ROOT / dataset_cfg.get("beam_root", "datasets/BEAM")
    modes = eval_config["evaluation"].get("modes", ["prompted", "agentic"])

    chat_path = beam_root / "chats" / chat_size / case_id / "chat.json"
    probing_questions_path = beam_root / "chats" / chat_size / case_id / "probing_questions" / "probing_questions.json"

    if not chat_path.exists():
        raise FileNotFoundError(f"Chat file not found: {chat_path}")
    if not probing_questions_path.exists():
        raise FileNotFoundError(f"Probing questions not found: {probing_questions_path}")

    logger.info(
        "[Case %s] size=%s modes=%s%s",
        case_id,
        chat_size,
        modes,
        " [eval_only]" if eval_only else "",
    )

    # Workspace setup
    workspace_root = _PROJECT_ROOT / dataset_cfg.get("workspace_root", _WORKSPACE_ROOT_DEFAULT)
    case_dir = workspace_root / f"{chat_size}_{case_id}"
    workspace_dir = str(case_dir / ".reme")

    if eval_only:
        if not case_dir.exists() or not Path(workspace_dir).exists():
            logger.warning(
                f"[Case {case_id}] eval_only: workspace not found at {case_dir}, skipping",
            )
            return {
                "case_id": case_id,
                "chat_size": chat_size,
                "error": "workspace missing in eval_only mode",
                "questions": [],
            }
    else:
        if case_dir.exists():
            shutil.rmtree(case_dir)
            logger.info(f"[Case {case_id}] Cleaned existing workspace: {case_dir}")
        case_dir.mkdir(parents=True, exist_ok=True)

    # Pre-initialize ReMe's loguru logger with the correct log_dir
    output_cfg = eval_config.get("output", {})
    if output_cfg.get("log_to_file", False):
        reme_log_dir = os.environ.get("REME_LOG_DIR")
        if reme_log_dir:
            from reme.utils import get_logger

            get_logger(
                log_dir=reme_log_dir,
                level=os.environ.get("REME_LOG_LEVEL", "INFO"),
                log_to_console=output_cfg.get("log_to_console", True),
                log_to_file=True,
                force_init=True,
            )

    cfg = resolve_app_config(
        config=eval_config["reme"]["config"],
        workspace_dir=workspace_dir,
        log_to_console=output_cfg.get("log_to_console", True),
        log_to_file=output_cfg.get("log_to_file", False),
        enable_logo=False,
    )

    app = Application(**cfg)
    await app.start()

    sessions_ingested = 0
    try:
        if not eval_only:
            # ── Phase 1: Ingest sessions ──────────────────────────────
            sessions = load_beam_chat(chat_path, chat_size, case_id)
            logger.info(f"[Case {case_id}] Loaded {len(sessions)} sessions from chat.json")

            for i, session in enumerate(sessions):
                logger.info(
                    f"[Case {case_id}] Ingesting session {i+1}/{len(sessions)}: "
                    f"id={session['session_id']} date={session['date']} "
                    f"msgs={len(session['messages'])}",
                )
                resp = await app.run_job(
                    "auto_memory",
                    messages=session["messages"],
                    session_id=session["session_id"],
                    date=session["date"],
                )
                if not resp.success:
                    logger.warning(f"[Case {case_id}] auto_memory failed: {resp.answer}")
                else:
                    logger.info(
                        f"[Case {case_id}] auto_memory success: " f"{resp.answer[:100] if resp.answer else ''}",
                    )
                await app.run_job("index_update")
                sessions_ingested += 1

            # Final digest update
            logger.info(f"[Case {case_id}] Running digest_update...")
            await app.run_job("digest_update")
            logger.info(f"[Case {case_id}] Ingestion complete.")

        # ── Phase 2: Answer + Judge probing questions ───────────────
        with open(probing_questions_path, encoding="utf-8") as f:
            probing_questions = json.load(f)

        total_questions = sum(len(v) for v in probing_questions.values())
        logger.info(f"[Case {case_id}] Total probing questions: {total_questions}")

        all_question_results = []
        q_idx = 0

        for q_type in probing_questions:
            logger.info(
                f"[Case {case_id}] Question type: {q_type} " f"({len(probing_questions[q_type])} questions)",
            )

            for i, q in enumerate(probing_questions[q_type]):
                q_idx += 1
                question = q["question"]
                rubric = q.get("rubric", [])
                logger.info(
                    f"[Case {case_id}] [{q_idx}/{total_questions}] " f"{q_type} Q{i+1}: {question[:100]}...",
                )

                q_result = {
                    "question_type": q_type,
                    "question_index": i,
                    "question": question,
                    "rubric": rubric,
                }

                # Prompted answer
                if "prompted" in modes:
                    try:
                        prompted_answer, prompted_meta = await answer_question_prompted(
                            app,
                            question,
                        )
                    except Exception as e:
                        logger.error(f"[Case {case_id}] Prompted answer failed: {e}")
                        prompted_answer = f"(error: {e})"
                        prompted_meta = {"error": str(e)}

                    if not prompted_answer:
                        prompted_answer = "(no answer generated)"
                    logger.info(f"[Case {case_id}] Prompted answer: {prompted_answer[:200]}...")

                    # Judge prompted answer
                    logger.info(f"[Case {case_id}] Judging prompted ({q_type})...")
                    prompted_judgment = await judge_answer(
                        app,
                        question,
                        prompted_answer,
                        rubric,
                        question_type=q_type,
                    )
                    logger.info(
                        f"[Case {case_id}] Prompted score: " f"{prompted_judgment['llm_judge_score']:.3f}",
                    )

                    q_result["prompted_response"] = prompted_answer
                    q_result["prompted_judgment"] = prompted_judgment
                    q_result["prompted_metadata"] = prompted_meta

                # Agentic answer
                if "agentic" in modes:
                    try:
                        agentic_answer, agentic_meta = await answer_question_agentic(
                            app,
                            question,
                        )
                    except Exception as e:
                        logger.error(f"[Case {case_id}] Agentic answer failed: {e}")
                        agentic_answer = f"(error: {e})"
                        agentic_meta = {"error": str(e)}

                    if not agentic_answer:
                        agentic_answer = "(no answer generated)"
                    logger.info(f"[Case {case_id}] Agentic answer: {agentic_answer[:200]}...")

                    # Judge agentic answer
                    logger.info(f"[Case {case_id}] Judging agentic ({q_type})...")
                    agentic_judgment = await judge_answer(
                        app,
                        question,
                        agentic_answer,
                        rubric,
                        question_type=q_type,
                    )
                    logger.info(
                        f"[Case {case_id}] Agentic score: " f"{agentic_judgment['llm_judge_score']:.3f}",
                    )

                    q_result["agentic_response"] = agentic_answer
                    q_result["agentic_judgment"] = agentic_judgment
                    q_result["agentic_metadata"] = agentic_meta

                all_question_results.append(q_result)

    finally:
        await app.close()

    return {
        "case_id": case_id,
        "chat_size": chat_size,
        "sessions_ingested": sessions_ingested,
        "total_questions": len(all_question_results),
        "questions": all_question_results,
    }


# ---------------------------------------------------------------------------
# Worker: runs a single case in its own process with its own event loop
# ---------------------------------------------------------------------------
def _evaluate_case_worker(task_input: tuple) -> dict:
    """Worker function for multiprocessing. Each process gets its own event loop."""
    eval_config, case_id, log_level, reme_log_level, eval_only, log_dir = task_input
    import asyncio  # pylint: disable=import-outside-toplevel

    _configure_worker(log_level, reme_log_level, log_dir=log_dir)

    # Suppress httpx GC noise
    logging.getLogger("asyncio").setLevel(logging.CRITICAL)

    return asyncio.run(evaluate_case(eval_config, case_id, eval_only=eval_only))


def _indexed_worker(indexed_input: tuple) -> tuple:
    """Module-level wrapper for imap_unordered with index tracking."""
    idx, task_input = indexed_input
    return idx, _evaluate_case_worker(task_input)


def _resolve_num_workers(configured: int) -> int:
    """Resolve num_workers: 0=auto (cpu_count-2, min 1), 1=sequential, >1=parallel."""
    if configured == 0:
        return max(1, (os.cpu_count() or 4) - 2)
    return max(1, configured)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(  # pylint: disable=too-many-statements
    config_path: str | None = None,
    log_level: str = "INFO",
    reme_log_level: str = "INFO",
    eval_only: bool = False,
):
    """Run the BEAM evaluation pipeline.

    Args:
        config_path: Path to the YAML config file.
        log_level: Log level for the eval runner.
        reme_log_level: Log level for reme internal logs.
        eval_only: If True, skip ingestion and only run query+judge using
            existing workspaces.
    """
    from multiprocessing import Pool  # pylint: disable=import-outside-toplevel

    # Load config BEFORE logging setup so log_dir is available
    eval_config = load_eval_config(config_path)

    # Resolve per-run log directory from config
    output_cfg = eval_config.get("output", {})
    log_dir_abs = None
    if output_cfg.get("log_to_file", False):
        log_dir_raw = output_cfg.get("log_dir", "logs")
        log_prefix = output_cfg.get("log_prefix", "beam")
        run_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir_abs = str(_PROJECT_ROOT / log_dir_raw / f"{log_prefix}_{run_ts}")

    setup_logging(log_level, reme_log_level, log_dir=log_dir_abs)
    dataset_cfg = eval_config["dataset"]
    chat_size = dataset_cfg["chat_size"]
    beam_root = _PROJECT_ROOT / dataset_cfg.get("beam_root", "datasets/BEAM")

    # Determine which cases to run
    case_ids = dataset_cfg.get("case_ids") or []
    if not case_ids:
        case_ids = get_available_cases(beam_root, chat_size)

    # Pagination
    start = dataset_cfg.get("start_index", 0)
    num_items = dataset_cfg.get("num_items", 0)
    if num_items > 0:
        case_ids = case_ids[start : start + num_items]
    elif start > 0:
        case_ids = case_ids[start:]

    if not case_ids:
        logger.error(f"No cases found for chat_size={chat_size}")
        return

    logger.info(
        "Evaluating %d case(s) for chat_size=%s: %s%s",
        len(case_ids),
        chat_size,
        case_ids,
        " [eval_only: query+judge only]" if eval_only else "",
    )

    # Resolve parallelism
    num_workers = _resolve_num_workers(eval_config["evaluation"].get("num_workers", 1))
    logger.info(f"Using {num_workers} worker(s)")

    # Create output directory
    output_dir = _PROJECT_ROOT / eval_config["output"]["dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build task args
    task_args = [(eval_config, case_id, log_level, reme_log_level, eval_only, log_dir_abs) for case_id in case_ids]

    # Progress tracking
    total_items = len(task_args)
    completed_count = [0]
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
            _timer_stop.wait(600)
            if not _timer_stop.is_set():
                with progress_lock:
                    _print_progress()

    _timer_stop = threading.Event()
    timer_thread = threading.Thread(target=_progress_timer, daemon=True)
    timer_thread.start()

    # Run evaluation
    if num_workers == 1:
        results = []
        for task_input in task_args:
            result = _evaluate_case_worker(task_input)
            results.append(result)
            with progress_lock:
                completed_count[0] += 1
    else:
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
    output_file = output_dir / f"results_{chat_size}_{timestamp}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Results saved to {output_file}")

    # Final progress
    _print_progress("FINAL")

    # Print concise summary
    print("\n" + "=" * 70)
    print(f"  BEAM EVALUATION RESULTS  |  size={chat_size}  cases={len(results)}")
    print("=" * 70)

    modes = eval_config["evaluation"].get("modes", ["prompted", "agentic"])

    for mode in modes:
        mode_key = f"{mode}_judgment"

        # Per-type stats
        type_scores: dict[str, list[float]] = {}
        type_binary_scores: dict[str, list[float]] = {}
        all_scores: list[float] = []
        all_binary_scores: list[float] = []

        for case_result in results:
            if "error" in case_result:
                continue
            for q in case_result.get("questions", []):
                judgment = q.get(mode_key, {})
                score = judgment.get("llm_judge_score", 0.0)
                # Binary: convert each rubric item score to 0/1, then average
                judge_responses = judgment.get("llm_judge_responses", [])
                if judge_responses:
                    binary_scores_per_item = [1.0 if r.get("score", 0) >= 1.0 else 0.0 for r in judge_responses]
                    binary_score = sum(binary_scores_per_item) / len(binary_scores_per_item)
                else:
                    binary_score = 1.0 if score > 0.99 else 0.0
                qtype = q["question_type"]
                if qtype not in type_scores:
                    type_scores[qtype] = []
                    type_binary_scores[qtype] = []
                type_scores[qtype].append(score)
                type_binary_scores[qtype].append(binary_score)
                all_scores.append(score)
                all_binary_scores.append(binary_score)

        print(f"\n  ── {mode.upper()} ──")
        if all_scores:
            for qtype in sorted(type_scores.keys()):
                scores = type_scores[qtype]
                avg = sum(scores) / len(scores) if scores else 0
                bin_scores = type_binary_scores[qtype]
                bin_avg = sum(bin_scores) / len(bin_scores) if bin_scores else 0
                print(f"    {qtype:<40s}: {avg:.3f}  binary={bin_avg:.3f}  ({len(scores)} Qs)")
            overall = sum(all_scores) / len(all_scores) if all_scores else 0
            binary_overall = sum(all_binary_scores) / len(all_binary_scores) if all_binary_scores else 0
            print(f"    {'-'*38}")
            print(f"    {'OVERALL':<40s}: {overall:.3f}  binary={binary_overall:.3f}  ({len(all_scores)} Qs)")
        else:
            print("    (no results)")

    # Per-case summary
    print("\n  ── Per-Case Summary ──")
    for case_result in results:
        case_id = case_result["case_id"]
        if "error" in case_result:
            print(f"    Case {case_id}: ERROR — {case_result['error']}")
            continue
        n_qs = case_result.get("total_questions", 0)
        n_sessions = case_result.get("sessions_ingested", 0)
        parts = [f"Case {case_id}: {n_sessions} sessions, {n_qs} questions"]
        for mode in modes:
            mode_key = f"{mode}_judgment"
            questions = case_result.get("questions", [])
            scores = [q.get(mode_key, {}).get("llm_judge_score", 0.0) for q in questions]
            if scores:
                avg = sum(scores) / len(scores)
                # Binary: 0/1 per rubric item, average per question, then across questions
                bin_scores = []
                for q in questions:
                    judge_responses = q.get(mode_key, {}).get("llm_judge_responses", [])
                    if judge_responses:
                        item_bins = [1.0 if r.get("score", 0) >= 1.0 else 0.0 for r in judge_responses]
                        bin_scores.append(sum(item_bins) / len(item_bins))
                    else:
                        s = q.get(mode_key, {}).get("llm_judge_score", 0.0)
                        bin_scores.append(1.0 if s > 0.99 else 0.0)
                bin_avg = sum(bin_scores) / len(bin_scores)
                parts.append(f"{mode}={avg:.3f} binary={bin_avg:.3f}")
        print(f"    {' | '.join(parts)}")

    print("=" * 70)
    total_elapsed = time.time() - start_time
    print(f"\n  Total time: {total_elapsed/60:.1f} min")
    print("\n" + "=" * 70)
    print("  [DONE] BEAM EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BEAM evaluation runner")
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
        "-q",
        "--quiet",
        action="store_true",
        help="Shortcut for --log-level WARNING --reme-log-level WARNING",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Skip ingestion. Reuse existing workspaces and only run query+judge.",
    )
    args = parser.parse_args()

    if args.quiet:
        args.log_level = "WARNING"
        args.reme_log_level = "WARNING"

    main(args.config, args.log_level, args.reme_log_level, eval_only=args.eval_only)
