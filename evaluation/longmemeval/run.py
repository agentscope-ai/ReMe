"""LongMemEval evaluation runner for ReMe.

Evaluates ReMe's long-term memory capability using the LongMemEval dataset.
Each item gets an isolated workspace; sessions are ingested in chronological order;
dream is triggered when sessions cross midnight (23:00); finally questions are
answered via search and judged by an LLM.

Usage:
    python evaluation/longmemeval/run.py
    python evaluation/longmemeval/run.py --config evaluation/longmemeval/config.yaml
"""

import json
import logging
import os
import re
import shutil
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Load .env from project root
_PROJECT_ROOT = Path(__file__).parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# Workspace root for evaluation items
_WORKSPACE_ROOT = _PROJECT_ROOT / "memory_workspaces"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("longmemeval")


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
# LLM-as-Judge
# ---------------------------------------------------------------------------
BINARY_JUDGE_PROMPT = """\
You are an evaluation judge. Given a question, the ground-truth answer, \
and a system's response, determine if the system's response correctly answers the question.

Question: {question}
Ground-truth answer: {answer}
System response: {response}

Does the system's response correctly answer the question? Consider semantic equivalence, not exact wording.
Reply with ONLY a JSON object: {{"verdict": "yes" or "no", "reason": "brief explanation"}}"""

SCORE_JUDGE_PROMPT = """\
You are an evaluation judge. Given a question, the ground-truth answer, \
and a system's response, rate the quality of the system's response on a scale of 0-5.

Scoring rubric:
- 0: Completely wrong or irrelevant
- 1: Mostly wrong with minor relevant elements
- 2: Partially correct but missing key information
- 3: Mostly correct but with notable omissions or inaccuracies
- 4: Correct with minor issues
- 5: Perfectly correct and complete

Question: {question}
Ground-truth answer: {answer}
System response: {response}

Reply with ONLY a JSON object: {{"score": <0-5>, "reason": "brief explanation"}}"""


async def judge_response(
    question: str,
    ground_truth: str,
    response: str,
    metric: str,
    judge_llm,
) -> dict:
    """Call LLM judge to evaluate a response using reme's as_llm component."""
    from agentscope.message import Msg

    if metric == "binary":
        prompt = BINARY_JUDGE_PROMPT.format(question=question, answer=ground_truth, response=response)
    elif metric == "score":
        prompt = SCORE_JUDGE_PROMPT.format(question=question, answer=ground_truth, response=response)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    user_msg = Msg(name="user", role="user", content=[{"type": "text", "text": prompt}])
    chat_response = await judge_llm.model([user_msg])
    # Extract text from response content blocks
    raw_text = ""
    for block in chat_response.content:
        if hasattr(block, "text"):
            raw_text += block.text
    raw_text = raw_text.strip()

    # Extract JSON from response
    try:
        json_match = re.search(r"\{[^}]+\}", raw_text)
        if json_match:
            result = json.loads(json_match.group())
        else:
            result = json.loads(raw_text)
    except json.JSONDecodeError:
        result = {"raw": raw_text, "error": "failed to parse judge response"}

    result["metric"] = metric
    return result


# ---------------------------------------------------------------------------
# Main evaluation pipeline
# ---------------------------------------------------------------------------
async def evaluate_item(item: dict, eval_config: dict, item_index: int) -> dict:
    """Evaluate a single LongMemEval item end-to-end."""
    from reme import Application
    from reme.config import resolve_app_config
    from reme.enumeration import ComponentEnum

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
        f"type={item['question_type']} sessions={len(sorted_sessions)}",
    )

    # Use fixed workspace directory (clean it for fresh evaluation)
    item_dir = _WORKSPACE_ROOT / f"item_{item_index}"
    workspace_dir = str(item_dir / ".reme")
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
        # ── Phase 1: Ingest sessions ──────────────────────────────────
        prev_dt = None
        dream_dates_triggered = set()

        for idx, (_, session_dt, session_id, messages) in enumerate(sorted_sessions):
            # Check if dream should be triggered before this session
            if prev_dt is not None and should_trigger_dream(prev_dt, session_dt, dream_trigger_hour):
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

        # ── Phase 2: Final dream for the last day ─────────────────────
        if prev_dt is not None:
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
                    logger.warning(f"[Item {item_index}] Final dream failed: {e}")
                dream_dates_triggered.add(last_dream_date)
                # Index update after final dream
                await app.run_job("index_update")

        # ── Phase 3: Digest update ────────────────────────────────────
        await app.run_job("digest_update")

        # ── Phase 4: Ask question via bench_query_job (ReAct agent) ──
        question = item["question"]
        question_date_raw = item.get("question_date", "")
        question_dt = parse_haystack_date(question_date_raw) if question_date_raw else None
        query_time = to_iso(question_dt) if question_dt else ""
        logger.info(
            f"[Item {item_index}] Asking: {question[:80]}... query_time={query_time}",
        )

        query_resp = await app.run_job(
            "bench_query_job",
            query=question,
            query_time=query_time,
        )
        system_response = (query_resp.answer or "").strip()
        if not system_response:
            system_response = "(no answer generated)"

        logger.info(f"[Item {item_index}] System response: {system_response[:200]}...")

        # ── Phase 5: Judge (using reme's as_llm component) ────────────
        judge_llm = app.context.components[ComponentEnum.AS_LLM]["judge"]
        judgments = {}
        for metric in eval_config["evaluation"].get("metrics", ["binary", "score"]):
            logger.info(f"[Item {item_index}] Judging with metric={metric}...")
            judgment = await judge_response(
                question=question,
                ground_truth=item["answer"],
                response=system_response,
                metric=metric,
                judge_llm=judge_llm,
            )
            judgments[metric] = judgment
            logger.info(f"[Item {item_index}] {metric} result: {judgment}")

    finally:
        await app.close()

    return {
        "question_id": item["question_id"],
        "question_type": item["question_type"],
        "question": question,
        "ground_truth": item["answer"],
        "system_response": system_response,
        "judgments": judgments,
        "sessions_ingested": len(sorted_sessions),
        "dreams_triggered": len(dream_dates_triggered),
    }


# ---------------------------------------------------------------------------
# Worker: runs a single item in its own process with its own event loop
# ---------------------------------------------------------------------------
def _evaluate_item_worker(task_input: tuple) -> dict:
    """Worker function for multiprocessing. Each process gets its own event loop."""
    item, eval_config, item_index = task_input
    import asyncio  # pylint: disable=import-outside-toplevel

    return asyncio.run(evaluate_item(item, eval_config, item_index))


def _resolve_num_workers(configured: int) -> int:
    """Resolve num_workers: 0=auto (cpu_count-2, min 1), 1=sequential, >1=parallel."""
    if configured == 0:
        return max(1, (os.cpu_count() or 4) - 2)
    return max(1, configured)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(config_path: str | None = None):
    """Run the LongMemEval evaluation pipeline."""
    from multiprocessing import Pool  # pylint: disable=import-outside-toplevel

    eval_config = load_eval_config(config_path)
    dataset_cfg = eval_config["dataset"]

    # Load dataset
    dataset_path = _PROJECT_ROOT / dataset_cfg["path"]
    logger.info(f"Loading dataset from {dataset_path}")
    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)

    start = dataset_cfg.get("start_index", 0)
    num_items = dataset_cfg.get("num_items", 1)
    items_to_eval = data[start : start + num_items]

    # Filter by question_type if specified
    question_types = dataset_cfg.get("question_types") or []
    if question_types:
        before_filter = len(items_to_eval)
        items_to_eval = [item for item in items_to_eval if item.get("question_type") in question_types]
        logger.info(
            f"Filtered by question_types={question_types}: {before_filter} -> {len(items_to_eval)} items",
        )

    logger.info(f"Evaluating {len(items_to_eval)} item(s) starting from index {start}")

    # Resolve parallelism
    num_workers = _resolve_num_workers(eval_config["evaluation"].get("num_workers", 1))
    logger.info(f"Using {num_workers} worker(s)")

    # Create output directory
    output_dir = _PROJECT_ROOT / eval_config["output"]["dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build task args
    task_args = [(item, eval_config, start + i) for i, item in enumerate(items_to_eval)]

    # Run evaluation
    if num_workers == 1:
        # Sequential mode
        results = []
        for task_input in task_args:
            result = _evaluate_item_worker(task_input)
            results.append(result)
    else:
        # Parallel mode
        with Pool(processes=num_workers) as pool:
            results = pool.map(_evaluate_item_worker, task_args)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"results_{dataset_cfg['variant']}_{timestamp}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Results saved to {output_file}")

    # Print concise summary: per-item scores + aggregate stats
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    scores = []
    binary_correct = 0
    for r in results:
        score = r.get("judgments", {}).get("score", {}).get("score", "N/A")
        binary = r.get("judgments", {}).get("binary", {}).get("verdict", "N/A")
        print(f"  [{r['question_id']}] type={r['question_type']}  binary={binary}  score={score}/5")
        if isinstance(score, (int, float)):
            scores.append(score)
        if binary == "yes":
            binary_correct += 1

    print("\n" + "-" * 60)
    total = len(results)
    if scores:
        avg_score = sum(scores) / len(scores)
        print(f"  Items: {total}")
        print(f"  Binary accuracy: {binary_correct}/{total} ({100*binary_correct/total:.1f}%)")
        print(f"  Avg score: {avg_score:.2f}/5")
    else:
        print(f"  Items: {total} (no valid scores)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LongMemEval evaluation runner")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    args = parser.parse_args()
    main(args.config)
