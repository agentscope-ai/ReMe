"""LongMemEval evaluation runner for ReMe.

Evaluates ReMe's long-term memory capability using the LongMemEval dataset.
Each item gets an isolated workspace; sessions are ingested in chronological order;
dream is triggered when sessions cross midnight (23:00); finally questions are
answered via search and judged by an LLM.

Usage:
    python evaluation/longmemeval/run.py
    python evaluation/longmemeval/run.py --config evaluation/longmemeval/config.yaml
"""

import asyncio
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


def should_trigger_dream(prev_dt: datetime, curr_dt: datetime, trigger_hour: int = 23) -> bool:
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
        zip(item["haystack_dates"], item["haystack_session_ids"], item["haystack_sessions"])
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
        formatted.append({
            "name": role,
            "role": role,
            "content": msg["content"],
            "created_at": to_iso(session_dt),
        })
    return formatted



# ---------------------------------------------------------------------------
# LLM-as-Judge
# ---------------------------------------------------------------------------
ANSWER_PROMPT = """You are a helpful assistant with access to a memory system. Based on the retrieved memory context below, answer the user's question concisely and accurately.

Retrieved context:
{context}

Question: {question}

Answer the question based ONLY on the retrieved context. If the context does not contain enough information, say so."""

BINARY_JUDGE_PROMPT = """You are an evaluation judge. Given a question, the ground-truth answer, and a system's response, determine if the system's response correctly answers the question.

Question: {question}
Ground-truth answer: {answer}
System response: {response}

Does the system's response correctly answer the question? Consider semantic equivalence, not exact wording.
Reply with ONLY a JSON object: {{"verdict": "yes" or "no", "reason": "brief explanation"}}"""

SCORE_JUDGE_PROMPT = """You are an evaluation judge. Given a question, the ground-truth answer, and a system's response, rate the quality of the system's response on a scale of 0-5.

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
    from agentscope.message import Msg

    reme_cfg = eval_config["reme"]
    dream_trigger_hour = reme_cfg.get("dream_trigger_hour", 23)
    dream_scan_days = reme_cfg.get("dream_scan_days", 2)
    dream_max_units = reme_cfg.get("dream_max_units", 5)

    # Sort sessions by time
    sorted_sessions = sessions_sorted_by_time(item)
    logger.info(
        f"[Item {item_index}] question_id={item['question_id']} "
        f"type={item['question_type']} sessions={len(sorted_sessions)}"
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

        for idx, (orig_idx, session_dt, session_id, messages) in enumerate(sorted_sessions):
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
                            f"answer={dream_resp.answer[:100] if dream_resp.answer else ''}"
                        )
                    except Exception as e:
                        logger.warning(f"[Item {item_index}] Dream failed for {dream_date}: {e}")
                    dream_dates_triggered.add(dream_date)

            # Format and ingest the session
            formatted_msgs = format_messages_for_reme(messages, session_dt)
            date_str = session_dt.strftime("%Y-%m-%d")

            logger.info(
                f"[Item {item_index}] Ingesting session {idx+1}/{len(sorted_sessions)} "
                f"id={session_id} date={date_str} msgs={len(formatted_msgs)}"
            )
            resp = await app.run_job(
                "auto_memory",
                messages=formatted_msgs,
                session_id=session_id,
                date=date_str,
            )
            if not resp.success:
                logger.warning(
                    f"[Item {item_index}] auto_memory failed for session {session_id}: {resp.answer}"
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

        # ── Phase 3: Reindex + digest update ──────────────────────────
        logger.info(f"[Item {item_index}] Running reindex...")
        await app.run_job("reindex")
        await app.run_job("digest_update")

        # ── Phase 4: Ask question via search + answer generation ─────
        question = item["question"]
        logger.info(f"[Item {item_index}] Searching for answer to: {question[:80]}...")
        search_resp = await app.run_job("search", query=question, limit=10)

        if search_resp.success and search_resp.answer:
            search_context = search_resp.answer
        else:
            search_context = "(no relevant information found)"

        logger.info(f"[Item {item_index}] Search context: {search_context[:200]}...")

        # Generate answer using the answer model
        answer_llm = app.context.components[ComponentEnum.AS_LLM]["answer"]
        answer_prompt = ANSWER_PROMPT.format(context=search_context, question=question)
        answer_msg = Msg(name="user", role="user", content=[{"type": "text", "text": answer_prompt}])
        answer_response = await answer_llm.model([answer_msg])
        system_response = ""
        for block in answer_response.content:
            if hasattr(block, "text"):
                system_response += block.text
        system_response = system_response.strip()
        if not system_response:
            system_response = search_context  # fallback to raw search results

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
# Entry point
# ---------------------------------------------------------------------------
async def main(config_path: str | None = None):
    eval_config = load_eval_config(config_path)
    dataset_cfg = eval_config["dataset"]

    # Load dataset
    dataset_path = _PROJECT_ROOT / dataset_cfg["path"]
    logger.info(f"Loading dataset from {dataset_path}")
    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)

    start = dataset_cfg.get("start_index", 0)
    num_items = dataset_cfg.get("num_items", 1)
    items_to_eval = data[start: start + num_items]
    logger.info(f"Evaluating {len(items_to_eval)} item(s) starting from index {start}")

    # Create output directory
    output_dir = _PROJECT_ROOT / eval_config["output"]["dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run evaluation
    results = []
    for i, item in enumerate(items_to_eval):
        logger.info(f"{'='*60}")
        logger.info(f"Evaluating item {start + i}: {item['question_id']}")
        logger.info(f"{'='*60}")
        result = await evaluate_item(item, eval_config, start + i)
        results.append(result)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"results_{dataset_cfg['variant']}_{timestamp}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Results saved to {output_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    for r in results:
        print(f"\n  Question ID: {r['question_id']}")
        print(f"  Type: {r['question_type']}")
        print(f"  Question: {r['question'][:60]}...")
        print(f"  Ground Truth: {r['ground_truth'][:60]}...")
        print(f"  System Response: {r['system_response'][:60]}...")
        for metric, judgment in r["judgments"].items():
            if metric == "binary":
                print(f"  Binary: {judgment.get('verdict', 'N/A')} — {judgment.get('reason', '')[:60]}")
            elif metric == "score":
                print(f"  Score: {judgment.get('score', 'N/A')}/5 — {judgment.get('reason', '')[:60]}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LongMemEval evaluation runner")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    args = parser.parse_args()
    asyncio.run(main(args.config))
