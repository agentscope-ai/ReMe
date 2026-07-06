"""Golden-session evaluation for LongMemEval.

Tests prompt-based answer accuracy when the LLM is given the *golden*
answer sessions directly (no retrieval). For each question, only the
sessions listed in `answer_session_ids` are used as context.

Usage:
    # Test first item with 1 worker
    python benchmark/longmemeval/run_golden_session.py --num_items 1 --num_workers 1

    # Run all 500 items with 32 concurrent workers
    python benchmark/longmemeval/run_golden_session.py --num_workers 32
"""

import argparse
import asyncio
import json
import logging
import os
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("golden_session_eval")

# ---------------------------------------------------------------------------
# Constants (same as run.py)
# ---------------------------------------------------------------------------
DATASET_PATH = _PROJECT_ROOT / "datasets" / "longmemeval" / "longmemeval_s_cleaned.json"
JUDGE_PROMPTS_PATH = _PROJECT_ROOT / "datasets" / "longmemeval" / "llm-as-judge.json"

# Model configuration – mirrors longmemeval.yaml prompted / judge components
ANSWER_MODEL = os.environ.get("PROMPTED_MODEL_NAME", "qwen3.7-max")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL_NAME", "qwen3.7-max")
API_KEY = os.environ.get("LLM_API_KEY", "")
BASE_URL = os.environ.get("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

# Prompted-answer system prompt (identical to run.py)
PROMPTED_SYSTEM_PROMPT = (
    "You are a memory retrieval assistant. You will be given retrieved memory chunks "
    "and a question. Think carefully step by step about the retrieved context, "
    "then output ONLY the direct factual answer.\n\n"
    "## Rules\n"
    "- Answer based ONLY on the retrieved context provided below.\n"
    "- Then provide a very CONCISE answer (short phrase about core information)."
)

PROMPTED_TEMPORAL_HINT = "\n\nCurrent time context: {query_time}\n"

BINARY_JUDGE_PROMPT = """\
{judge_instruction}

Question: {question}
Ground-truth answer: {answer}
System response: {response}

Reply with ONLY a JSON object: {{"verdict": "yes" or "no", "reason": "brief explanation"}}"""


# ---------------------------------------------------------------------------
# Judge prompts (per question type)
# ---------------------------------------------------------------------------
def _load_judge_prompts() -> dict:
    with open(JUDGE_PROMPTS_PATH, encoding="utf-8") as f:
        return json.load(f)


_JUDGE_PROMPTS: dict = _load_judge_prompts()


def get_judge_instruction(question_type: str) -> str:
    return _JUDGE_PROMPTS.get(question_type, _JUDGE_PROMPTS["__default__"])


# ---------------------------------------------------------------------------
# Date utilities
# ---------------------------------------------------------------------------
def parse_haystack_date(date_str: str) -> datetime:
    m = re.match(r"(\d{4}/\d{2}/\d{2})\s+\(\w+\)\s+(\d{2}:\d{2})", date_str)
    if not m:
        raise ValueError(f"Cannot parse haystack date: {date_str!r}")
    return datetime.strptime(f"{m.group(1)} {m.group(2)}", "%Y/%m/%d %H:%M")


def to_iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


# ---------------------------------------------------------------------------
# Session formatting
# ---------------------------------------------------------------------------
def format_golden_sessions(
    item: dict, question_dt: datetime | None = None
) -> tuple[str, int, int, list[dict]]:
    """Build context text from the golden answer sessions.

    Args:
        item: Dataset item containing haystack and answer_session_ids.
        question_dt: If provided, sessions with timestamp > question_dt are filtered out.

    Returns:
        (context_text, total_count, kept_count, filtered_records)
        filtered_records: list of {question_id, session_id, session_date, question_date}
    """
    sid_to_idx = {sid: i for i, sid in enumerate(item["haystack_session_ids"])}
    answer_sids = item["answer_session_ids"]
    total_count = len(answer_sids)
    qid = item["question_id"]

    parts = []
    kept_count = 0
    filtered_records: list[dict] = []
    for sid in answer_sids:
        idx = sid_to_idx.get(sid)
        if idx is None:
            logger.warning(f"answer_session_id {sid!r} not found in haystack for qid={qid}")
            continue
        date_str = item["haystack_dates"][idx]
        # Filter: skip sessions that occur after question time
        if question_dt is not None:
            try:
                session_dt = parse_haystack_date(date_str)
                if session_dt > question_dt:
                    logger.info(
                        f"[qid={qid}] Filtered future session {sid!r} "
                        f"(session={session_dt}, question={question_dt})"
                    )
                    filtered_records.append({
                        "question_id": qid,
                        "session_id": sid,
                        "session_date": date_str,
                        "question_date": item.get("question_date", ""),
                    })
                    continue
            except ValueError:
                pass  # keep session if date cannot be parsed
        msgs = item["haystack_sessions"][idx]
        session_text = f"--- Session: {sid} (Date: {date_str}) ---\n"
        for msg in msgs:
            role_label = "User" if msg["role"] == "user" else "Assistant"
            session_text += f"{role_label}: {msg['content']}\n"
        parts.append(session_text)
        kept_count += 1

    return "\n".join(parts), total_count, kept_count, filtered_records


# ---------------------------------------------------------------------------
# LLM calls
# ---------------------------------------------------------------------------
async def call_answer_llm(client, context_text: str, question: str, query_time: str) -> str:
    """Call the answer LLM with think mode, same prompt structure as run.py."""
    sys_prompt = PROMPTED_SYSTEM_PROMPT
    if query_time:
        sys_prompt += PROMPTED_TEMPORAL_HINT.format(query_time=query_time)

    user_content = (
        f"## Retrieved Memory Context\n\n{context_text}\n\n"
        f"## Question\n{question}\n\n"
        f"Please provide the direct factual answer based on the above context."
    )

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_content},
    ]

    resp = await client.chat.completions.create(
        model=ANSWER_MODEL,
        messages=messages,
        max_tokens=8192,
        extra_body={"enable_thinking": True},
    )

    # Extract the final answer (non-thinking content)
    content = resp.choices[0].message.content or ""
    return content.strip()


async def call_judge_llm(client, question: str, ground_truth: str, response: str, question_type: str) -> dict:
    """Call the judge LLM, same logic as run.py judge_response."""
    judge_instruction = get_judge_instruction(question_type)
    prompt = BINARY_JUDGE_PROMPT.format(
        judge_instruction=judge_instruction,
        question=question,
        answer=ground_truth,
        response=response,
    )

    messages = [{"role": "user", "content": prompt}]

    resp = await client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=messages,
        max_tokens=2048,
    )

    raw_text = (resp.choices[0].message.content or "").strip()

    try:
        json_match = re.search(r"\{[^}]+\}", raw_text)
        if json_match:
            result = json.loads(json_match.group())
        else:
            result = json.loads(raw_text)
    except json.JSONDecodeError:
        result = {"raw": raw_text, "error": "failed to parse judge response"}

    result["metric"] = "binary"
    result["question_type"] = question_type
    return result


# ---------------------------------------------------------------------------
# Evaluate one item
# ---------------------------------------------------------------------------
async def evaluate_item(client, item: dict, item_index: int, no_time_filter: bool = False) -> dict:
    """Evaluate a single item with golden sessions."""
    question = item["question"]
    question_date_raw = item.get("question_date", "")
    try:
        question_dt = parse_haystack_date(question_date_raw) if question_date_raw else None
    except ValueError:
        question_dt = None
    query_time = to_iso(question_dt) if question_dt else ""

    # Build golden session context (filter sessions after question time unless disabled)
    filter_dt = None if no_time_filter else question_dt
    context_text, total_sessions, kept_sessions, filtered_records = format_golden_sessions(item, filter_dt)
    filtered_count = total_sessions - kept_sessions
    if not context_text:
        context_text = "(no golden sessions found)"

    logger.info(
        f"[Item {item_index}] qid={item['question_id']} type={item['question_type']} "
        f"sessions={kept_sessions}/{total_sessions}"
        + (f" (filtered {filtered_count} future sessions)" if filtered_count else "")
        + f" q={question[:60]}..."
    )

    # Answer
    try:
        response = await call_answer_llm(client, context_text, question, query_time)
    except Exception as e:
        logger.error(f"[Item {item_index}] Answer LLM failed: {e}")
        response = "(answer failed)"

    if not response:
        response = "(no answer generated)"

    logger.info(f"[Item {item_index}] Response: {response[:150]}...")

    # Judge
    try:
        judgment = await call_judge_llm(
            client, question, item["answer"], response, item["question_type"]
        )
    except Exception as e:
        logger.error(f"[Item {item_index}] Judge LLM failed: {e}")
        judgment = {"verdict": "no", "reason": f"judge failed: {e}", "metric": "binary", "question_type": item["question_type"]}

    logger.info(f"[Item {item_index}] Judgment: {judgment}")

    return {
        "question_id": item["question_id"],
        "question_type": item["question_type"],
        "question": question,
        "ground_truth": item["answer"],
        "response": response,
        "judgment": judgment,
        "filtered_sessions": filtered_records,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def run_eval(num_items: int, start_index: int, num_workers: int, no_time_filter: bool = False):
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)

    # Load dataset
    with open(DATASET_PATH, encoding="utf-8") as f:
        data = json.load(f)

    items = data[start_index : start_index + num_items]
    total = len(items)
    filter_mode = "no_time_filter" if no_time_filter else "time_filter"
    logger.info(f"Evaluating {total} items (start={start_index}, workers={num_workers}, mode={filter_mode})")

    sem = asyncio.Semaphore(num_workers)
    results: list[dict | None] = [None] * total
    completed = [0]
    start_time = time.time()

    async def _worker(i: int, item: dict):
        async with sem:
            result = await evaluate_item(client, item, start_index + i, no_time_filter=no_time_filter)
            results[i] = result
            completed[0] += 1
            elapsed = time.time() - start_time
            pct = 100.0 * completed[0] / total
            eta = (elapsed / completed[0] * (total - completed[0])) / 60 if completed[0] else 0
            if completed[0] % max(1, total // 20) == 0 or completed[0] == total:
                print(
                    f"[PROGRESS] {completed[0]}/{total} ({pct:.1f}%) "
                    f"elapsed={elapsed/60:.1f}min ETA={eta:.1f}min",
                    flush=True,
                )

    tasks = [asyncio.create_task(_worker(i, item)) for i, item in enumerate(items)]
    await asyncio.gather(*tasks)

    # ── Setup output directory ──
    output_dir = _PROJECT_ROOT / "benchmark" / "longmemeval" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Collect and save filtered (future) sessions ──
    all_filtered: list[dict] = []
    for r in results:
        if r is not None:
            all_filtered.extend(r.get("filtered_sessions", []))

    total_golden_sessions = sum(
        len(item["answer_session_ids"]) for item in items
    )
    kept_golden = total_golden_sessions - len(all_filtered)

    filtered_file = output_dir / f"filtered_future_sessions_{filter_mode}_{timestamp}.json"
    with open(filtered_file, "w", encoding="utf-8") as f:
        json.dump(all_filtered, f, ensure_ascii=False, indent=2)
    logger.info(
        f"Filtered future sessions: {len(all_filtered)} saved to {filtered_file}"
    )

    # ── Save results ──
    output_file = output_dir / f"golden_session_{filter_mode}_{timestamp}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Results saved to {output_file}")

    # ── Print summary ──
    print("\n" + "=" * 60)
    print("GOLDEN SESSION EVALUATION RESULTS")
    print("=" * 60)

    # Golden session time filter summary
    print(
        f"\n  ── Golden Session Time Filter ──\n"
        f"  Total golden sessions : {total_golden_sessions}\n"
        f"  Kept (valid)          : {kept_golden}\n"
        f"  Filtered (future)     : {len(all_filtered)}  -> {filtered_file}"
    )

    correct = 0
    type_stats: dict = defaultdict(lambda: {"correct": 0, "total": 0})

    for r in results:
        if r is None:
            continue
        qtype = r["question_type"]
        verdict = r.get("judgment", {}).get("verdict", "N/A")
        type_stats[qtype]["total"] += 1
        if verdict == "yes":
            correct += 1
            type_stats[qtype]["correct"] += 1
        print(f"  [{r['question_id']}] type={r['question_type']}  verdict={verdict}")

    print("\n" + "-" * 60)
    print(f"  Items: {total}")
    print(f"  Overall accuracy: {correct}/{total} ({100*correct/total:.1f}%)")
    print("\n  Per-type accuracy:")
    for qtype in sorted(type_stats.keys()):
        s = type_stats[qtype]
        acc = 100 * s["correct"] / s["total"] if s["total"] else 0
        print(f"    {qtype}: {s['correct']}/{s['total']} ({acc:.1f}%)")

    elapsed = time.time() - start_time
    print(f"\n  Total time: {elapsed/60:.1f} min")
    print("=" * 60)
    print("  [DONE] EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Golden session evaluation")
    parser.add_argument("--num_items", type=int, default=500, help="Number of items to evaluate (default: all 500)")
    parser.add_argument("--start_index", type=int, default=0, help="Start index in dataset")
    parser.add_argument("--num_workers", type=int, default=32, help="Concurrent workers (default: 32)")
    parser.add_argument("--no_time_filter", action="store_true", help="Disable time filtering (keep all golden sessions regardless of time)")
    args = parser.parse_args()

    asyncio.run(run_eval(args.num_items, args.start_index, args.num_workers, args.no_time_filter))
