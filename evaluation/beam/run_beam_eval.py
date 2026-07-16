#!/usr/bin/env python3
"""BEAM evaluation runner for ReMe.

Tests ReMe's memory capability using the BEAM 100K dataset, case 1.
- Ingests BEAM chat.json into ReMe's memory system via auto_memory
- Answers probing questions using ReMe search + prompted LLM
- Evaluates answers using BEAM's rubric-based LLM-as-judge (qwen3.7-max)

Usage:
    python evaluation/beam/run_beam_eval.py
    python evaluation/beam/run_beam_eval.py --chat-size 100K --case-id 1
    python evaluation/beam/run_beam_eval.py --eval-only   # reuse existing workspace
"""

import argparse
import asyncio
import json
import logging
import os
import re
import shutil
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

BEAM_ROOT = _PROJECT_ROOT / "datasets" / "BEAM"
RESULTS_DIR = _PROJECT_ROOT / "evaluation" / "beam" / "results"
WORKSPACE_ROOT = _PROJECT_ROOT / "memory_workspaces" / "beam"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("beam_eval")

# Suppress noisy loggers
for _name in ["httpx", "httpcore", "openai", "asyncio", "watchfiles", "filelock"]:
    logging.getLogger(_name).setLevel(logging.WARNING)


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
    with open(chat_path) as f:
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
                messages.append({
                    "name": role,
                    "role": role,
                    "content": msg["content"],
                    "created_at": dt.strftime("%Y-%m-%dT%H:%M:%S"),
                })

        sessions.append({
            "session_id": f"beam_{chat_size}_{case_id}_batch{batch_num}",
            "date": first_dt.strftime("%Y-%m-%d"),
            "messages": messages,
        })

    return sessions


# ---------------------------------------------------------------------------
# ReMe ingestion
# ---------------------------------------------------------------------------
async def ingest_sessions(app, sessions: list[dict]):
    """Ingest all sessions into ReMe memory system."""
    for i, session in enumerate(sessions):
        logger.info(
            f"Ingesting session {i+1}/{len(sessions)}: "
            f"id={session['session_id']} date={session['date']} "
            f"msgs={len(session['messages'])}"
        )
        resp = await app.run_job(
            "auto_memory",
            messages=session["messages"],
            session_id=session["session_id"],
            date=session["date"],
        )
        if not resp.success:
            logger.warning(f"auto_memory failed: {resp.answer}")
        else:
            logger.info(f"  auto_memory success: {resp.answer[:100] if resp.answer else ''}")

        # Index update after each session
        await app.run_job("index_update")

    # Final digest update
    logger.info("Running digest_update...")
    await app.run_job("digest_update")
    logger.info("Ingestion complete.")


# ---------------------------------------------------------------------------
# Answer generation (prompted approach: search + LLM)
# ---------------------------------------------------------------------------
async def answer_question_prompted(
    app, question: str
) -> tuple[str, dict]:
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


# ---------------------------------------------------------------------------
# Answer generation (agentic approach: agentic_answer job)
# ---------------------------------------------------------------------------
async def answer_question_agentic(app, question: str) -> tuple[str, dict]:
    """Answer a probing question using ReMe's agentic agentic_answer job.

    Returns (answer, metadata)
    """
    query_resp = await app.run_job(
        "agentic_answer",
        query=question,
    )
    answer = (query_resp.answer or "").strip()

    return answer, {"mode": "agentic"}


# ---------------------------------------------------------------------------
# BEAM rubric-based LLM-as-Judge (delegates to beam_rubric_judge_step)
# ---------------------------------------------------------------------------
async def judge_answer(app, question: str, llm_response: str, rubric: list[str], question_type: str = "") -> dict:
    """Judge an answer via the answer_judge job.

    The job is defined in beam.yaml and uses the ``judge`` agent_wrapper
    (configured with qwen3.7-max).  The actual prompt and parsing logic live
    in ``reme/steps/benchmark/beam/llm_judge.py``.
    """
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
async def run_beam_eval(
    chat_size: str = "100K",
    case_id: str = "1",
    eval_only: bool = False,
    mode: str = "prompted",
):
    """Run the full BEAM evaluation pipeline.

    Args:
        chat_size: BEAM chat size (100K, 500K, 1M, 10M)
        case_id: Case directory name (e.g. "1")
        eval_only: If True, skip ingestion and reuse existing workspace
        mode: "prompted" (search + LLM) or "agentic" (agentic_answer job)
    """
    # Setup paths
    chat_path = BEAM_ROOT / "chats" / chat_size / case_id / "chat.json"
    probing_questions_path = (
        BEAM_ROOT / "chats" / chat_size / case_id
        / "probing_questions" / "probing_questions.json"
    )
    workspace_dir = WORKSPACE_ROOT / f"{chat_size}_{case_id}"

    if not chat_path.exists():
        raise FileNotFoundError(f"Chat file not found: {chat_path}")
    if not probing_questions_path.exists():
        raise FileNotFoundError(f"Probing questions not found: {probing_questions_path}")

    logger.info(f"BEAM evaluation: size={chat_size} case={case_id} mode={mode}")
    logger.info(f"Chat file: {chat_path}")
    logger.info(f"Probing questions: {probing_questions_path}")
    logger.info(f"Workspace: {workspace_dir}")

    # Create/clean workspace
    if eval_only:
        if not workspace_dir.exists():
            logger.error(f"Workspace not found for eval_only: {workspace_dir}")
            return
        logger.info("eval_only mode: reusing existing workspace")
    else:
        if workspace_dir.exists():
            shutil.rmtree(workspace_dir)
            logger.info(f"Cleaned existing workspace: {workspace_dir}")
        workspace_dir.mkdir(parents=True, exist_ok=True)

    # Initialize ReMe
    from reme import Application
    from reme.config import resolve_app_config

    reme_cfg = resolve_app_config(
        config="beam.yaml",
        workspace_dir=str(workspace_dir / ".reme"),
        log_to_console=True,
        log_to_file=False,
        enable_logo=False,
    )
    app = Application(**reme_cfg)
    await app.start()

    try:
        # ── Phase 1: Ingest sessions ──────────────────────────────
        if not eval_only:
            sessions = load_beam_chat(chat_path, chat_size, case_id)
            logger.info(f"Loaded {len(sessions)} sessions from chat.json")
            await ingest_sessions(app, sessions)

        # ── Phase 2: Answer probing questions ─────────────────────
        with open(probing_questions_path) as f:
            probing_questions = json.load(f)

        total_questions = sum(len(v) for v in probing_questions.values())
        logger.info(f"Total probing questions: {total_questions}")

        answers = {}
        q_idx = 0
        for key in probing_questions:
            logger.info(f"\n{'='*60}")
            logger.info(f"Question type: {key} ({len(probing_questions[key])} questions)")
            logger.info(f"{'='*60}")

            question_answers = []
            for i, q in enumerate(probing_questions[key]):
                q_idx += 1
                question = q["question"]
                logger.info(f"\n[{q_idx}/{total_questions}] {key} Q{i+1}: {question[:100]}...")

                try:
                    if mode == "agentic":
                        answer, metadata = await answer_question_agentic(app, question)
                    else:
                        answer, metadata = await answer_question_prompted(
                            app, question
                        )
                except Exception as e:
                    logger.error(f"  Answer generation failed: {e}")
                    answer = f"(error: {e})"
                    metadata = {"error": str(e)}

                if not answer:
                    answer = "(no answer generated)"

                logger.info(f"  Answer: {answer[:200]}...")

                q_copy = dict(q)
                q_copy["llm_response"] = answer
                q_copy["_metadata"] = metadata
                question_answers.append(q_copy)

            answers[key] = question_answers

        # Save answers
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        answers_file = RESULTS_DIR / f"answers_{chat_size}_{case_id}_{mode}.json"
        with open(answers_file, "w", encoding="utf-8") as f:
            json.dump(answers, f, indent=4, ensure_ascii=False)
        logger.info(f"\nAnswers saved to {answers_file}")

        # ── Phase 3: Judge answers ────────────────────────────────
        logger.info(f"\n{'='*60}")
        logger.info("Phase 3: LLM-as-Judge evaluation")
        logger.info(f"{'='*60}")

        evaluation = {}
        for key in answers:
            logger.info(f"\nJudging: {key}")
            eval_results = []
            for i, q in enumerate(answers[key]):
                logger.info(f"  Judging {key} Q{i+1}...")
                result = await judge_answer(
                    app,
                    q["question"],
                    q["llm_response"],
                    q["rubric"],
                    question_type=key,
                )
                logger.info(f"  Score: {result['llm_judge_score']:.3f}")
                if "event_ordering" in result:
                    eo = result["event_ordering"]
                    logger.info(f"  EO: P={eo.get('precision', 0):.3f} R={eo.get('recall', 0):.3f} F1={eo.get('f1', 0):.3f} tau={eo.get('tau_norm', 0):.3f} final={eo.get('final_score', 0):.3f}")
                eval_results.append(result)
            evaluation[key] = eval_results

        # Save evaluation
        eval_file = RESULTS_DIR / f"evaluation_{chat_size}_{case_id}_{mode}.json"
        with open(eval_file, "w", encoding="utf-8") as f:
            json.dump(evaluation, f, indent=4, ensure_ascii=False)
        logger.info(f"\nEvaluation saved to {eval_file}")

        # ── Phase 4: Print summary ────────────────────────────────
        print("\n" + "=" * 70)
        print(f"  BEAM EVALUATION RESULTS  |  size={chat_size}  case={case_id}  mode={mode}")
        print("=" * 70)

        total_score = 0.0
        total_q_count = 0
        type_scores = {}

        for key in evaluation:
            scores = [r["llm_judge_score"] for r in evaluation[key]]
            avg = sum(scores) / len(scores) if scores else 0
            type_scores[key] = avg
            total_score += sum(scores)
            total_q_count += len(scores)
            print(f"  {key:40s}: {avg:.3f}  ({len(scores)} Qs)")

        overall = total_score / total_q_count if total_q_count else 0
        print("-" * 70)
        print(f"  {'OVERALL':40s}: {overall:.3f}  ({total_q_count} Qs)")
        print("=" * 70)

        # Save summary
        summary = {
            "chat_size": chat_size,
            "case_id": case_id,
            "mode": mode,
            "overall_score": overall,
            "total_questions": total_q_count,
            "per_type_scores": type_scores,
            "answers_file": str(answers_file),
            "evaluation_file": str(eval_file),
        }
        summary_file = RESULTS_DIR / f"summary_{chat_size}_{case_id}_{mode}.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)
        logger.info(f"Summary saved to {summary_file}")

    finally:
        await app.close()

    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BEAM evaluation for ReMe")
    parser.add_argument("--chat-size", type=str, default="100K",
                        help="BEAM chat size (100K, 500K, 1M, 10M)")
    parser.add_argument("--case-id", type=str, default="1",
                        help="Case directory name")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip ingestion, reuse existing workspace")
    parser.add_argument("--mode", type=str, default="prompted",
                        choices=["prompted", "agentic"],
                        help="Answer mode: prompted (search+LLM) or agentic (ReAct)")
    args = parser.parse_args()

    start = time.time()
    summary = asyncio.run(run_beam_eval(
        chat_size=args.chat_size,
        case_id=args.case_id,
        eval_only=args.eval_only,
        mode=args.mode,
    ))
    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed/60:.1f} min")
