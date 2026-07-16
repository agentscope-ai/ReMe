#!/usr/bin/env python3
"""Batch BEAM evaluation runner — runs all cases for a given chat size.

Usage:
    /home/jiangniurou.xyf/miniconda3/envs/reme/bin/python3 evaluation/beam/run_beam_eval_batch.py --chat-size 100K
    /home/jiangniurou.xyf/miniconda3/envs/reme/bin/python3 evaluation/beam/run_beam_eval_batch.py --chat-size 100K --eval-only
    /home/jiangniurou.xyf/miniconda3/envs/reme/bin/python3 evaluation/beam/run_beam_eval_batch.py --chat-size 100K --case-ids 1 2 3
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

BEAM_ROOT = _PROJECT_ROOT / "datasets" / "BEAM"
RESULTS_DIR = _PROJECT_ROOT / "evaluation" / "beam" / "results"

# Add project root to path so we can import run_beam_eval
sys.path.insert(0, str(_PROJECT_ROOT))


def get_available_cases(chat_size: str) -> list[str]:
    """Return sorted list of case IDs for a given chat size."""
    chats_dir = BEAM_ROOT / "chats" / chat_size
    if not chats_dir.exists():
        return []
    return sorted(
        [d.name for d in chats_dir.iterdir() if d.is_dir()],
        key=lambda x: int(x),
    )


async def run_batch(chat_size: str, case_ids: list[str], eval_only: bool, mode: str):
    """Run BEAM evaluation for all specified cases sequentially."""
    from evaluation.beam.run_beam_eval import run_beam_eval

    total = len(case_ids)
    all_summaries = {}
    failed_cases = []

    start = time.time()
    print(f"\n{'='*70}")
    print(f"  BEAM BATCH EVALUATION")
    print(f"  chat_size={chat_size}  cases={case_ids}  mode={mode}  eval_only={eval_only}")
    print(f"{'='*70}\n")

    for i, case_id in enumerate(case_ids, 1):
        print(f"\n{'#'*70}")
        print(f"  CASE {case_id}  ({i}/{total})")
        print(f"{'#'*70}\n")

        case_start = time.time()
        try:
            summary = await run_beam_eval(
                chat_size=chat_size,
                case_id=case_id,
                eval_only=eval_only,
                mode=mode,
            )
            all_summaries[case_id] = summary
            case_elapsed = time.time() - case_start
            print(f"\n  Case {case_id} done in {case_elapsed/60:.1f} min")
        except Exception as e:
            case_elapsed = time.time() - case_start
            print(f"\n  Case {case_id} FAILED after {case_elapsed/60:.1f} min: {e}")
            import traceback
            traceback.print_exc()
            failed_cases.append(case_id)
            all_summaries[case_id] = {"error": str(e)}

        elapsed_so_far = time.time() - start
        avg_per_case = elapsed_so_far / i
        remaining = avg_per_case * (total - i)
        print(f"\n  Progress: {i}/{total}  Elapsed: {elapsed_so_far/60:.1f} min  "
              f"ETA: {remaining/60:.1f} min")

    # ── Aggregate results ──
    print(f"\n{'='*70}")
    print(f"  AGGREGATE RESULTS  |  size={chat_size}  mode={mode}")
    print(f"{'='*70}")

    # Compute average scores across successful cases
    type_scores = {}
    successful_cases = []

    for case_id, summary in all_summaries.items():
        if "error" in summary:
            continue
        successful_cases.append(case_id)
        scores = summary.get("scores", {})
        for qtype, score in scores.items():
            if qtype not in type_scores:
                type_scores[qtype] = []
            type_scores[qtype].append(score)

    # Print per-type averages
    all_scores = []
    for qtype in sorted(type_scores.keys()):
        scores = type_scores[qtype]
        avg = sum(scores) / len(scores) if scores else 0
        all_scores.append(avg)
        print(f"  {qtype:<40s}: {avg:.3f}  ({len(scores)} cases)")

    overall = sum(all_scores) / len(all_scores) if all_scores else 0
    print(f"  {'-'*68}")
    print(f"  {'OVERALL':<40s}: {overall:.3f}  ({len(successful_cases)} successful cases)")

    if failed_cases:
        print(f"\n  FAILED cases: {failed_cases}")

    print(f"{'='*70}")

    total_elapsed = time.time() - start
    print(f"\nTotal time: {total_elapsed/60:.1f} min")

    # Save aggregate results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    aggregate = {
        "chat_size": chat_size,
        "mode": mode,
        "successful_cases": successful_cases,
        "failed_cases": failed_cases,
        "type_averages": {k: sum(v)/len(v) for k, v in type_scores.items()},
        "overall": overall,
        "per_case": all_summaries,
    }
    agg_file = RESULTS_DIR / f"aggregate_{chat_size}_{mode}.json"
    with open(agg_file, "w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=4, ensure_ascii=False)
    print(f"\nAggregate saved to {agg_file}")

    return aggregate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch BEAM evaluation")
    parser.add_argument("--chat-size", type=str, default="100K")
    parser.add_argument("--case-ids", nargs="+", default=None,
                        help="Specific case IDs to run (default: all)")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip ingestion, reuse existing workspace")
    parser.add_argument("--mode", type=str, default="prompted",
                        choices=["prompted", "agentic"])
    args = parser.parse_args()

    case_ids = args.case_ids or get_available_cases(args.chat_size)
    if not case_ids:
        print(f"No cases found for chat_size={args.chat_size}")
        sys.exit(1)

    print(f"Cases to run: {case_ids}")
    asyncio.run(run_batch(
        chat_size=args.chat_size,
        case_ids=case_ids,
        eval_only=args.eval_only,
        mode=args.mode,
    ))
