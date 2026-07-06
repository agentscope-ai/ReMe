#!/usr/bin/env python3
"""Write benchmark prediction session ids into each LongMemEval question dir."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_DATASET_DIR = Path("/Users/yuli/workspace/ReMe/longmemeval")
DEFAULT_SIMULATOR = Path("/Users/yuli/workspace/ReMe/scripts/simulate_longmemeval_benchmark.py")


def load_simulator(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("simulate_longmemeval_benchmark", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load simulator from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def prediction_payload(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "predicted_haystack_session_ids": result["predicted_haystack_session_ids"],
        "score": result["score"],
        "hit_answer_session_ids": result["hit_answer_session_ids"],
    }


def iter_question_dirs(dataset_dir: Path) -> list[Path]:
    question_dirs = [
        path for path in dataset_dir.iterdir() if path.is_dir() and path.name.isdigit()
    ]
    return sorted(question_dirs, key=lambda path: int(path.name))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write agent.json predictions for LongMemEval benchmark variants."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--simulator", type=Path, default=DEFAULT_SIMULATOR)
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sim = load_simulator(args.simulator)
    question_dirs = iter_question_dirs(args.dataset_dir)

    counts = {
        "questions": 0,
        "baseline_score_sum": 0.0,
        "memory_time_range_soft_score_sum": 0.0,
        "memory_time_range_hard_filter_score_sum": 0.0,
    }

    for question_dir in question_dirs:
        baseline = sim.evaluate_question(
            question_dir,
            args.top_k,
            time_aware=False,
        )
        soft = sim.evaluate_question(
            question_dir,
            args.top_k,
            time_aware=False,
            use_reme_time_range=True,
            filter_by_time_range=False,
        )
        hard = sim.evaluate_question(
            question_dir,
            args.top_k,
            time_aware=False,
            use_reme_time_range=True,
            filter_by_time_range=True,
        )

        payload = {
            "question_index": int(question_dir.name),
            "question_id": baseline["question_id"],
            "question_type": baseline["question_type"],
            "top_k": args.top_k,
            "gold_answer_session_ids": baseline["gold_answer_session_ids"],
            "baseline_no_time": prediction_payload(baseline),
            "memory_time_range_soft_boost": prediction_payload(soft),
            "memory_time_range_hard_filter": prediction_payload(hard),
        }
        write_json(question_dir / "agent.json", payload)

        counts["questions"] += 1
        counts["baseline_score_sum"] += baseline["score"]
        counts["memory_time_range_soft_score_sum"] += soft["score"]
        counts["memory_time_range_hard_filter_score_sum"] += hard["score"]

    summary = {
        "dataset_dir": str(args.dataset_dir),
        "questions": counts["questions"],
        "top_k": args.top_k,
        "average_scores": {
            "baseline_no_time": counts["baseline_score_sum"] / counts["questions"],
            "memory_time_range_soft_boost": counts["memory_time_range_soft_score_sum"]
            / counts["questions"],
            "memory_time_range_hard_filter": counts[
                "memory_time_range_hard_filter_score_sum"
            ]
            / counts["questions"],
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
