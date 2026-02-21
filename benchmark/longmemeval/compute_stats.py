"""
LongMemEval Evaluation Statistics Analyzer

Computes detailed statistics from evaluation results including:
- Overall accuracy
- Accuracy by question type
- Timing statistics (summary, retrieval)
- Memory extraction statistics

Usage:
    python bench/longmemeval/compute_stats.py \
        --results_dir bench/longmemeval/bench_results/longmemeval_reme
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_results(results_dir: str) -> list[dict]:
    """Load all question result files from the directory.

    Args:
        results_dir: Path to the results directory

    Returns:
        List of result dictionaries
    """
    results_path = Path(results_dir)
    results = []

    # Load individual question files
    question_files = sorted(results_path.glob("question_*.json"))

    for file_path in question_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                result = json.load(f)
                results.append(result)
        except Exception as e:
            print(f"⚠️ Error loading {file_path}: {e}")

    return results


def compute_accuracy_stats(results: list[dict]) -> dict[str, Any]:
    """Compute overall and per-type accuracy statistics.

    Args:
        results: List of result dictionaries

    Returns:
        Dictionary with accuracy statistics
    """
    total = len(results)
    correct = 0
    incorrect = 0
    error = 0

    # Per question type statistics
    type_stats = defaultdict(lambda: {"total": 0, "correct": 0, "incorrect": 0, "error": 0})

    for r in results:
        qtype = r.get("question_type", "unknown")
        judgment = r.get("judgment", {})
        is_correct = judgment.get("is_correct")

        type_stats[qtype]["total"] += 1

        if is_correct is True:
            correct += 1
            type_stats[qtype]["correct"] += 1
        elif is_correct is False:
            incorrect += 1
            type_stats[qtype]["incorrect"] += 1
        else:
            error += 1
            type_stats[qtype]["error"] += 1

    # Compute accuracies
    overall = {
        "total": total,
        "correct": correct,
        "incorrect": incorrect,
        "error": error,
        "accuracy": correct / total if total > 0 else 0,
        "accuracy_valid": correct / (correct + incorrect) if (correct + incorrect) > 0 else 0,
    }

    by_type = {}
    for qtype, stats in type_stats.items():
        valid = stats["correct"] + stats["incorrect"]
        by_type[qtype] = {
            **stats,
            "accuracy": stats["correct"] / stats["total"] if stats["total"] > 0 else 0,
            "accuracy_valid": stats["correct"] / valid if valid > 0 else 0,
        }

    return {
        "overall": overall,
        "by_question_type": by_type,
    }


def compute_timing_stats(results: list[dict]) -> dict[str, Any]:
    """Compute timing statistics.

    Args:
        results: List of result dictionaries

    Returns:
        Dictionary with timing statistics
    """
    summary_times = []
    retrieve_times = []

    for r in results:
        summary_ms = r.get("summary_duration_ms", 0)
        retrieve_ms = r.get("retrieve_duration_ms", 0)

        if summary_ms > 0:
            summary_times.append(summary_ms)
        if retrieve_ms > 0:
            retrieve_times.append(retrieve_ms)

    def compute_stats(times: list[float]) -> dict:
        if not times:
            return {"count": 0, "total_ms": 0, "avg_ms": 0, "min_ms": 0, "max_ms": 0}

        return {
            "count": len(times),
            "total_ms": sum(times),
            "total_min": sum(times) / 1000 / 60,
            "avg_ms": sum(times) / len(times),
            "min_ms": min(times),
            "max_ms": max(times),
        }

    return {
        "summary": compute_stats(summary_times),
        "retrieve": compute_stats(retrieve_times),
        "total_time_min": (sum(summary_times) + sum(retrieve_times)) / 1000 / 60,
    }


def compute_memory_stats(results: list[dict]) -> dict[str, Any]:
    """Compute memory extraction statistics.

    Args:
        results: List of result dictionaries

    Returns:
        Dictionary with memory statistics
    """
    memory_counts = []
    session_counts = []

    for r in results:
        memories = r.get("extracted_memories", [])
        num_sessions = r.get("num_sessions", 0)

        memory_counts.append(len(memories))
        session_counts.append(num_sessions)

    def compute_stats(counts: list[int]) -> dict:
        if not counts:
            return {"count": 0, "total": 0, "avg": 0, "min": 0, "max": 0}

        return {
            "count": len(counts),
            "total": sum(counts),
            "avg": sum(counts) / len(counts),
            "min": min(counts),
            "max": max(counts),
        }

    return {
        "memories_per_question": compute_stats(memory_counts),
        "sessions_per_question": compute_stats(session_counts),
    }


def print_report(
    accuracy_stats: dict,
    timing_stats: dict,
    memory_stats: dict,
    results_dir: str,
):
    """Print formatted statistics report.

    Args:
        accuracy_stats: Accuracy statistics
        timing_stats: Timing statistics
        memory_stats: Memory statistics
        results_dir: Path to results directory
    """
    print("\n" + "=" * 80)
    print("LONGMEMEVAL EVALUATION STATISTICS")
    print(f"Results Directory: {results_dir}")
    print("=" * 80)

    # Overall accuracy
    overall = accuracy_stats["overall"]
    print("\n📊 Overall Accuracy:")
    print(f"  Total Questions:    {overall['total']}")
    print(f"  ✅ Correct:         {overall['correct']} ({100 * overall['accuracy']:.2f}%)")
    print(
        f"  ❌ Incorrect:       {overall['incorrect']} "
        f"({100 * overall['incorrect'] / overall['total'] if overall['total'] > 0 else 0:.2f}%)",
    )
    if overall["error"] > 0:
        print(f"  ⚠️ Error:           {overall['error']} ({100 * overall['error'] / overall['total']:.2f}%)")
    print(f"  Accuracy (valid):   {100 * overall['accuracy_valid']:.2f}%")

    # Accuracy by question type
    print("\n📊 Accuracy by Question Type:")
    print("-" * 60)
    print(f"{'Question Type':<30} {'Correct':<10} {'Total':<10} {'Accuracy':<10}")
    print("-" * 60)

    by_type = accuracy_stats["by_question_type"]
    for qtype in sorted(by_type.keys()):
        stats = by_type[qtype]
        print(f"{qtype:<30} {stats['correct']:<10} {stats['total']:<10} {100 * stats['accuracy']:.2f}%")

    print("-" * 60)

    # Timing statistics
    print("\n⏱️ Timing Statistics:")
    summary = timing_stats["summary"]
    retrieve = timing_stats["retrieve"]

    print("  Memory Summarization:")
    print(f"    Total Time:  {summary['total_min']:.2f} min")
    print(f"    Avg per Q:   {summary['avg_ms']:.0f} ms")
    print(f"    Min/Max:     {summary['min_ms']:.0f} / {summary['max_ms']:.0f} ms")

    print("  Memory Retrieval:")
    print(f"    Total Time:  {retrieve['total_min']:.2f} min")
    print(f"    Avg per Q:   {retrieve['avg_ms']:.0f} ms")
    print(f"    Min/Max:     {retrieve['min_ms']:.0f} / {retrieve['max_ms']:.0f} ms")

    print(f"  Total Time:    {timing_stats['total_time_min']:.2f} min")

    # Memory statistics
    print("\n📝 Memory Statistics:")
    mem = memory_stats["memories_per_question"]
    sess = memory_stats["sessions_per_question"]

    print("  Extracted Memories per Question:")
    print(f"    Total:   {mem['total']}")
    print(f"    Average: {mem['avg']:.1f}")
    print(f"    Min/Max: {mem['min']} / {mem['max']}")

    print("  Sessions per Question:")
    print(f"    Average: {sess['avg']:.1f}")
    print(f"    Min/Max: {sess['min']} / {sess['max']}")

    print("\n" + "=" * 80)


def save_statistics(
    accuracy_stats: dict,
    timing_stats: dict,
    memory_stats: dict,
    output_file: str,
):
    """Save statistics to JSON file.

    Args:
        accuracy_stats: Accuracy statistics
        timing_stats: Timing statistics
        memory_stats: Memory statistics
        output_file: Path to output file
    """
    stats = {
        "accuracy": accuracy_stats,
        "timing": timing_stats,
        "memory": memory_stats,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4, ensure_ascii=False)

    print(f"\n📁 Statistics saved to: {output_file}")


def main(results_dir: str, output_file: str = None):
    """Main function to compute and display statistics.

    Args:
        results_dir: Path to results directory
        output_file: Optional path to save statistics JSON
    """
    print(f"\nLoading results from: {results_dir}")

    results = load_results(results_dir)

    if not results:
        print("❌ No results found!")
        return

    print(f"Loaded {len(results)} question results")

    # Compute statistics
    accuracy_stats = compute_accuracy_stats(results)
    timing_stats = compute_timing_stats(results)
    memory_stats = compute_memory_stats(results)

    # Print report
    print_report(accuracy_stats, timing_stats, memory_stats, results_dir)

    # Save to file if specified
    if output_file:
        save_statistics(accuracy_stats, timing_stats, memory_stats, output_file)
    else:
        # Default output file in results directory
        default_output = Path(results_dir) / "statistics.json"
        save_statistics(accuracy_stats, timing_stats, memory_stats, str(default_output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute statistics from LongMemEval evaluation results",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="bench_results/longmemeval_reme",
        help="Path to results directory containing question_*.json files",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to save statistics JSON (default: <results_dir>/statistics.json)",
    )

    args = parser.parse_args()

    main(
        results_dir=args.results_dir,
        output_file=args.output_file,
    )
