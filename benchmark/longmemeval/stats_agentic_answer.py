#!/usr/bin/env python3
"""Summarise the ``agentic_answer`` results across all LongMemEval samples.

Reports progress (how many of the 500 samples produced ``mem_answer.json``) and a
breakdown of answer *status*:
  - answered      — a non-empty answer that is not "not provided";
  - not_provided  — the agent gave up ("not provided");
  - empty         — ``mem_answer.json`` exists but the answer is blank;
  - missing       — no ``mem_answer.json`` yet.

Everything is broken down by ``question_type``. This script does NOT judge answer
correctness (there is no grader for ``mem_answer`` yet) — it only tracks progress
and collects predicted-vs-golden pairs. Tool-call statistics are read from the
aggregate written by ``run_agentic_answer.py`` when it is present.

Examples:
    python benchmark/longmemeval/stats_agentic_answer.py
    python benchmark/longmemeval/stats_agentic_answer.py --list-run-failed
    python benchmark/longmemeval/stats_agentic_answer.py --list-unanswered
    python benchmark/longmemeval/stats_agentic_answer.py --json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "datasets" / "longmemeval"
LOGBASE = REPO / "logs" / "agentic_answer"
AGGREGATE = LOGBASE / "aggregate.json"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list-unanswered", action="store_true", help="list samples answered 'not provided' or empty")
    p.add_argument("--list-run-failed", action="store_true", help="list launched samples with no readable output")
    p.add_argument("--json", action="store_true", help="emit the summary as JSON")
    return p.parse_args()


def sample_ids() -> list[str]:
    """List all sample IDs (numeric workspace dirs), numerically sorted."""
    ids = [p.name for p in DATA.iterdir() if p.is_dir() and p.name.isdigit()]
    return sorted(ids, key=int)


def pct(num: int, den: int) -> str:
    """Format a percentage."""
    return f"{(100.0 * num / den):.1f}%" if den else "n/a"


def logged_sample_ids() -> list[str]:
    """List sample IDs that have an agentic_answer launch log."""
    logdir = LOGBASE / "agentic_answer"
    if not logdir.exists():
        return []
    ids = [p.stem for p in logdir.glob("*.log") if p.stem.isdigit()]
    return sorted(ids, key=int)


def answer_status(pred: str, has_file: bool) -> str:
    """Classify an answer into answered / not_provided / empty / missing."""
    if not has_file:
        return "missing"
    if not pred:
        return "empty"
    if "not provided" in pred.lower():
        return "not_provided"
    return "answered"


def load_tool_calls() -> dict[str, int]:
    """Map idx -> num_tool_calls from the aggregate, if it exists."""
    if not AGGREGATE.exists():
        return {}
    try:
        with AGGREGATE.open(encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    return {s["idx"]: s.get("num_tool_calls", 0) for s in data.get("samples", []) if "idx" in s}


def main() -> int:
    """Main entry point."""
    args = parse_args()
    ids = sample_ids()
    total = len(ids)
    tool_calls = load_tool_calls()

    rows, unreadable = [], []
    finished_ids = set()
    for idx in ids:
        query_path = DATA / idx / "query.json"
        mem_path = DATA / idx / "mem_answer.json"
        qtype = "(unknown)"
        try:
            with query_path.open(encoding="utf-8") as f:
                qtype = json.load(f).get("question_type") or "(unknown)"
        except (OSError, json.JSONDecodeError):
            pass

        has_file = mem_path.exists()
        pred = ""
        if has_file:
            try:
                with mem_path.open(encoding="utf-8") as f:
                    pred = str(json.load(f).get("answer") or "").strip()
                finished_ids.add(idx)
            except (OSError, json.JSONDecodeError):
                unreadable.append(idx)
                has_file = False

        rows.append({"idx": idx, "type": qtype, "status": answer_status(pred, has_file)})

    finished = [r for r in rows if r["status"] != "missing"]
    n = len(finished)
    launched = logged_sample_ids()
    run_failed = [idx for idx in launched if idx not in finished_ids]

    # Overall status tallies.
    status_counts: dict[str, int] = defaultdict(int)
    for r in rows:
        status_counts[r["status"]] += 1
    answered = status_counts["answered"]
    unanswered = [r["idx"] for r in rows if r["status"] in ("not_provided", "empty")]

    calls_vals = [tool_calls[i] for i in finished_ids if i in tool_calls]
    avg_calls = sum(calls_vals) / len(calls_vals) if calls_vals else 0.0

    # Per question_type breakdown.
    by_type: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "answered": 0})
    for r in finished:
        by_type[r["type"]]["n"] += 1
        by_type[r["type"]]["answered"] += 1 if r["status"] == "answered" else 0

    if args.json:
        print(
            json.dumps(
                {
                    "total": total,
                    "finished": n,
                    "pending": total - n - len(unreadable),
                    "unreadable": unreadable,
                    "launched": len(launched),
                    "run_failed": run_failed,
                    "status_counts": dict(status_counts),
                    "answered_rate": round(answered / n, 4) if n else None,
                    "avg_tool_calls": round(avg_calls, 2) if calls_vals else None,
                    "by_type": {
                        t: {**c, "answered_rate": round(c["answered"] / c["n"], 4)} for t, c in by_type.items()
                    },
                    "unanswered": unanswered,
                    "aggregate": str(AGGREGATE) if AGGREGATE.exists() else None,
                },
                ensure_ascii=False,
                indent=2,
            ),
        )
        return 0

    print("=" * 60)
    print("LongMemEval agentic_answer 统计")
    print("=" * 60)
    print(f"样例总数            : {total}")
    print(f"已完成 (有产出)     : {n}  ({pct(n, total)})")
    print(f"未完成              : {total - n - len(unreadable)}")
    if unreadable:
        print(f"损坏/无法解析       : {len(unreadable)}  {unreadable}")
    print(f"已启动过 (有 log)   : {len(launched)}")
    print(f"运行失败/无可读产出 : {len(run_failed)}")
    print("-" * 60)
    print(f"已作答 (非 not provided): {answered}  ({pct(answered, n)} of finished)")
    print(f"  其中 not provided : {status_counts['not_provided']}")
    print(f"  其中 空答案       : {status_counts['empty']}")
    if calls_vals:
        print(f"平均工具调用次数    : {avg_calls:.1f}   (来自 {AGGREGATE.name})")
    else:
        print("平均工具调用次数    : n/a   (先跑 run_agentic_answer.py 生成 aggregate.json)")
    print("-" * 60)
    print("按 question_type:")
    print(f"  {'type':<24} {'n':>4} {'已作答率':>12}")
    for t in sorted(by_type):
        c = by_type[t]
        print(f"  {t:<24} {c['n']:>4} {pct(c['answered'], c['n']):>12}")

    if args.list_unanswered:
        print("-" * 60)
        print(f"not provided / 空答案的样例 ({len(unanswered)}): {unanswered}")
    if args.list_run_failed:
        print("-" * 60)
        print(f"运行失败/无可读 mem_answer.json 的样例 ({len(run_failed)}): {run_failed}")
        for idx in run_failed:
            print(f"  {idx}: {LOGBASE / 'agentic_answer' / f'{idx}.log'}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
