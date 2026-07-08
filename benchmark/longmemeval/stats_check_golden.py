#!/usr/bin/env python3
"""Summarise the ``check_golden.json`` verdicts across all LongMemEval samples.

Reports progress (how many of the 500 samples have finished) and accuracy:
  - golden answer accuracy = share of finished samples whose golden answer the
    auditor judged reasonable (``verdict.golden_answer_reasonable``);
  - answer_session_ids accuracy = share whose claimed answer sessions the auditor
    judged sufficient (``verdict.answer_session_ids_reasonable``);
  - answer-session date sanity (``verdict.answer_session_date_ok``).

Everything is also broken down by ``question_type``. Use ``--list-bad`` to print
the samples whose golden answer was judged NOT reasonable.

Examples:
    python benchmark/longmemeval/stats_check_golden.py
    python benchmark/longmemeval/stats_check_golden.py --list-bad
    python benchmark/longmemeval/stats_check_golden.py --list-run-failed
    python benchmark/longmemeval/stats_check_golden.py --json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "datasets" / "longmemeval"
LOGDIR = REPO / "logs" / "check_golden"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--list-bad", action="store_true", help="list samples whose golden answer is NOT reasonable")
    p.add_argument(
        "--list-bad-sessions",
        action="store_true",
        help="list samples whose answer_session_ids is NOT reasonable",
    )
    p.add_argument(
        "--list-run-failed",
        action="store_true",
        help="list launched samples that did not produce readable output",
    )
    p.add_argument("--json", action="store_true", help="emit the summary as JSON")
    return p.parse_args()


def sample_ids() -> list[str]:
    """List all sample IDs."""
    ids = [p.name for p in DATA.iterdir() if p.is_dir() and p.name.isdigit()]
    return sorted(ids, key=int)


def pct(num: int, den: int) -> str:
    """Format a percentage."""
    return f"{(100.0 * num / den):.1f}%" if den else "n/a"


def logged_sample_ids() -> list[str]:
    """List all sample IDs that have been launched but not finished."""
    if not LOGDIR.exists():
        return []
    ids = [p.stem for p in LOGDIR.glob("*.log") if p.stem.isdigit()]
    return sorted(ids, key=int)


def main() -> int:
    """Main entry point."""
    args = parse_args()
    ids = sample_ids()
    total = len(ids)

    done, unreadable = [], []
    finished_ids = set()
    for idx in ids:
        path = DATA / idx / "check_golden.json"
        if not path.exists():
            continue
        try:
            with path.open(encoding="utf-8") as f:
                data = json.load(f)
            data["_idx"] = idx
            done.append(data)
            finished_ids.add(idx)
        except (OSError, json.JSONDecodeError):
            unreadable.append(idx)

    n = len(done)
    launched = logged_sample_ids()
    run_failed = [idx for idx in launched if idx not in finished_ids]

    # Overall tallies.
    golden_ok = sum(1 for d in done if d.get("verdict", {}).get("golden_answer_reasonable") is True)
    sess_ok = sum(1 for d in done if d.get("verdict", {}).get("answer_session_ids_reasonable") is True)
    date_ok = sum(1 for d in done if d.get("verdict", {}).get("answer_session_date_ok") is True)
    confs = [
        d["verdict"]["confidence"] for d in done if isinstance(d.get("verdict", {}).get("confidence"), (int, float))
    ]
    avg_conf = sum(confs) / len(confs) if confs else 0.0

    # Per question_type breakdown.
    by_type: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "golden_ok": 0, "sess_ok": 0})
    for d in done:
        v = d.get("verdict", {})
        t = d.get("question_type") or "(unknown)"
        by_type[t]["n"] += 1
        by_type[t]["golden_ok"] += 1 if v.get("golden_answer_reasonable") is True else 0
        by_type[t]["sess_ok"] += 1 if v.get("answer_session_ids_reasonable") is True else 0

    bad_golden = [d["_idx"] for d in done if d.get("verdict", {}).get("golden_answer_reasonable") is not True]
    bad_sessions = [d["_idx"] for d in done if d.get("verdict", {}).get("answer_session_ids_reasonable") is not True]

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
                    "golden_answer_accuracy": round(golden_ok / n, 4) if n else None,
                    "answer_session_ids_accuracy": round(sess_ok / n, 4) if n else None,
                    "answer_session_date_ok_rate": round(date_ok / n, 4) if n else None,
                    "avg_confidence": round(avg_conf, 4),
                    "golden_ok": golden_ok,
                    "sess_ok": sess_ok,
                    "date_ok": date_ok,
                    "by_type": {t: {**c, "golden_acc": round(c["golden_ok"] / c["n"], 4)} for t, c in by_type.items()},
                    "bad_golden": bad_golden,
                    "bad_sessions": bad_sessions,
                },
                ensure_ascii=False,
                indent=2,
            ),
        )
        return 0

    print("=" * 60)
    print("LongMemEval check_golden 统计")
    print("=" * 60)
    print(f"样例总数            : {total}")
    print(f"已完成 (有产出)     : {n}  ({pct(n, total)})")
    print(f"未完成              : {total - n - len(unreadable)}")
    if unreadable:
        print(f"损坏/无法解析       : {len(unreadable)}  {unreadable}")
    print(f"已启动过 (有 log)   : {len(launched)}")
    print(f"运行失败/无可读产出 : {len(run_failed)}")
    print("-" * 60)
    print(f"golden answer 准确率 : {pct(golden_ok, n)}   ({golden_ok}/{n} 判为合理)")
    print(f"answer_session 合理率: {pct(sess_ok, n)}   ({sess_ok}/{n})")
    print(f"answer 日期一致率    : {pct(date_ok, n)}   ({date_ok}/{n})")
    print(f"平均置信度          : {avg_conf:.3f}")
    print("-" * 60)
    print("按 question_type:")
    print(f"  {'type':<24} {'n':>4} {'golden准确率':>14} {'session合理率':>14}")
    for t in sorted(by_type):
        c = by_type[t]
        print(f"  {t:<24} {c['n']:>4} {pct(c['golden_ok'], c['n']):>14} {pct(c['sess_ok'], c['n']):>14}")

    if args.list_bad:
        print("-" * 60)
        print(f"golden answer 判为不合理的样例 ({len(bad_golden)}): {bad_golden}")
    if args.list_bad_sessions:
        print("-" * 60)
        print(f"answer_session_ids 判为不合理的样例 ({len(bad_sessions)}): {bad_sessions}")
    if args.list_run_failed:
        print("-" * 60)
        print(f"运行失败/无可读 check_golden.json 的样例 ({len(run_failed)}): {run_failed}")
        for idx in run_failed:
            print(f"  {idx}: {LOGDIR / f'{idx}.log'}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
