#!/usr/bin/env python3
"""Summarise the ``check_golden.json`` verdicts across all LongMemEval samples.

Reports progress (how many of the 500 samples have finished) and accuracy:
  - golden answer accuracy = share of finished samples whose golden answer the
    auditor judged correct (``verdict.golden_answer_correct``);
  - answer_session_ids accuracy = share whose claimed answer sessions the auditor
    judged exactly correct (``verdict.answer_session_ids_correct``).

Everything is also broken down by ``question_type``. Use ``--list-bad`` to print
the samples whose golden answer was judged NOT correct.

Examples:
    python benchmark/longmemeval/stats_golden_check.py
    python benchmark/longmemeval/stats_golden_check.py --list-bad
    python benchmark/longmemeval/stats_golden_check.py --list-run-failed
    python benchmark/longmemeval/stats_golden_check.py --json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "datasets" / "longmemeval"
LOGDIR = REPO / "logs" / "golden_check"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--list-bad", action="store_true", help="list samples whose golden answer is NOT correct")
    p.add_argument(
        "--list-bad-sessions",
        action="store_true",
        help="list samples whose answer_session_ids is NOT correct",
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


def load_json(path: Path) -> dict:
    """Load a JSON object, returning {} on any error."""
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def question_type_for(idx: str, data: dict) -> str:
    """Return question_type from the output, session review, or query.json."""
    question_type = str(data.get("question_type") or "").strip()
    if question_type:
        return question_type

    review_path_raw = str(data.get("session_review_path") or "").strip()
    review_path = Path(review_path_raw) if review_path_raw else DATA / idx / "session_review.json"
    if not review_path.is_absolute():
        review_path = REPO / review_path
    review = load_json(review_path)
    review_question_type = str((review.get("query") or {}).get("question_type") or "").strip()
    if review_question_type:
        return review_question_type

    query = load_json(DATA / idx / "query.json")
    return str(query.get("question_type") or "(unknown)").strip() or "(unknown)"


def verdict_bool(verdict: dict, new_key: str, old_key: str) -> bool:
    """Read a verdict boolean, accepting the old field name for compatibility."""
    if verdict.get(new_key) is True:
        return True
    if verdict.get(new_key) is False:
        return False
    return verdict.get(old_key) is True


def has_current_verdict(data: dict) -> bool:
    """Return True when ``check_golden.json`` uses the current golden_check schema."""
    verdict = data.get("verdict") if isinstance(data, dict) else None
    if not isinstance(verdict, dict):
        return False
    return isinstance(verdict.get("golden_answer_correct"), bool) and isinstance(
        verdict.get("answer_session_ids_correct"),
        bool,
    )


def write_golden_check_list(done: list[dict], output_path: Path) -> None:
    """Write all readable check_golden records as JSONL."""
    with output_path.open("w", encoding="utf-8") as f:
        for data in done:
            f.write(json.dumps(data, ensure_ascii=False))
            f.write("\n")


def main() -> int:
    """Main entry point."""
    args = parse_args()
    ids = sample_ids()
    total = len(ids)

    done, unreadable, stale = [], [], []
    finished_ids = set()
    for idx in ids:
        path = DATA / idx / "check_golden.json"
        if not path.exists():
            continue
        try:
            with path.open(encoding="utf-8") as f:
                data = json.load(f)
            if not has_current_verdict(data):
                stale.append(idx)
                continue
            data["_idx"] = idx
            data["_question_type"] = question_type_for(idx, data)
            done.append(data)
            finished_ids.add(idx)
        except (OSError, json.JSONDecodeError):
            unreadable.append(idx)

    n = len(done)
    output_path = Path.cwd() / "golden_check_list.jsonl"
    write_golden_check_list(done, output_path)
    launched = logged_sample_ids()
    run_failed = [idx for idx in launched if idx not in finished_ids]

    # Overall tallies.
    golden_ok = sum(
        1 for d in done if verdict_bool(d.get("verdict", {}), "golden_answer_correct", "golden_answer_reasonable")
    )
    sess_ok = sum(
        1
        for d in done
        if verdict_bool(d.get("verdict", {}), "answer_session_ids_correct", "answer_session_ids_reasonable")
    )
    both_ok = sum(
        1
        for d in done
        if verdict_bool(d.get("verdict", {}), "golden_answer_correct", "golden_answer_reasonable")
        and verdict_bool(d.get("verdict", {}), "answer_session_ids_correct", "answer_session_ids_reasonable")
    )

    # Per question_type breakdown.
    by_type: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "golden_ok": 0, "sess_ok": 0, "both_ok": 0})
    for d in done:
        v = d.get("verdict", {})
        golden_is_ok = verdict_bool(v, "golden_answer_correct", "golden_answer_reasonable")
        sess_is_ok = verdict_bool(v, "answer_session_ids_correct", "answer_session_ids_reasonable")
        t = d.get("_question_type") or "(unknown)"
        by_type[t]["n"] += 1
        by_type[t]["golden_ok"] += 1 if golden_is_ok else 0
        by_type[t]["sess_ok"] += 1 if sess_is_ok else 0
        by_type[t]["both_ok"] += 1 if golden_is_ok and sess_is_ok else 0

    bad_golden = [
        d["_idx"]
        for d in done
        if not verdict_bool(d.get("verdict", {}), "golden_answer_correct", "golden_answer_reasonable")
    ]
    bad_sessions = [
        d["_idx"]
        for d in done
        if not verdict_bool(d.get("verdict", {}), "answer_session_ids_correct", "answer_session_ids_reasonable")
    ]

    if args.json:
        print(
            json.dumps(
                {
                    "total": total,
                    "finished": n,
                    "pending": total - n - len(unreadable),
                    "unreadable": unreadable,
                    "stale": stale,
                    "launched": len(launched),
                    "run_failed": run_failed,
                    "golden_answer_accuracy": round(golden_ok / n, 4) if n else None,
                    "answer_session_ids_accuracy": round(sess_ok / n, 4) if n else None,
                    "both_correct_rate": round(both_ok / n, 4) if n else None,
                    "golden_ok": golden_ok,
                    "sess_ok": sess_ok,
                    "both_ok": both_ok,
                    "by_type": {
                        t: {
                            **c,
                            "golden_acc": round(c["golden_ok"] / c["n"], 4),
                            "session_acc": round(c["sess_ok"] / c["n"], 4),
                            "both_acc": round(c["both_ok"] / c["n"], 4),
                        }
                        for t, c in by_type.items()
                    },
                    "bad_golden": bad_golden,
                    "bad_sessions": bad_sessions,
                    "golden_check_list": str(output_path),
                },
                ensure_ascii=False,
                indent=2,
            ),
        )
        return 0

    print("=" * 60)
    print("LongMemEval golden_check 统计")
    print("=" * 60)
    print(f"样例总数            : {total}")
    print(f"已完成 (有产出)     : {n}  ({pct(n, total)})")
    print(f"未完成              : {total - n - len(unreadable)}")
    if unreadable:
        print(f"损坏/无法解析       : {len(unreadable)}  {unreadable}")
    if stale:
        print(f"旧格式待重跑        : {len(stale)}  {stale}")
    print(f"已合并 JSONL        : {output_path}")
    print(f"已启动过 (有 log)   : {len(launched)}")
    print(f"运行失败/无可读产出 : {len(run_failed)}")
    print("-" * 60)
    print(f"golden answer 正确率 : {pct(golden_ok, n)}   ({golden_ok}/{n})")
    print(f"answer_session 正确率: {pct(sess_ok, n)}   ({sess_ok}/{n})")
    print(f"两者都正确          : {pct(both_ok, n)}   ({both_ok}/{n})")
    print("-" * 60)
    print("按 question_type:")
    print(f"  {'type':<24} {'n':>4} {'golden正确率':>14} {'session正确率':>14} {'都正确':>10}")
    for t in sorted(by_type):
        c = by_type[t]
        print(
            f"  {t:<24} {c['n']:>4} {pct(c['golden_ok'], c['n']):>14} "
            f"{pct(c['sess_ok'], c['n']):>14} {pct(c['both_ok'], c['n']):>10}",
        )

    if args.list_bad:
        print("-" * 60)
        print(f"golden answer 判为不正确的样例 ({len(bad_golden)}): {bad_golden}")
    if args.list_bad_sessions:
        print("-" * 60)
        print(f"answer_session_ids 判为不正确的样例 ({len(bad_sessions)}): {bad_sessions}")
    if args.list_run_failed:
        print("-" * 60)
        print(f"运行失败/无可读 check_golden.json 的样例 ({len(run_failed)}): {run_failed}")
        for idx in run_failed:
            print(f"  {idx}: {LOGDIR / f'{idx}.log'}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
