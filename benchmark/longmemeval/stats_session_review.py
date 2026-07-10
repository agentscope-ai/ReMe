#!/usr/bin/env python3
"""Summarise LongMemEval ``session_review.json`` artifacts.

This script is for upstream health checks before running ``golden_check``.
Samples with retryable per-session failures should be rerun as a whole; samples
with non-retryable fallback reviews are reported separately.

Examples:
    python benchmark/longmemeval/stats_session_review.py
    python benchmark/longmemeval/stats_session_review.py --list-failed
    python benchmark/longmemeval/stats_session_review.py --list-fallback
    python benchmark/longmemeval/stats_session_review.py --json
"""

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "datasets" / "longmemeval"
LOGDIR = REPO / "logs" / "session_review"
OUTPUT_FILENAME = "session_review.json"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list-failed", action="store_true", help="list samples with retryable failed per-session reviews")
    p.add_argument("--list-fallback", action="store_true", help="list non-retryable fallback reviews")
    p.add_argument("--list-missing", action="store_true", help="list samples missing session_review.json")
    p.add_argument("--list-run-failed", action="store_true", help="list launched samples without a healthy output")
    p.add_argument("--json", action="store_true", help="emit the summary as JSON")
    return p.parse_args()


def sample_ids() -> list[str]:
    """List all numeric sample IDs."""
    ids = [p.name for p in DATA.iterdir() if p.is_dir() and p.name.isdigit()]
    return sorted(ids, key=int)


def pct(num: int, den: int) -> str:
    """Format a percentage."""
    return f"{(100.0 * num / den):.1f}%" if den else "n/a"


def load_json(path: Path) -> dict:
    """Load a JSON object, returning {} on any error."""
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def logged_sample_ids() -> list[str]:
    """List sample IDs that have a session_review runner log."""
    if not LOGDIR.exists():
        return []
    ids = [p.stem for p in LOGDIR.glob("*.log") if p.stem.isdigit()]
    return sorted(ids, key=int)


def review_block(data: dict) -> dict:
    """Return the review block when present."""
    review = data.get("review") if isinstance(data, dict) else None
    return review if isinstance(review, dict) else {}


def failure_details(data: dict) -> list[dict]:
    """Return retryable failed_reviews when present."""
    failed_reviews = review_block(data).get("failed_reviews")
    if not isinstance(failed_reviews, list):
        return []
    return [item for item in failed_reviews if isinstance(item, dict) and not item.get("fallback")]


def fallback_details(data: dict) -> list[dict]:
    """Return non-retryable fallback review details when present."""
    review = review_block(data)
    fallback_reviews = review.get("fallback_reviews")
    if isinstance(fallback_reviews, list):
        return [item for item in fallback_reviews if isinstance(item, dict)]

    failed_reviews = review.get("failed_reviews")
    if isinstance(failed_reviews, list):
        return [item for item in failed_reviews if isinstance(item, dict) and item.get("fallback")]
    return []


def failure_count(data: dict) -> int:
    """Return retryable failed review count."""
    review = review_block(data)
    raw = review.get("num_failed_reviews")
    raw_fallback = review.get("num_fallback_reviews")
    if isinstance(raw, int) and isinstance(raw_fallback, int):
        return max(0, raw - raw_fallback)
    return len(failure_details(data))


def fallback_count(data: dict) -> int:
    """Return non-retryable fallback review count."""
    review = review_block(data)
    raw = review.get("num_fallback_reviews")
    if isinstance(raw, int):
        return raw
    return len(fallback_details(data))


def main() -> int:
    """Main entry point."""
    args = parse_args()
    ids = sample_ids()
    total = len(ids)

    healthy, failed, fallback, missing, unreadable = [], [], [], [], []
    total_failed_sessions = 0
    total_fallback_sessions = 0
    failed_details_by_id: dict[str, list[dict]] = {}
    fallback_details_by_id: dict[str, list[dict]] = {}

    for idx in ids:
        path = DATA / idx / OUTPUT_FILENAME
        if not path.exists():
            missing.append(idx)
            continue
        data = load_json(path)
        if not data:
            unreadable.append(idx)
            continue
        n_failed = failure_count(data)
        n_fallback = fallback_count(data)
        if n_failed:
            failed.append(idx)
            total_failed_sessions += n_failed
            failed_details_by_id[idx] = failure_details(data)
        if n_fallback:
            fallback.append(idx)
            total_fallback_sessions += n_fallback
            fallback_details_by_id[idx] = fallback_details(data)
        if not n_failed:
            healthy.append(idx)

    launched = logged_sample_ids()
    healthy_set = set(healthy)
    run_failed = [idx for idx in launched if idx not in healthy_set]

    if args.json:
        print(
            json.dumps(
                {
                    "total": total,
                    "healthy": len(healthy),
                    "failed_samples": failed,
                    "failed_sample_count": len(failed),
                    "failed_session_count": total_failed_sessions,
                    "fallback_samples": fallback,
                    "fallback_sample_count": len(fallback),
                    "fallback_session_count": total_fallback_sessions,
                    "missing": missing,
                    "unreadable": unreadable,
                    "launched": len(launched),
                    "run_failed_or_unhealthy": run_failed,
                    "failed_details": failed_details_by_id,
                    "fallback_details": fallback_details_by_id,
                },
                ensure_ascii=False,
                indent=2,
            ),
        )
        return 0

    print("=" * 60)
    print("LongMemEval session_review 统计")
    print("=" * 60)
    print(f"样例总数              : {total}")
    print(f"可继续产出            : {len(healthy)}  ({pct(len(healthy), total)})")
    print(f"有可重试失败          : {len(failed)}")
    print(f"可重试失败 session    : {total_failed_sessions}")
    print(f"有不可重试 fallback   : {len(fallback)}")
    print(f"fallback session      : {total_fallback_sessions}")
    print(f"缺少 session_review   : {len(missing)}")
    print(f"损坏/无法解析         : {len(unreadable)}")
    print(f"已启动过 (有 log)     : {len(launched)}")
    print(f"运行失败/非健康产出   : {len(run_failed)}")
    print("-" * 60)
    print("有可重试 failed_reviews 的样例需要整体重跑：")
    if failed:
        print(" ".join(failed))
        print("重跑命令示例：")
        print(f"python benchmark/longmemeval/run_session_review.py --start {failed[0]} --end {failed[0]}")
    else:
        print("(none)")
    if fallback:
        print("-" * 60)
        print("不可重试 fallback 的样例不用重跑：")
        print(" ".join(fallback))

    if args.list_failed and failed:
        print("-" * 60)
        for idx in failed:
            details = failed_details_by_id.get(idx) or []
            print(f"{idx}: {DATA / idx / OUTPUT_FILENAME} failed_sessions={len(details)}")
            for item in details:
                session_id = item.get("session_id", "(unknown)")
                error = str(item.get("error") or "").replace("\n", " ")
                print(f"  - {session_id}: {error}")
    if args.list_fallback and fallback:
        print("-" * 60)
        for idx in fallback:
            details = fallback_details_by_id.get(idx) or []
            print(f"{idx}: {DATA / idx / OUTPUT_FILENAME} fallback_sessions={len(details)}")
            for item in details:
                session_id = item.get("session_id", "(unknown)")
                reason = str(item.get("fallback_reason") or "fallback")
                error = str(item.get("error") or "").replace("\n", " ")
                raw_saved = "yes" if item.get("raw_session") else "no"
                print(f"  - {session_id}: reason={reason} raw_session_saved={raw_saved} error={error}")
    if args.list_missing and missing:
        print("-" * 60)
        print(f"缺少 session_review.json 的样例 ({len(missing)}): {missing}")
    if args.list_run_failed and run_failed:
        print("-" * 60)
        print(f"运行失败/非健康产出的样例 ({len(run_failed)}): {run_failed}")
        for idx in run_failed:
            print(f"  {idx}: {LOGDIR / f'{idx}.log'}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
