"""CLI runner: `python -m reme2.mcp.test`

Boots one Application per profile against a fresh temp vault, runs the
profile's CHECKS list in order, prints a per-check status line plus a
final summary. Exit code is the number of failed checks across both
profiles.

Per-profile suites are intentionally sequential — later checks rely on
side-effects from earlier ones (e.g. `sync.append` after `sync.create`).
Cross-profile work happens in independent vaults.
"""

from __future__ import annotations

import asyncio
import shutil
import sys
import traceback

from . import test_expert, test_service
from ._helpers import AppContext, make_context


SUITES = {
    "expert":  test_expert.CHECKS,
    "service": test_service.CHECKS,
}


async def _run_suite(profile: str, checks: list[tuple[str, callable]]) -> tuple[int, int]:
    """Run one suite. Returns (passed, total)."""
    print(f"\n========== profile: {profile} ==========")
    ctx: AppContext | None = None
    tmp = None
    passed = 0
    try:
        ctx, tmp = await make_context(profile)
        print(f"  bootstrapped: vault={ctx.vault}, jobs={len(ctx.jobs)}, "
              f"file_store={len(ctx.file_store)}")
        for label, fn in checks:
            try:
                summary = await fn(ctx)
                print(f"  ✓ {label:32s} {summary}")
                passed += 1
            except Exception as e:
                print(f"  ✗ {label:32s} {type(e).__name__}: {e}")
                traceback.print_exc()
    except Exception as e:
        print(f"  ✗ bootstrap failed: {type(e).__name__}: {e}")
        traceback.print_exc()
    finally:
        if ctx is not None:
            try:
                await ctx.app.close()
            except Exception as e:  # noqa: BLE001
                print(f"  (warn) app.close failed: {e}")
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)
    return passed, len(checks)


async def _main() -> int:
    summary: dict[str, tuple[int, int]] = {}
    for profile, checks in SUITES.items():
        summary[profile] = await _run_suite(profile, checks)

    print("\n========== summary ==========")
    failed = 0
    for profile, (passed, total) in summary.items():
        marker = "✓" if passed == total else "✗"
        print(f"  {marker} {profile}: {passed}/{total}")
        failed += (total - passed)
    return 0 if failed == 0 else failed


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
