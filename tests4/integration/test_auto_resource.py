"""Integration test for the auto_resource job.

Drives the ``auto_resource`` step against a real LLM. Three scenarios:

1. **CREATE (added)** / **UPDATE (modified)**: places a resource file in
   ``resource/{date}/``, calls ``auto_resource`` with the matching change.
   The current AS-backed step only captures the agent's read+reason
   transcript to ``resource/{date}/session_state_{sid}.jsonl`` — daily-note
   writing is handled by whichever ``auto_memory_*`` step comes next in
   the chain (the CC variant fork-writes from session_id; the AS variant
   needs explicit messages and is wired separately). So this test asserts
   on the session_state landing + agent fact coverage, not on a daily
   note file.

2. **DELETE (deleted)**: seeds a resource note under
   ``daily/{date}/{sid}.md``, calls ``auto_resource`` with
   change="deleted".  Expects the note file to be removed (the step
   stamps ``path`` on its metadata only in this branch).

Requires LLM_API_KEY (and optionally LLM_BASE_URL / LLM_MODEL_NAME) in the
environment or a .env file at the repo root. Hits the real LLM API.
"""

import asyncio
import sys
from pathlib import Path

INTEGRATION_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(INTEGRATION_DIR))

# pylint: disable=wrong-import-position
from _vault_fixture import vault_env  # noqa: E402

from reme4.steps.evolve.auto_resource import _compute_session_id  # noqa: E402

RESOURCE_FILENAME = "project-roadmap.md"
RESOURCE_CONTENT_V1 = """\
# Project Roadmap 2026 Q3

## Goals
- Launch v2.0 API by July 15
- Migrate 80% of users to new auth system by August 1
- Reduce p99 latency to < 200ms

## Milestones
| Date       | Milestone              | Owner   |
|------------|------------------------|---------|
| 2026-07-01 | API beta release       | Alice   |
| 2026-07-15 | API GA                 | Alice   |
| 2026-08-01 | Auth migration done    | Bob     |
| 2026-08-15 | Performance target met | Charlie |

## Risks
- Auth migration blocked on legacy client deprecation (ETA: June 30)
- Performance target requires Redis cluster upgrade (budget approved)
"""

RESOURCE_CONTENT_V2 = """\
# Project Roadmap 2026 Q3 (Revised)

## Goals
- Launch v2.0 API by July 20 (delayed 5 days from original July 15)
- Migrate 80% of users to new auth system by August 1
- Reduce p99 latency to < 150ms (tightened from 200ms)

## Milestones
| Date       | Milestone              | Owner   |
|------------|------------------------|---------|
| 2026-07-05 | API beta release       | Alice   |
| 2026-07-20 | API GA                 | Alice   |
| 2026-08-01 | Auth migration done    | Bob     |
| 2026-08-15 | Performance target met | Charlie |
| 2026-08-20 | Post-launch review     | Dave    |

## Risks
- Auth migration blocked on legacy client deprecation (resolved June 28)
- Performance target requires Redis cluster upgrade (completed July 1)
- New risk: third-party OAuth provider rate limiting during migration
"""


def _read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def test_auto_resource_create():
    """CREATE branch: agent reads the resource file and saves session_state.

    Daily-note writing belongs to the auto_memory_* step that follows; this
    test only asserts that auto_resource_step ran the agent and persisted
    its transcript under ``resource/{date}/session_state_{sid}.jsonl``.
    """

    async def run():
        with vault_env() as env:
            app = await env.make_app()
            try:
                today = env.today

                print("\n" + "=" * 70)
                print("[setup] vault_root =", env.vault_dir)
                print("[setup] today      =", today)
                print("=" * 70)

                file_path = env.place_resource(RESOURCE_FILENAME, RESOURCE_CONTENT_V1)
                session_id = _compute_session_id(RESOURCE_FILENAME)
                expected_session_jsonl = env.vault_dir / "resource" / today / f"session_reme_{session_id}.jsonl"

                print(f"[CREATE] file_path           = {file_path}")
                print(f"[CREATE] session_id          = {session_id}")
                print(f"[CREATE] expected transcript = {expected_session_jsonl.relative_to(env.vault_dir)}")

                with env.record_agents(prefix="agent_resource_create") as recorder:
                    response = await app.run_job(
                        "auto_resource",
                        file_path=file_path,
                        change="added",
                    )
                dumped = await recorder.dump()
                for p in dumped:
                    print(f"[CREATE] agent memory dumped: {p}")

                assert response.success is True, f"CREATE job failed: {response.answer!r}"
                meta = response.metadata or {}
                assert meta.get("action") == "added", f"Unexpected action: {meta!r}"
                assert meta.get("session_id") == session_id, f"Unexpected session_id: {meta!r}"

                assert expected_session_jsonl.is_file(), (
                    f"agent session_state not persisted at {expected_session_jsonl}; "
                    f"session_state files under resource/: "
                    f"{[p.name for p in env.session_state_files(prefix='session_reme_')]}"
                )

                # Read the agent transcript and check it actually opened
                # the resource file (the file_path should show up in a
                # tool-call argument) so we know the step did its job.
                transcript = _read_text(expected_session_jsonl)
                print("\n" + "=" * 70)
                print(f"[CREATE] {expected_session_jsonl.name} ({len(transcript)} bytes)")
                topic_hits = [
                    needle
                    for needle in ("v2.0", "July 15", "Alice", "Bob", "p99", "200ms", "Redis", file_path)
                    if needle in transcript
                ]
                print(f"[CREATE] facts visible in transcript: {topic_hits}")
                assert topic_hits, (
                    "agent transcript shows no signal it actually read the resource file; "
                    f"transcript head:\n{transcript[:500]}"
                )

                print("\n" + "=" * 70)
                print("test_auto_resource_create passed")
                print("=" * 70)
            finally:
                await env.close_all()

    asyncio.run(run())


def test_auto_resource_update():
    """UPDATE branch: same contract as CREATE — only session_state changes."""

    async def run():
        with vault_env() as env:
            app = await env.make_app()
            try:
                today = env.today

                print("\n" + "=" * 70)
                print("[setup] vault_root =", env.vault_dir)
                print("[setup] today      =", today)
                print("=" * 70)

                # First run as "added" so the resource file exists and the
                # initial transcript lands.
                file_path = env.place_resource(RESOURCE_FILENAME, RESOURCE_CONTENT_V1)
                session_id = _compute_session_id(RESOURCE_FILENAME)
                session_jsonl = env.vault_dir / "resource" / today / f"session_reme_{session_id}.jsonl"

                response = await app.run_job("auto_resource", file_path=file_path, change="added")
                assert response.success is True, f"Initial create failed: {response.answer!r}"
                assert session_jsonl.is_file(), "initial added run did not save session_state"
                size_before = session_jsonl.stat().st_size
                print(f"[UPDATE] transcript before modify ({size_before} bytes)")

                # Now update the resource file and call with "modified".
                env.place_resource(RESOURCE_FILENAME, RESOURCE_CONTENT_V2)

                with env.record_agents(prefix="agent_resource_update") as recorder:
                    response = await app.run_job(
                        "auto_resource",
                        file_path=file_path,
                        change="modified",
                    )
                dumped = await recorder.dump()
                for p in dumped:
                    print(f"[UPDATE] agent memory dumped: {p}")

                assert response.success is True, f"UPDATE job failed: {response.answer!r}"
                meta = response.metadata or {}
                assert meta.get("action") == "modified", f"Unexpected action: {meta!r}"
                assert meta.get("session_id") == session_id, f"Unexpected session_id: {meta!r}"

                size_after = session_jsonl.stat().st_size
                print(f"[UPDATE] transcript after modify  ({size_after} bytes)")
                assert size_after > size_before, (
                    f"transcript did not grow after modified run " f"({size_before} -> {size_after})"
                )

                transcript = _read_text(session_jsonl)
                new_hits = [
                    needle
                    for needle in ("July 20", "150ms", "Dave", "rate limiting", "resolved")
                    if needle in transcript
                ]
                print(f"[UPDATE] V2 facts visible in transcript: {new_hits}")
                assert new_hits, "modified run added no V2 content to the transcript; " f"tail:\n{transcript[-800:]}"

                print("\n" + "=" * 70)
                print("test_auto_resource_update passed")
                print("=" * 70)
            finally:
                await env.close_all()

    asyncio.run(run())


def test_auto_resource_delete():
    """DELETE a resource note (change=deleted)."""

    async def run():
        with vault_env() as env:
            app = await env.make_app()
            try:
                today = env.today

                print("\n" + "=" * 70)
                print("[setup] vault_root =", env.vault_dir)
                print("[setup] today      =", today)
                print("=" * 70)

                session_id = _compute_session_id(RESOURCE_FILENAME)
                file_path = f"resource/{today}/{RESOURCE_FILENAME}"

                note_filename = session_id
                seed_body = "---\nname: test\ndescription: test note\n---\n\nSome content.\n"
                note_path = env.seed_daily_note(note_filename, seed_body)
                assert note_path.is_file()
                print(f"[DELETE] seeded note: {note_path}")

                response = await app.run_job(
                    "auto_resource",
                    file_path=file_path,
                    change="deleted",
                )

                assert response.success is True, f"DELETE job failed: {response.answer!r}"
                meta = response.metadata or {}
                assert meta.get("action") == "deleted"
                assert not note_path.is_file(), f"Note file still exists after delete: {note_path}"

                print(f"[DELETE] note removed: {note_path}")
                print("\n" + "=" * 70)
                print("test_auto_resource_delete passed")
                print("=" * 70)
            finally:
                await env.close_all()

    asyncio.run(run())


if __name__ == "__main__":
    print("=== auto_resource integration test ===")
    test_auto_resource_create()
    test_auto_resource_update()
    test_auto_resource_delete()
    print("\nAll integration tests passed!")
