"""Integration test for the auto_resource job.

Drives the ``auto_resource`` step against a real LLM. Three scenarios:

1. **CREATE (added)** / **UPDATE (modified)**: places a resource file in
   ``resource/{date}/``, calls ``auto_resource`` with a ``changes`` batch,
   and expects the agent to write/update the same-name daily note.

2. **DELETE (deleted)**: seeds a resource note under
   ``daily/{date}/{resource_stem}.md``, calls ``auto_resource`` with a
   deleted change, and expects the note file to be removed.

Requires LLM_API_KEY (and optionally LLM_BASE_URL / LLM_MODEL_NAME) in the
environment or a .env file at the repo root. Hits the real LLM API.
"""

import asyncio
import sys
from pathlib import Path

INTEGRATION_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(INTEGRATION_DIR))

# pylint: disable=wrong-import-position
from _workspace_fixture import workspace_env  # noqa: E402

from reme.steps.evolve.auto_resource import _compute_agent_session_id, _compute_note_stem  # noqa: E402

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


def _print_text_file(label: str, path: Path) -> str:
    text = _read_text(path)
    print("\n" + "=" * 70)
    print(f"[{label}] {path} ({len(text)} bytes)")
    print(f"[{label}] body:\n{text}")
    print("=" * 70)
    return text


def _print_message_files(label: str, paths: list[Path]) -> None:
    for idx, path in enumerate(paths, 1):
        if not path.is_file():
            print(f"[{label}] message file missing: {path}")
            continue
        text = _read_text(path)
        print("\n" + "=" * 70)
        print(f"[{label}] message {idx}: {path} ({len(text)} bytes)")
        print(text)
        print("=" * 70)


def test_auto_resource_create():
    """CREATE branch: agent writes the same-name daily note and saves its AgentScope session."""

    async def run():
        with workspace_env() as env:
            app = await env.make_app()
            try:
                today = env.today

                print("\n" + "=" * 70)
                print("[setup] workspace_root =", env.workspace_dir)
                print("[setup] today      =", today)
                print("=" * 70)

                file_path = env.place_resource(RESOURCE_FILENAME, RESOURCE_CONTENT_V1)
                note_stem = _compute_note_stem(RESOURCE_FILENAME)
                agent_session_id = _compute_agent_session_id(file_path)
                expected_session_jsonl = env.workspace_dir / "session" / "agentscope" / f"{agent_session_id}.jsonl"

                print(f"[CREATE] file_path           = {file_path}")
                print(f"[CREATE] note_stem           = {note_stem}")
                print(f"[CREATE] agent_session_id    = {agent_session_id}")
                print(f"[CREATE] expected transcript = {expected_session_jsonl.relative_to(env.workspace_dir)}")

                with env.record_agents(prefix="agent_resource_create") as recorder:
                    response = await app.run_job(
                        "auto_resource",
                        changes=[{"path": file_path, "change": "added"}],
                    )
                dumped = await recorder.dump()
                for p in dumped:
                    print(f"[CREATE] agent memory dumped: {p}")

                assert response.success is True, f"CREATE job failed: {response.answer!r}"
                meta = response.metadata or {}
                result_meta = (meta.get("results") or [{}])[0].get("metadata") or {}
                assert result_meta.get("action") == "added", f"Unexpected action: {meta!r}"
                assert result_meta.get("session_id") == note_stem, f"Unexpected session_id: {meta!r}"
                assert result_meta.get("path") == f"daily/{today}/{note_stem}.md", f"Unexpected note path: {meta!r}"
                note_path = env.workspace_dir / "daily" / today / f"{note_stem}.md"
                assert note_path.is_file()

                assert expected_session_jsonl.is_file(), (
                    f"agent session not persisted at {expected_session_jsonl}; "
                    f"AgentScope files: "
                    f"{[p.name for p in (env.workspace_dir / 'session' / 'agentscope').glob('*.jsonl')]}"
                )

                note_text = _print_text_file("CREATE result.md", note_path)
                _print_message_files("CREATE intermediate messages", [*dumped, expected_session_jsonl])

                note_hits = [
                    needle
                    for needle in ("v2.0", "July 15", "Alice", "Bob", "p99", "200ms", "Redis")
                    if needle in note_text
                ]
                print(f"[CREATE] landed note facts: {note_hits}")
                assert (
                    len(note_hits) >= 3
                ), f"CREATE note missed expected facts {note_hits!r}\n--- NOTE ---\n{note_text}"

                transcript = _read_text(expected_session_jsonl)
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
    """UPDATE branch: agent updates the same-name daily note and appends to its AgentScope session."""

    async def run():
        with workspace_env() as env:
            app = await env.make_app()
            try:
                today = env.today

                print("\n" + "=" * 70)
                print("[setup] workspace_root =", env.workspace_dir)
                print("[setup] today      =", today)
                print("=" * 70)

                # First run as "added" so the resource file exists and the
                # initial transcript lands.
                file_path = env.place_resource(RESOURCE_FILENAME, RESOURCE_CONTENT_V1)
                note_stem = _compute_note_stem(RESOURCE_FILENAME)
                agent_session_id = _compute_agent_session_id(file_path)
                session_jsonl = env.workspace_dir / "session" / "agentscope" / f"{agent_session_id}.jsonl"

                response = await app.run_job("auto_resource", changes=[{"path": file_path, "change": "added"}])
                assert response.success is True, f"Initial create failed: {response.answer!r}"
                assert session_jsonl.is_file(), "initial added run did not save the AgentScope session"
                size_before = session_jsonl.stat().st_size
                print(f"[UPDATE] transcript before modify ({size_before} bytes)")

                # Now update the resource file and call with "modified".
                env.place_resource(RESOURCE_FILENAME, RESOURCE_CONTENT_V2)

                with env.record_agents(prefix="agent_resource_update") as recorder:
                    response = await app.run_job(
                        "auto_resource",
                        changes=[{"path": file_path, "change": "modified"}],
                    )
                dumped = await recorder.dump()
                for p in dumped:
                    print(f"[UPDATE] agent memory dumped: {p}")

                assert response.success is True, f"UPDATE job failed: {response.answer!r}"
                meta = response.metadata or {}
                result_meta = (meta.get("results") or [{}])[0].get("metadata") or {}
                assert result_meta.get("action") == "modified", f"Unexpected action: {meta!r}"
                assert result_meta.get("session_id") == note_stem, f"Unexpected session_id: {meta!r}"
                assert result_meta.get("path") == f"daily/{today}/{note_stem}.md", f"Unexpected note path: {meta!r}"
                note_path = env.workspace_dir / "daily" / today / f"{note_stem}.md"
                assert note_path.is_file()

                size_after = session_jsonl.stat().st_size
                print(f"[UPDATE] transcript after modify  ({size_after} bytes)")
                assert size_after > size_before, (
                    f"transcript did not grow after modified run " f"({size_before} -> {size_after})"
                )

                note_text = _print_text_file("UPDATE result.md", note_path)
                _print_message_files("UPDATE intermediate messages", [*dumped, session_jsonl])

                note_hits = [
                    needle
                    for needle in ("July 20", "150ms", "Dave", "rate limiting", "resolved")
                    if needle in note_text
                ]
                print(f"[UPDATE] landed note facts: {note_hits}")
                assert (
                    len(note_hits) >= 2
                ), f"UPDATE note missed expected facts {note_hits!r}\n--- NOTE ---\n{note_text}"

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
        with workspace_env() as env:
            app = await env.make_app()
            try:
                today = env.today

                print("\n" + "=" * 70)
                print("[setup] workspace_root =", env.workspace_dir)
                print("[setup] today      =", today)
                print("=" * 70)

                note_stem = _compute_note_stem(RESOURCE_FILENAME)
                file_path = f"resource/{today}/{RESOURCE_FILENAME}"

                seed_body = "---\nname: test\ndescription: test note\n---\n\nSome content.\n"
                note_path = env.seed_daily_note(note_stem, seed_body)
                assert note_path.is_file()
                print(f"[DELETE] seeded note: {note_path}")

                response = await app.run_job(
                    "auto_resource",
                    changes=[{"path": file_path, "change": "deleted"}],
                )

                assert response.success is True, f"DELETE job failed: {response.answer!r}"
                meta = response.metadata or {}
                result_meta = (meta.get("results") or [{}])[0].get("metadata") or {}
                assert result_meta.get("action") == "deleted"
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
