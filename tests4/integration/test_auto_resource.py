"""Integration test for the auto_resource job.

Drives the ``auto_resource`` step against a real LLM. Three scenarios:

1. **CREATE (added)**: places a resource file in ``resource/{date}/``,
   calls ``auto_resource`` with change="added".  Expects a new note
   ``daily/{date}/resource_{hash}.md`` with key facts from the file.

2. **UPDATE (modified)**: seeds an existing resource note, updates the
   resource file, calls ``auto_resource`` with change="modified".
   Expects the note to reflect the updated content.

3. **DELETE (deleted)**: seeds a resource note, calls ``auto_resource``
   with change="deleted".  Expects the note file to be removed.

Requires LLM_API_KEY (and optionally LLM_BASE_URL / LLM_MODEL_NAME) in the
environment or a .env file at the repo root. Hits the real LLM API.
"""

import asyncio
import json
import os
import tempfile
from datetime import date as _date
from pathlib import Path

from agentscope.agent import Agent

from reme4 import Application
from reme4.config import resolve_app_config
from reme4.steps.evolve.auto_resource import _compute_session_id
from reme4.utils import load_env

load_env()

DUMP_DIR = Path(__file__).resolve().parent / "agent_logs"

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


def _today() -> str:
    return _date.today().isoformat()


class _temp_chdir:
    def __init__(self, path):
        self.path = path
        self._old = None

    def __enter__(self):
        self._old = os.getcwd()
        os.chdir(self.path)
        return self

    def __exit__(self, *exc):
        os.chdir(self._old)


async def _make_app() -> Application:
    cfg = resolve_app_config(log_to_console=False, log_to_file=False, enable_logo=False)
    app = Application(**cfg)
    await app.start()
    return app


def _place_resource(vault_root: Path, today: str, filename: str, content: str) -> str:
    """Write a resource file and return its vault-relative path."""
    resource_dir = vault_root / "resource" / today
    resource_dir.mkdir(parents=True, exist_ok=True)
    path = resource_dir / filename
    path.write_text(content, encoding="utf-8")
    return f"resource/{today}/{filename}"


def _seed_resource_note(vault_root: Path, today: str, session_id: str, body: str) -> Path:
    """Pre-seed a resource note in daily/{date}/."""
    day_dir = vault_root / "daily" / today
    day_dir.mkdir(parents=True, exist_ok=True)
    path = day_dir / f"{session_id}.md"
    path.write_text(body, encoding="utf-8")
    return path


def _read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")


class _AgentMemoryRecorder:
    """Monkey-patches Agent.__init__ to capture every agent created inside
    the ``with`` block, then dumps each agent's memory to a jsonl file.
    """

    def __init__(self, dump_dir: Path, prefix: str = "agent_memory"):
        self.dump_dir = dump_dir
        self.prefix = prefix
        self.agents: list[Agent] = []
        self._orig_init = None
        self.dumped_paths: list[Path] = []

    def __enter__(self):
        self._orig_init = Agent.__init__
        agents = self.agents
        orig = self._orig_init

        def _capturing_init(agent_self, *args, **kwargs):
            orig(agent_self, *args, **kwargs)
            agents.append(agent_self)

        Agent.__init__ = _capturing_init
        return self

    def __exit__(self, *exc):
        Agent.__init__ = self._orig_init

    async def dump(self) -> list[Path]:
        """Serialize captured agent transcripts to disk."""
        self.dump_dir.mkdir(parents=True, exist_ok=True)
        for stale in self.dump_dir.glob(f"{self.prefix}_*.jsonl"):
            stale.unlink()

        for idx, agent in enumerate(self.agents, 1):
            messages = agent.state.context
            name = getattr(agent, "name", "agent") or "agent"
            out_path = self.dump_dir / f"{self.prefix}_{idx:02d}_{name}.jsonl"
            with out_path.open("w", encoding="utf-8") as f:
                for msg in messages:
                    f.write(json.dumps(msg.model_dump(), ensure_ascii=False, default=str) + "\n")
            self.dumped_paths.append(out_path)
        return self.dumped_paths


def test_auto_resource_create():
    """CREATE a resource note from a new file (change=added)."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            app = await _make_app()
            try:
                vault_root = Path(app.config.vault_dir).absolute()
                today = _today()

                print("\n" + "=" * 70)
                print("[setup] vault_root =", vault_root)
                print("[setup] today      =", today)
                print("=" * 70)

                file_path = _place_resource(vault_root, today, RESOURCE_FILENAME, RESOURCE_CONTENT_V1)
                session_id = _compute_session_id(RESOURCE_FILENAME)
                # daily_create prepends "session_agent_" to the session_id
                expected_note_path = f"daily/{today}/session_agent_{session_id}.md"

                print(f"[CREATE] file_path    = {file_path}")
                print(f"[CREATE] session_id   = {session_id}")
                print(f"[CREATE] expected note= {expected_note_path}")

                with _AgentMemoryRecorder(DUMP_DIR, prefix="agent_resource_create") as recorder:
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
                assert meta.get("path") == expected_note_path, f"Unexpected path: {meta!r}"
                assert meta.get("action") == "added"

                note_path = vault_root / expected_note_path
                assert note_path.is_file(), f"Created note not found at {note_path}"

                note_text = _read_text(note_path)
                print("\n" + "=" * 70)
                print(f"[CREATE] {note_path} ({len(note_text)} bytes)")
                print(f"[CREATE] body:\n{note_text}")
                print("=" * 70)

                topic_hits = [
                    needle
                    for needle in ("v2.0", "July 15", "Alice", "Bob", "p99", "200ms", "Redis")
                    if needle in note_text
                ]
                print(f"[CREATE] landed topic facts: {topic_hits}")
                assert (
                    len(topic_hits) >= 3
                ), f"CREATE only captured {topic_hits!r} of expected facts\n--- CREATE ---\n{note_text}"

                print("\n" + "=" * 70)
                print("test_auto_resource_create passed")
                print("=" * 70)
            finally:
                await app.close()

    asyncio.run(run())


def test_auto_resource_update():
    """UPDATE an existing resource note (change=modified)."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            app = await _make_app()
            try:
                vault_root = Path(app.config.vault_dir).absolute()
                today = _today()

                print("\n" + "=" * 70)
                print("[setup] vault_root =", vault_root)
                print("[setup] today      =", today)
                print("=" * 70)

                # First create a note via "added"
                file_path = _place_resource(vault_root, today, RESOURCE_FILENAME, RESOURCE_CONTENT_V1)
                session_id = _compute_session_id(RESOURCE_FILENAME)

                response = await app.run_job("auto_resource", file_path=file_path, change="added")
                assert response.success is True, f"Initial create failed: {response.answer!r}"

                note_abs = vault_root / "daily" / today / f"session_agent_{session_id}.md"
                note_before = _read_text(note_abs)
                print(f"[UPDATE] note before update ({len(note_before)} bytes)")

                # Now update the resource file and call with "modified"
                _place_resource(vault_root, today, RESOURCE_FILENAME, RESOURCE_CONTENT_V2)

                with _AgentMemoryRecorder(DUMP_DIR, prefix="agent_resource_update") as recorder:
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
                assert meta.get("action") == "modified"

                note_after = _read_text(note_abs)
                print("\n" + "=" * 70)
                print(f"[UPDATE] {note_abs} ({len(note_before)} -> {len(note_after)} bytes)")
                print(f"[UPDATE] body after:\n{note_after}")
                print("=" * 70)

                new_hits = [
                    needle
                    for needle in ("July 20", "150ms", "Dave", "rate limiting", "resolved")
                    if needle in note_after
                ]
                print(f"[UPDATE] landed new facts: {new_hits}")
                assert (
                    len(new_hits) >= 2
                ), f"UPDATE only landed {new_hits!r} of expected new facts\n--- AFTER ---\n{note_after}"

                print("\n" + "=" * 70)
                print("test_auto_resource_update passed")
                print("=" * 70)
            finally:
                await app.close()

    asyncio.run(run())


def test_auto_resource_delete():
    """DELETE a resource note (change=deleted)."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            app = await _make_app()
            try:
                vault_root = Path(app.config.vault_dir).absolute()
                today = _today()

                print("\n" + "=" * 70)
                print("[setup] vault_root =", vault_root)
                print("[setup] today      =", today)
                print("=" * 70)

                session_id = _compute_session_id(RESOURCE_FILENAME)
                file_path = f"resource/{today}/{RESOURCE_FILENAME}"

                # Seed the note file (daily_create prepends "session_agent_")
                note_filename = f"session_agent_{session_id}"
                seed_body = "---\nname: test\ndescription: test note\n---\n\nSome content.\n"
                note_path = _seed_resource_note(vault_root, today, note_filename, seed_body)
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
                await app.close()

    asyncio.run(run())


if __name__ == "__main__":
    print("=== auto_resource integration test ===")
    test_auto_resource_create()
    test_auto_resource_update()
    test_auto_resource_delete()
    print("\nAll integration tests passed!")
