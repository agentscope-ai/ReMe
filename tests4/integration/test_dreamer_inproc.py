"""dreamer in-process integration test.

Loads the default reme4 config, seeds a rich workspace via the shared
``vault_env`` fixture (pre-existing digest nodes spread across the three
buckets + a new daily that exercises CREATE and UPDATE in each bucket),
reindexes so search can hit the pre-existing nodes, then calls ``dream``
and prints what happened.

Phase 1 classifies each sub-unit into one of {procedure, personal,
wiki}; Phase 2 dispatches to the bucket-specific integrate prompt
and writes via the canonical ``write`` / ``edit`` tools.

Usage (from anywhere):
    python tests4/integration/test_dreamer_inproc.py
    python tests4/integration/test_dreamer_inproc.py \\
        daily/2026-05-28/auth-refactor/notes.md

Each run wipes ``daily/``, ``digest/``, ``resource/``, and
``reme_metadata/`` under a freshly-built throwaway vault before
reseeding, so the dreamer always starts from the same fixture state.
See ``_vault_fixture.py`` (``seed_dream_vault`` / ``DREAM_INPUT_PATH``)
for what gets created and the expected CREATE / UPDATE landings per
bucket.

Required env (from .env or shell):
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL_NAME    — for the Phase 1/2 agents
"""

import asyncio
import sys
from pathlib import Path

# Make ``_vault_fixture`` importable as a top-level module regardless of
# the caller's cwd.
INTEGRATION_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(INTEGRATION_DIR))

# pylint: disable=wrong-import-position
from _vault_fixture import DREAM_INPUT_PATH, vault_env  # noqa: E402


async def main() -> None:
    """Seed a vault, reindex it, run ``dream`` on the seeded daily note."""
    rel_input = sys.argv[1] if len(sys.argv) > 1 else DREAM_INPUT_PATH

    with vault_env() as env:
        seeded = env.seed_dream_vault()
        print(f"--- seeded {len(seeded)} fixture file(s) under {env.vault_dir}")

        app = await env.make_reme()
        print(f"--- vault_dir: {env.vault_dir}")
        print(f"--- input:     {rel_input}")

        try:
            # Reindex first so search can actually find the pre-seeded
            # digest/ nodes — otherwise Phase 2 recall returns empty and
            # every sub-unit ends up as CREATE (UPDATE path not exercised).
            print("\n--- reindexing vault so Phase 2 recall has something to hit")
            await app.run_job("reindex")

            print(f"\n--- running dream path={rel_input}")
            with env.record_agents(prefix="dream") as recorder:
                resp = await app.run_job("dream", path=rel_input)
            dumped = await recorder.dump()
            print(f"\n--- dumped {len(dumped)} agent memory file(s) to {recorder.dump_dir}")
            for p in dumped:
                print(f"  {p.relative_to(recorder.dump_dir)}")

            print("\n=== Response.success ===")
            print(resp.success)
            print("\n=== Response.answer ===")
            print(resp.answer)
            print("\n=== Response.metadata (DreamResult fields) ===")
            for k, v in (resp.metadata or {}).items():
                if isinstance(v, list) and len(v) > 8:
                    print(f"  {k}: list({len(v)} items) head={v[:3]!r}")
                else:
                    print(f"  {k}: {v!r}")
        finally:
            await env.close_all()

        print("\n=== digest/ tree after dream ===")
        digest_files = env.digest_files()
        if not digest_files:
            print("  (no digest files)")
        for p in digest_files:
            print(f"\n--- {p.relative_to(env.vault_dir)} ---")
            # print(p.read_text(encoding="utf-8"))


if __name__ == "__main__":
    asyncio.run(main())
