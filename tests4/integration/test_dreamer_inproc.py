"""dreamer in-process integration test.

Loads the default reme4 config, seeds a rich workspace (pre-existing
digest nodes spread across the three buckets + a new daily that
exercises CREATE and UPDATE in each bucket), reindexes so search can
hit the pre-existing nodes, then calls `dream` and prints what
happened.

Phase 1 classifies each sub-unit into one of {procedure, personal,
wiki}; Phase 2 dispatches to the bucket-specific integrate prompt
and writes via the canonical `write` / `edit` tools.

Usage (from anywhere):
    VAULT_PATH=tests4/integration/vault python tests4/integration/test_dreamer_inproc.py
    VAULT_PATH=tests4/integration/vault python tests4/integration/test_dreamer_inproc.py \\
        daily/2026-05-28/auth-refactor/notes.md

Defaults:
    VAULT_PATH unset → tests4/integration/vault
    Each run wipes `daily/`, `digest/`, and `reme_metadata/` under the
    vault before reseeding, so the dreamer always starts from the same
    fixture state. See _dreamer_fixture.py for what gets created and
    the expected CREATE / UPDATE landings per bucket.

Required env (from .env or shell):
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL_NAME    — for the Phase 1/2 agents
"""

import asyncio
import json
import os
import sys
from pathlib import Path

from agentscope.agent import Agent

# Make `reme4` importable regardless of the caller's cwd; and make the
# fixture module importable as a top-level name.
REPO_ROOT = Path(__file__).resolve().parents[2]
INTEGRATION_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(INTEGRATION_DIR))

# pylint: disable=wrong-import-position
from _dreamer_fixture import clean_vault, seed_vault, INPUT_PATH  # noqa: E402

VAULT = os.environ.get("VAULT_PATH", "tests4/integration/vault")


class _AgentMemoryRecorder:
    """Monkey-patches Agent.__init__ to capture every agent created inside
    the ``with`` block, then dumps each agent's context history to a jsonl
    file under ``<vault>/agent_logs/`` on dump().

    Used to inspect the actual ReAct trace of Phase 1 extract + Phase 2
    integrate (per sub-unit) — what tools were called in what order, what
    candidates were recalled, what the LLM decided.
    """

    def __init__(self, vault: Path, prefix: str = "dream"):
        self.dump_dir = vault / "agent_logs"
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
        """Dump all captured agents' context to <vault>/agent_logs/."""
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


async def main() -> None:
    """Main function for testing the ReMeFs CLI."""
    from reme4 import ReMe  # noqa: E402
    from reme4.config import resolve_app_config  # noqa: E402
    from reme4.utils import load_env  # noqa: E402

    os.chdir(REPO_ROOT)  # so load_env() picks up the repo's .env
    load_env()

    vault = Path(VAULT).resolve()
    vault.mkdir(parents=True, exist_ok=True)

    removed = clean_vault(vault)
    if removed:
        print(f"--- cleaned {len(removed)} dir(s) under {vault}: {', '.join(removed)}")
    seeded = seed_vault(vault)
    print(f"--- seeded {len(seeded)} fixture file(s) under {vault}")

    rel_input = sys.argv[1] if len(sys.argv) > 1 else INPUT_PATH

    cfg = resolve_app_config(vault_dir=str(vault))
    print(f"--- vault_dir: {cfg.get('vault_dir')}")
    print(f"--- input:     {rel_input}")

    app = ReMe(**cfg)
    await app.start()
    try:
        # Reindex first so search can actually find the pre-seeded
        # digest/ nodes — otherwise Phase 2 recall returns empty and
        # every sub-unit ends up as CREATE (UPDATE path not exercised).
        print("\n--- reindexing vault so Phase 2 recall has something to hit")
        await app.run_job("reindex")

        print(f"\n--- running dream path={rel_input}")
        with _AgentMemoryRecorder(vault, prefix="dream") as recorder:
            resp = await app.run_job("dream", path=rel_input)
        dumped = await recorder.dump()
        print(f"\n--- dumped {len(dumped)} agent memory file(s) to {recorder.dump_dir}")
        for p in dumped:
            print(f"  {p.relative_to(vault)}")

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
        await app.close()

    print("\n=== digest/ tree after dream ===")
    digest_root = vault / "digest"
    if not digest_root.exists():
        print("  (no digest/ created)")
        return
    files = sorted(digest_root.rglob("*.md"))
    if not files:
        print("  (digest/ is empty)")
    for p in files:
        print(f"\n--- {p.relative_to(vault)} ---")
        # print(p.read_text(encoding="utf-8"))


if __name__ == "__main__":
    asyncio.run(main())
