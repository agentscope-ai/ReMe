"""Integration test for the auto_memory job (planner + writer end-to-end).

Drives the full ``auto_memory`` orchestrator against a real LLM. The scenario
seeds one existing daily note covering an ongoing event, then feeds a
10-message conversation that:

1. continues the existing event with new status / facts (expects an UPDATE
   that preserves the old facts and appends the new ones), and
2. introduces a brand-new topic (expects a CREATE under today's daily folder
   with a stem distinct from the seeded one).

Requires LLM_API_KEY (and optionally LLM_BASE_URL / LLM_MODEL_NAME) in the
environment or a .env file at the repo root. Hits the real Anthropic API.
"""

import asyncio
import json
import os
import tempfile
from datetime import date as _date
from pathlib import Path

from agentscope.agent import ReActAgent

from reme4 import Application
from reme4.config import resolve_app_config
from reme4.utils import load_env

load_env()

# Where agent.memory jsonl dumps land — same directory as this test file.
DUMP_DIR = Path(__file__).resolve().parent


SEED_STEM = "auth-middleware-rewrite"
SEED_BODY = """---
name: auth-middleware-rewrite
description: JWT auth middleware rewrite driven by legal/compliance requirements around session token storage
---

# 背景

- 项目：JWT auth middleware 重写，替换旧的 session middleware
- 动机：legal/compliance 要求，旧的 session token 存储方式不符合新合规要求
- 决策：采用 RS256 签名，密钥放在 KMS，refresh token 写 redis 集群
- 团队：Alice 主导，Bob 协助

# 时间线

- 2026-05-20 立项 kickoff
- 2026-05-23 设计评审通过

# 当下状态

- 进度：实现中
- 卡点：暂无
- 下一步：完成 refresh token 写入流程
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


def _seed_note(vault_root: Path, today: str) -> Path:
    """Write the existing auth-middleware-rewrite note under today's daily folder."""
    day_dir = vault_root / "daily" / today
    day_dir.mkdir(parents=True, exist_ok=True)
    path = day_dir / f"{SEED_STEM}.md"
    path.write_text(SEED_BODY, encoding="utf-8")
    return path


def _make_messages() -> list[dict]:
    """A 10-turn conversation: first half continues the auth event, second half opens a new topic."""
    return [
        {
            "name": "user",
            "role": "user",
            "content": "状态更新：PR #432（auth middleware rewrite）今天已经合并到 dev 分支，等待 staging 验收。",
        },
        {
            "name": "assistant",
            "role": "assistant",
            "content": "好的，已记录。staging 验收前要先跑回归测试吗？",
        },
        {
            "name": "user",
            "role": "user",
            "content": (
                "对。测试时发现 refresh token TTL 设 7d 在 redis 集群挂了——"
                "redis maxmemory-policy 默认 allkeys-lru，会随机驱逐 token，导致用户被强制登出。"
            ),
        },
        {
            "name": "assistant",
            "role": "assistant",
            "content": "理解，要切到 volatile-ttl 才能只驱逐带 TTL 的 key，对吧？",
        },
        {
            "name": "user",
            "role": "user",
            "content": (
                "对，下一步：周五 2026-05-29 前把 redis 配置改成 volatile-ttl 并重测，" "blocked 在 SRE @lihua 的排期。"
            ),
        },
        {
            "name": "user",
            "role": "user",
            "content": (
                "切个话题，最近在调 pytorch 分布式训练。结论：DDP 启动推荐用 torchrun，" "比 mp.spawn 稳很多。"
            ),
        },
        {
            "name": "assistant",
            "role": "assistant",
            "content": "是因为信号处理的原因吗？",
        },
        {
            "name": "user",
            "role": "user",
            "content": (
                "主要是 NCCL backend 初始化更干净。mp.spawn 在 4 卡以上偶尔会卡死握手；"
                "复现版本 pytorch 2.5.1 + nccl 2.21.5。"
            ),
        },
        {
            "name": "user",
            "role": "user",
            "content": (
                "另外 batch size 用 64*world_size，per-rank lr 用 linear scaling rule "
                "（lr = base_lr * world_size）。"
            ),
        },
        {
            "name": "user",
            "role": "user",
            "content": "这两件事都先记一下。",
        },
    ]


def _read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")


class _AgentMemoryRecorder:
    """Monkey-patches ReActAgent.__init__ to capture every agent created inside
    the ``with`` block, then dumps each agent's memory to a jsonl file in
    DUMP_DIR on exit. One file per agent: ``agent_memory_<idx>_<name>.jsonl``,
    one message per line as ``Msg.to_dict()``.
    """

    def __init__(self, dump_dir: Path, prefix: str = "agent_memory"):
        """init"""
        self.dump_dir = dump_dir
        self.prefix = prefix
        self.agents: list[ReActAgent] = []
        self._orig_init = None
        self.dumped_paths: list[Path] = []

    def __enter__(self):
        """Monkey-patch ReActAgent.__init__."""
        self._orig_init = ReActAgent.__init__
        agents = self.agents
        orig = self._orig_init

        def _capturing_init(agent_self, *args, **kwargs):
            orig(agent_self, *args, **kwargs)
            agents.append(agent_self)

        ReActAgent.__init__ = _capturing_init
        return self

    def __exit__(self, *exc):
        """Restore the original __init__ and dump all agent memories."""
        ReActAgent.__init__ = self._orig_init

    async def dump(self) -> list[Path]:
        """Dump all agent memories."""
        # Wipe any prior dumps from this prefix so reruns don't accumulate stale files.
        for stale in self.dump_dir.glob(f"{self.prefix}_*.jsonl"):
            stale.unlink()

        for idx, agent in enumerate(self.agents, 1):
            messages = await agent.memory.get_memory()
            name = getattr(agent, "name", "agent") or "agent"
            out_path = self.dump_dir / f"{self.prefix}_{idx:02d}_{name}.jsonl"
            with out_path.open("w", encoding="utf-8") as f:
                for msg in messages:
                    f.write(json.dumps(msg.to_dict(), ensure_ascii=False, default=str) + "\n")
            self.dumped_paths.append(out_path)
        return self.dumped_paths


def test_auto_memory_updates_existing_and_creates_new():
    """End-to-end: planner survey + UPDATE one seeded note + CREATE one new note."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            app = await _make_app()
            try:
                vault_root = Path(app.config.vault_dir).absolute()
                today = _today()
                seed_path = _seed_note(vault_root, today)
                seed_before = _read_text(seed_path)
                assert "legal/compliance" in seed_before

                day_dir = vault_root / "daily" / today
                files_before = {p.name for p in day_dir.glob("*.md")}
                assert files_before == {f"{SEED_STEM}.md"}

                messages = _make_messages()

                print("\n" + "=" * 70)
                print("[setup] vault_root =", vault_root)
                print("[setup] today      =", today)
                print("[setup] seed_path  =", seed_path)
                print(f"[setup] seed body ({len(seed_before)} bytes):\n{seed_before}")
                print(f"[setup] feeding {len(messages)} messages to auto_memory:")
                for i, m in enumerate(messages, 1):
                    print(f"  {i:2d}. [{m['role']}] {m['content']}")
                print("=" * 70)

                with _AgentMemoryRecorder(DUMP_DIR) as recorder:
                    response = await app.run_job("auto_memory", messages=messages)
                dumped = await recorder.dump()
                print(f"\n[dump] captured {len(recorder.agents)} agent(s); wrote {len(dumped)} jsonl file(s):")
                for p in dumped:
                    print(f"  - {p}")

                # --- response-level assertions -------------------------------
                assert response.success is True, f"job failed: {response.answer!r}"
                meta = response.metadata or {}
                memory_updates = meta.get("memory_updates") or []

                print("\n" + "=" * 70)
                print(f"[planner] planned {len(memory_updates)} update(s):")
                for u in memory_updates:
                    print(f"  - path: {u.get('path')}")
                    print(f"    description: {u.get('description')}")
                print(f"\n[writer] written_count = {meta.get('written_count')}")
                print(f"[writer] answer:\n{response.answer}")
                print("=" * 70)

                assert (
                    len(memory_updates) >= 2
                ), f"expected at least 2 planned updates (1 UPDATE + 1 CREATE), got {memory_updates}"

                # Every emitted path must live under today's daily folder.
                paths = [u["path"] for u in memory_updates]
                for p in paths:
                    assert p.startswith(f"daily/{today}/") and p.endswith(
                        ".md",
                    ), f"path {p!r} violates daily/<today>/<stem>.md shape"

                # --- UPDATE branch: seeded note ------------------------------
                update_path_str = f"daily/{today}/{SEED_STEM}.md"
                assert (
                    update_path_str in paths
                ), f"planner did not reuse the seeded path {update_path_str!r}; got {paths}"
                seed_after = _read_text(seed_path)

                print("\n" + "=" * 70)
                print(f"[UPDATE] {seed_path} ({len(seed_before)} → {len(seed_after)} bytes)")
                print(f"[UPDATE] body after:\n{seed_after}")
                print("=" * 70)

                # Old facts must survive (timeline append + facts merge-and-dedupe).
                for old_fact in ("legal/compliance", "RS256", "Alice"):
                    assert (
                        old_fact in seed_after
                    ), f"UPDATE dropped pre-existing fact {old_fact!r}\n--- AFTER ---\n{seed_after}"
                # At least some of the new facts from the conversation must land.
                new_hits = [
                    needle
                    for needle in ("PR #432", "432", "volatile-ttl", "maxmemory-policy", "2026-05-29")
                    if needle in seed_after
                ]
                print("[UPDATE] preserved old facts: ['legal/compliance', 'RS256', 'Alice']")
                print(f"[UPDATE] landed new facts: {new_hits}")
                assert (
                    len(new_hits) >= 2
                ), f"UPDATE only landed {new_hits!r} of expected new facts\n--- AFTER ---\n{seed_after}"

                # --- CREATE branch: new topic --------------------------------
                files_after = {p.name for p in day_dir.glob("*.md")}
                new_files = files_after - files_before
                print(f"\n[CREATE] new files under daily/{today}/: {sorted(new_files)}")
                assert new_files, f"no new note created under daily/{today}/; planner paths: {paths}"
                # Find the file that actually covers the pytorch topic.
                pytorch_path: Path | None = None
                for fname in new_files:
                    text = _read_text(day_dir / fname)
                    if any(kw in text for kw in ("torchrun", "NCCL", "pytorch", "mp.spawn")):
                        pytorch_path = day_dir / fname
                        break
                assert pytorch_path is not None, f"no created note covers the pytorch topic; new files: {new_files}"
                pytorch_text = _read_text(pytorch_path)

                print("=" * 70)
                print(f"[CREATE] {pytorch_path} ({len(pytorch_text)} bytes)")
                print(f"[CREATE] body:\n{pytorch_text}")
                print("=" * 70)

                topic_hits = [
                    needle
                    for needle in ("torchrun", "mp.spawn", "NCCL", "2.5.1", "linear scaling", "world_size")
                    if needle in pytorch_text
                ]
                print(f"[CREATE] landed topic facts: {topic_hits}")
                assert (
                    len(topic_hits) >= 3
                ), f"CREATE only captured {topic_hits!r} of expected new-topic facts\n--- CREATE ---\n{pytorch_text}"
                # frontmatter sanity: name should equal the file stem, description non-empty.
                stem = pytorch_path.stem
                assert (
                    f"name: {stem}" in pytorch_text
                ), f"frontmatter name does not match stem {stem!r}\n{pytorch_text[:400]}"
                print(f"[CREATE] frontmatter name matches stem {stem!r}")

                print("\n" + "=" * 70)
                print("✓ test_auto_memory_updates_existing_and_creates_new passed")
                print("=" * 70)
            finally:
                await app.close()

    asyncio.run(run())


if __name__ == "__main__":
    print("=== auto_memory integration test ===")
    test_auto_memory_updates_existing_and_creates_new()
    print("\nAll integration tests passed!")
