"""Integration test for the auto_memory job (single-step).

Drives the ``auto_memory`` step against a real LLM. Two scenarios:

1. **CREATE**: calls ``auto_memory`` with a fresh ``session_id`` and
   conversation messages.  Expects a new note with the key facts.

2. **UPDATE**: seeds an existing daily note, calls ``auto_memory`` with
   the same ``session_id`` and new conversation messages.  Expects the
   old facts to survive and new facts to land.

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
from reme4.utils import load_env

load_env()

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


def _auth_messages() -> list[dict]:
    """Messages continuing the auth middleware thread."""
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
    ]


def _pytorch_messages() -> list[dict]:
    """Messages about a brand-new pytorch topic."""
    return [
        {
            "name": "user",
            "role": "user",
            "content": ("最近在调 pytorch 分布式训练。结论：DDP 启动推荐用 torchrun，" "比 mp.spawn 稳很多。"),
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
            "content": "先记一下。",
        },
    ]


def _read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")


class _AgentMemoryRecorder:
    """Monkey-patches Agent.__init__ to capture every agent created inside
    the ``with`` block, then dumps each agent's memory to a jsonl file in
    DUMP_DIR on exit.
    """

    def __init__(self, dump_dir: Path, prefix: str = "agent_memory"):
        """init"""
        self.dump_dir = dump_dir
        self.prefix = prefix
        self.agents: list[Agent] = []
        self._orig_init = None
        self.dumped_paths: list[Path] = []

    def __enter__(self):
        """Monkey-patch Agent.__init__."""
        self._orig_init = Agent.__init__
        agents = self.agents
        orig = self._orig_init

        def _capturing_init(agent_self, *args, **kwargs):
            orig(agent_self, *args, **kwargs)
            agents.append(agent_self)

        Agent.__init__ = _capturing_init
        return self

    def __exit__(self, *exc):
        """Restore the original __init__."""
        Agent.__init__ = self._orig_init

    async def dump(self) -> list[Path]:
        """Dump all agent context histories."""
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


def test_auto_memory_create():
    """CREATE a new note from scratch with a fresh session_id."""

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

                pytorch_session_id = "pytorch-distributed-training"
                with _AgentMemoryRecorder(DUMP_DIR, prefix="agent_create") as recorder:
                    response = await app.run_job(
                        "auto_memory",
                        messages=_pytorch_messages(),
                        session_id=pytorch_session_id,
                    )
                dumped = await recorder.dump()
                for p in dumped:
                    print(f"[CREATE] agent memory dumped: {p}")

                assert response.success is True, f"CREATE job failed: {response.answer!r}"
                meta = response.metadata or {}
                assert meta.get("created") is True, f"Expected created=True, got {meta!r}"
                assert meta.get("path") == f"daily/{today}/{pytorch_session_id}.md"

                pytorch_path = vault_root / meta["path"]
                assert pytorch_path.is_file(), f"created note not found at {pytorch_path}"

                pytorch_text = _read_text(pytorch_path)
                print("\n" + "=" * 70)
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
                ), f"CREATE only captured {topic_hits!r} of expected facts\n--- CREATE ---\n{pytorch_text}"

                stem = pytorch_path.stem
                assert (
                    f"name: {stem}" in pytorch_text
                ), f"frontmatter name does not match stem {stem!r}\n{pytorch_text[:400]}"

                print("\n" + "=" * 70)
                print("test_auto_memory_create passed")
                print("=" * 70)
            finally:
                await app.close()

    asyncio.run(run())


def test_auto_memory_update():
    """UPDATE an existing note — old facts must survive, new facts must land."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            app = await _make_app()
            try:
                vault_root = Path(app.config.vault_dir).absolute()
                today = _today()
                seed_path = _seed_note(vault_root, today)
                seed_before = _read_text(seed_path)
                assert "legal/compliance" in seed_before

                print("\n" + "=" * 70)
                print("[setup] vault_root =", vault_root)
                print("[setup] today      =", today)
                print("[setup] seed_path  =", seed_path)
                print("=" * 70)

                with _AgentMemoryRecorder(DUMP_DIR, prefix="agent_update") as recorder:
                    response = await app.run_job(
                        "auto_memory",
                        messages=_auth_messages(),
                        session_id=SEED_STEM,
                    )
                dumped = await recorder.dump()
                for p in dumped:
                    print(f"[UPDATE] agent memory dumped: {p}")

                assert response.success is True, f"UPDATE job failed: {response.answer!r}"
                meta = response.metadata or {}
                assert meta.get("created") is False, f"Expected created=False, got {meta!r}"
                assert meta.get("path") == f"daily/{today}/{SEED_STEM}.md"

                seed_after = _read_text(seed_path)
                print("\n" + "=" * 70)
                print(f"[UPDATE] {seed_path} ({len(seed_before)} -> {len(seed_after)} bytes)")
                print(f"[UPDATE] body after:\n{seed_after}")
                print("=" * 70)

                for old_fact in ("legal/compliance", "RS256", "Alice"):
                    assert (
                        old_fact in seed_after
                    ), f"UPDATE dropped pre-existing fact {old_fact!r}\n--- AFTER ---\n{seed_after}"

                new_hits = [
                    needle
                    for needle in ("PR #432", "432", "volatile-ttl", "maxmemory-policy", "2026-05-29")
                    if needle in seed_after
                ]
                print(f"[UPDATE] preserved old facts, landed new facts: {new_hits}")
                assert (
                    len(new_hits) >= 2
                ), f"UPDATE only landed {new_hits!r} of expected new facts\n--- AFTER ---\n{seed_after}"

                print("\n" + "=" * 70)
                print("test_auto_memory_update passed")
                print("=" * 70)
            finally:
                await app.close()

    asyncio.run(run())


if __name__ == "__main__":
    print("=== auto_memory integration test ===")
    test_auto_memory_create()
    test_auto_memory_update()
    print("\nAll integration tests passed!")
