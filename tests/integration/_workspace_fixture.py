"""Shared integration-test fixture: build a workspace test environment.

Every integration test in this directory used to repeat the same
boilerplate — temp dir, ``chdir``, ``load_env()``, ``_make_app()``,
``_today()``, ``_AgentMemoryRecorder``, ``_wait_for_glob`` /
``_wait_for_populated`` / ``_wait_for_server``, ad-hoc seed helpers
(daily notes, resource files, the dreamer's pre-existing digest nodes).

This module unifies all of that behind one entry point:

    from _workspace_fixture import workspace_env

    async def run():
        with workspace_env() as env:
            app = await env.make_app()
            try:
                # env.workspace_dir, env.today, env.place_resource(...),
                # env.seed_daily_note(...), env.seed_dream_workspace(),
                # env.wait_for_populated(...), env.record_agents(...) ...
                ...
            finally:
                await env.close_all()

``env.make_app()`` defaults to the standard config; pass
``config="cc"`` for the CC SDK wiring, or arbitrary kwargs to deep-merge
into ``resolve_app_config``. The workspace path is fixed to
``<tmp_workspace>/.reme`` so seed helpers can write files before the
app is started.

The dreamer-specific seed (4 pre-existing digest nodes spread across
the three buckets + 4 daily provenance stubs + a new daily note that
exercises CREATE and UPDATE in each bucket) is preserved as
``env.seed_dream_workspace()`` / ``DREAM_INPUT_PATH``.

Usage as a script (for the dreamer manual run):

    python tests/integration/_workspace_fixture.py /tmp/my-workspace
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import socket
import sys
import tempfile
import time
from datetime import date as _date
from pathlib import Path
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parents[2]
INTEGRATION_DIR = Path(__file__).resolve().parent


# ─────────────────────────────────────────────────────────────────────────────
# Dream preset — pre-existing digest nodes + a new daily note that exercises
# CREATE and UPDATE across the three buckets (procedure / personal / wiki).
# ─────────────────────────────────────────────────────────────────────────────

DREAM_INPUT_PATH = "daily/2026-05-28/auth-refactor/notes.md"

_DREAM_FILES: dict[str, str] = {
    # ----- pre-existing digest nodes (recall targets) -----
    "digest/wiki/jwt.md": """\
---
name: jwt
description: JSON Web Token — signed authentication token format
---

# JWT

JSON Web Token (RFC 7519). A compact, signed (JWS) or encrypted (JWE)
token used to assert identity and claims between parties.

## Structure
- Header — `alg`, `typ`, `kid`
- Payload — claims: `iss`, `sub`, `aud`, `exp`, `iat`
- Signature

## Related
Often issued by [[digest/wiki/oauth2.md]] flows.

derived_from:: [[daily/2026-05-15/auth-design/notes.md]]
""",
    "digest/wiki/oauth2.md": """\
---
name: oauth2
description: OAuth 2.0 — delegated authorization framework
---

# OAuth 2.0

RFC 6749. A delegated authorization framework: a resource owner grants
a client limited access to a protected resource via an access token
issued by an authorization server.

## Grant types
- Authorization code (with PKCE for public clients)
- Client credentials
- Refresh token

derived_from:: [[daily/2026-05-10/oauth-intro/notes.md]]
""",
    "digest/procedure/key-rotation.md": """\
---
name: key-rotation
description: Rotating signing keys for JWT issuance
---

# Key rotation (current — pre 2026-05-28 refactor)

Procedure for rotating the signing key used by [[digest/wiki/jwt.md]]
issuance.

## Steps
1. Generate new keypair offline.
2. Publish the public key to the JWKS endpoint with a fresh `kid`.
3. Wait 24h for clients to refresh their JWKS cache.
4. Cut over the signer to the new private key.
5. Mark the old `kid` as deprecated; remove after 30 days.

## Cadence
Default rotation cadence is **30 days**. Driven by historical practice;
no formal compliance requirement has tightened this so far.

derived_from:: [[daily/2026-05-20/rotation-plan/notes.md]]
""",
    "digest/personal/no-trailing-summary.md": """\
---
name: no-trailing-summary
description: 不要在回复末尾加总结段落
---

# 不要在回复末尾加总结段落

用户能看 diff,不需要在回复末尾重述刚做的事。

**Why**: diff 已经把"改了什么"摆在用户面前;再口述一遍是噪音。

**How to apply**: 任意编码 / 编辑任务回复结束时,直接停在最后一条
有信息量的话上,不要再补一段"以上就是本次的修改..."。

derived_from:: [[daily/2026-05-01/style-feedback/notes.md]]
""",
    # ----- daily provenance stubs (so the digest links don't dangle) -----
    "daily/2026-05-01/style-feedback/notes.md": """\
---
name: notes
description: style feedback to Claude on 2026-05-01
---

# Style feedback (2026-05-01)

每次任务结束都重述了一遍刚做的事——不需要,我能看 diff。以后直接停。
""",
    "daily/2026-05-10/oauth-intro/notes.md": """\
---
name: notes
description: OAuth 2.0 intro session
---

# OAuth 2.0 intro

简介 grant types: authorization code (with PKCE), client credentials,
refresh token。重点放在 PKCE 是给 public clients 用的。
""",
    "daily/2026-05-15/auth-design/notes.md": """\
---
name: notes
description: initial auth design discussion
---

# Auth design

讨论 JWT 的结构 (header / payload / signature) 和我们项目里的 claim
约定 (iss, sub, aud, exp, iat)。
""",
    "daily/2026-05-20/rotation-plan/notes.md": """\
---
name: notes
description: key rotation plan v1
---

# Key rotation plan v1

定下当前的 5 步轮换流程:offline 生成 keypair → 发布到 JWKS (新 kid)
→ 等 24h cache → 切签发 → 30 天后清旧 kid。周期定 30 天。
""",
    # ----- the NEW daily note dreamer will be invoked on -----
    DREAM_INPUT_PATH: """\
---
name: notes
description: auth refactor working notes — 2026-05-28
---

# Auth refactor — 2026-05-28

## 决定:JWT 轮换周期改为 24 小时

今天确定把 JWT 签名密钥的轮换周期从 30 天压到 **24 小时**。原因是
SOC2 合规审计批评:30 天的会话 token 太长,不满足"短期凭证"原则。

新流程不再依赖 JWKS cache 的 24h 等待,改成走 Redis 里的 `kid`
版本号实时下发。客户端在 token 验证失败时主动拉新 JWKS,而不是定
时轮询。

(这条同时更新 JWT 概念笔记和 key-rotation 流程笔记。)

## 新概念:kid 版本号机制

`kid` (key ID) 是 JWT header 里的字段。我们把它当成版本号来用:
Redis key `auth:jwks:current_kid` 保存当前活跃 kid;Auth Service
在签发 token 时读这个 key,客户端验证失败时也读这个 key 再拉对应
的 public key。这样无须等 cache TTL。

## 顺带复习:OAuth 2.0 是什么

(为了帮新同学接住上下文,这里把 OAuth 2.0 简单重述一下,不引入
新事实。)OAuth 2.0 (RFC 6749) 是一个委托授权框架:资源所有者
允许 client 通过 authorization server 颁发的 access token 来有
限度地访问受保护资源。常见 grant types: authorization code
(public client 用 PKCE)、client credentials、refresh token。
——这一段没有任何新内容,纯粹是给后面 JWT 24h 轮换决定铺垫读者
的背景知识。

## 观察:SOC2 审计在 30 天周期上的具体批评

审计员引用 SOC2 CC6.1 控制点:"会话凭证应有合理的短期有效期"。
30 天对应于人类工作周期,但对自动化客户端 token 来说过长。审计
要求 24h 或更短,且必须能在事件响应时立即吊销 (kid 切换可满足)。

## 偏好:小 PR 优先

后续这个 refactor 拆 PR 时,每个 PR 控制在 < 300 行。原因是 review
负担太大时容易被拍脑袋通过,这违背了 SOC2 审计中变更管理的精神。

## 偏好:回复结尾再补充

之前说过不要总结段落 (我能看 diff),今天再补充一点:也不要"接下来
的步骤"列表,除非我明确问 next steps。直接回答问题然后停。

## 关联:已有 digest

这次 refactor 直接更新这几篇已有 digest 笔记(这里用 wikilink
引用,方便检索关联):

- JWT 概念:[[digest/wiki/jwt.md]]
- 签名密钥轮换流程:[[digest/procedure/key-rotation.md]]
""",
}

_CLEAN_DIRS = ("daily", "digest", "resource", "metadata")


# ─────────────────────────────────────────────────────────────────────────────
# Small primitives
# ─────────────────────────────────────────────────────────────────────────────


def today() -> str:
    """ISO date today, e.g. ``2026-06-08`` — same shape every test expected."""
    return _date.today().isoformat()


@contextlib.contextmanager
def temp_chdir(path) -> Iterator[Path]:
    """``chdir`` to ``path`` for the block; restore the original cwd on exit."""
    old = os.getcwd()
    os.chdir(path)
    try:
        yield Path(path)
    finally:
        os.chdir(old)


def port_free(host: str, port: int) -> bool:
    """True iff (host, port) is currently bindable. Used by webhook tests
    to fail fast instead of racing the connector start-up against a
    listener that's already squatting the port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Agent transcript capture — monkey-patches ``Agent.__init__`` to grab every
# agent created within its ``with`` block, then dumps each agent's context
# to ``<dump_dir>/<prefix>_<idx>_<name>.jsonl`` on ``dump()``.
# ─────────────────────────────────────────────────────────────────────────────


class AgentMemoryRecorder:
    """Record every ``agentscope.agent.Agent`` instance created within the block.

    Used to surface the ReAct trace of Phase 1 / Phase 2 dreams, the
    daily-write fork, the resource-interpret agent, etc. — what tools
    were called in what order, what candidates were recalled, what the
    LLM decided. Dumps land under ``<workspace>/agent_logs/`` by default
    (created by :meth:`WorkspaceEnv.record_agents`) so they're auto-cleaned
    with the throwaway workspace. Pass an explicit ``dump_dir`` to persist
    them somewhere else for post-mortem inspection.
    """

    def __init__(self, dump_dir: Path, prefix: str = "agent"):
        self.dump_dir = dump_dir
        self.prefix = prefix
        self.agents: list[Any] = []
        self._orig_init = None
        self.dumped_paths: list[Path] = []

    def __enter__(self):
        from agentscope.agent import Agent  # local import — heavy module

        self._orig_init = Agent.__init__
        agents = self.agents
        orig = self._orig_init

        def _capturing_init(agent_self, *args, **kwargs):
            orig(agent_self, *args, **kwargs)
            agents.append(agent_self)

        Agent.__init__ = _capturing_init
        return self

    def __exit__(self, *exc):
        from agentscope.agent import Agent

        if self._orig_init is not None:
            Agent.__init__ = self._orig_init

    async def dump(self) -> list[Path]:
        """Serialize captured agent transcripts to ``<dump_dir>/`` and return paths."""
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


# ─────────────────────────────────────────────────────────────────────────────
# WorkspaceEnv — the value the ``workspace_env()`` context manager yields. Holds the
# resolved workspace path, app construction, seeding, wait helpers, recorder
# factory, and tracks any apps the test started so they get closed.
# ─────────────────────────────────────────────────────────────────────────────


class WorkspaceEnv:
    """A workspace test environment — temp workspace + helpers."""

    def __init__(self, workspace: Path, workspace_dir: Path):
        self.workspace = workspace
        self.workspace_dir = workspace_dir
        self.today = today()
        self._apps: list[Any] = []

    # ----- app construction -----------------------------------------------

    async def make_app(self, *, config: str | None = None, **overrides) -> Any:
        """Build and start an ``Application`` (default config) and track it for cleanup.

        Pass ``config="cc"`` for the CC SDK wiring; arbitrary kwargs are
        deep-merged into ``resolve_app_config`` (e.g. ``jobs={...}`` to
        inject a background connector).
        """
        from reme import Application
        from reme.config import resolve_app_config

        kwargs: dict[str, Any] = {
            "log_to_console": False,
            "log_to_file": False,
            "enable_logo": False,
            "workspace_dir": str(self.workspace_dir),
        }
        if config:
            kwargs["config"] = config
        kwargs.update(overrides)
        cfg = resolve_app_config(**kwargs)
        app = Application(**cfg)
        await app.start()
        self._apps.append(app)
        return app

    async def make_reme(self, **overrides) -> Any:
        """Same as ``make_app`` but returns a ``ReMe`` instance (alias of ``Application``)."""
        from reme import ReMe
        from reme.config import resolve_app_config

        kwargs: dict[str, Any] = {"workspace_dir": str(self.workspace_dir)}
        kwargs.update(overrides)
        cfg = resolve_app_config(**kwargs)
        app = ReMe(**cfg)
        await app.start()
        self._apps.append(app)
        return app

    async def close_all(self) -> None:
        """Close every app started via this env. Idempotent."""
        for app in self._apps:
            await app.close()
        self._apps.clear()

    # ----- seeding --------------------------------------------------------

    def clean(self) -> list[str]:
        """Remove fixture-managed subdirs (``daily/``, ``digest/``, ``resource/``,
        ``metadata/``) under the workspace so the next seed starts clean.
        Returns relative paths that were actually removed."""
        removed: list[str] = []
        for rel in _CLEAN_DIRS:
            target = self.workspace_dir / rel
            if target.exists():
                shutil.rmtree(target)
                removed.append(rel)
        return removed

    def seed_dream_workspace(self) -> list[str]:
        """Write the dreamer preset: pre-existing digest nodes + provenance
        stubs + the new daily note dreamer will be invoked on
        (``DREAM_INPUT_PATH``). Idempotent — skips files that already exist.
        Returns relative paths that were actually written."""
        seeded: list[str] = []
        for rel, body in _DREAM_FILES.items():
            target = self.workspace_dir / rel
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(body, encoding="utf-8")
            seeded.append(rel)
        return seeded

    def place_resource(
        self,
        filename: str,
        content: str,
        *,
        date: str | None = None,
    ) -> str:
        """Drop a file under ``resource/<date>/<filename>``. Returns the
        workspace-relative path so the caller can pass it straight to
        ``auto_resource``."""
        d = date or self.today
        resource_dir = self.workspace_dir / "resource" / d
        resource_dir.mkdir(parents=True, exist_ok=True)
        path = resource_dir / filename
        path.write_text(content, encoding="utf-8")
        return f"resource/{d}/{filename}"

    def seed_daily_note(
        self,
        stem: str,
        body: str,
        *,
        date: str | None = None,
    ) -> Path:
        """Write a note at ``daily/<date>/<stem>.md`` and return the absolute path."""
        d = date or self.today
        day_dir = self.workspace_dir / "daily" / d
        day_dir.mkdir(parents=True, exist_ok=True)
        path = day_dir / f"{stem}.md"
        path.write_text(body, encoding="utf-8")
        return path

    # ----- introspection --------------------------------------------------

    def daily_notes(self, *, date: str | None = None) -> list[Path]:
        """All ``.md`` files under ``daily/<date>/`` (sorted)."""
        d = date or self.today
        day_dir = self.workspace_dir / "daily" / d
        if not day_dir.is_dir():
            return []
        return sorted(day_dir.glob("*.md"))

    def digest_files(self) -> list[Path]:
        """All ``.md`` files anywhere under ``digest/`` (sorted)."""
        digest_root = self.workspace_dir / "digest"
        if not digest_root.is_dir():
            return []
        return sorted(digest_root.rglob("*.md"))

    def session_state_files(self, prefix: str = "session_state_") -> list[Path]:
        """All session-state jsonl files (the agent wrapper writes these under
        ``resource/`` whenever a ``session_id`` is provided)."""
        resource_dir = self.workspace_dir / "resource"
        if not resource_dir.exists():
            return []
        return sorted(resource_dir.rglob(f"{prefix}*.jsonl"))

    # ----- async wait helpers --------------------------------------------

    async def wait_for_glob(
        self,
        parent: Path,
        pattern: str,
        timeout: float,
        poll: float = 0.5,
    ) -> Path:
        """Poll ``parent.glob(pattern)`` until the first match appears or timeout."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if parent.is_dir():
                matches = sorted(parent.glob(pattern))
                if matches:
                    return matches[0]
            await asyncio.sleep(poll)
        listing = [p.name for p in parent.iterdir()] if parent.is_dir() else []
        raise TimeoutError(
            f"timeout after {timeout}s waiting for {parent}/{pattern}; dir listing: {listing}",
        )

    async def wait_for_populated(
        self,
        parent: Path,
        pattern: str,
        min_bytes: int,
        timeout: float,
        poll: float = 1.0,
    ) -> Path:
        """Like ``wait_for_glob`` but only returns once the file exceeds ``min_bytes``.

        ``daily_create`` writes a ~50-byte frontmatter-only stub before the
        agent fills in the body, so a simple ``glob`` check returns too early.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if parent.is_dir():
                for p in sorted(parent.glob(pattern)):
                    try:
                        if p.stat().st_size >= min_bytes:
                            return p
                    except OSError:
                        pass
            await asyncio.sleep(poll)
        listing = [(p.name, p.stat().st_size) for p in parent.iterdir()] if parent.is_dir() else []
        raise TimeoutError(
            f"timeout after {timeout}s waiting for {parent}/{pattern} with size>={min_bytes}; " f"current: {listing}",
        )

    async def wait_for_server(self, url: str, timeout: float = 10.0) -> None:
        """Poll ``url`` until the server answers (any 2xx/4xx counts). Used by
        the webhook test so the POST isn't racing uvicorn's bind."""
        import httpx  # local — keep optional

        deadline = time.monotonic() + timeout
        async with httpx.AsyncClient(timeout=1.0) as client:
            while time.monotonic() < deadline:
                try:
                    resp = await client.request("HEAD", url)
                    if resp.status_code in (200, 202, 404, 405):
                        return
                except (httpx.ConnectError, httpx.ReadError):
                    pass
                await asyncio.sleep(0.2)
        raise TimeoutError(f"server didn't come up at {url} within {timeout}s")

    # ----- agent recorder factory ----------------------------------------

    def record_agents(
        self,
        prefix: str = "agent",
        dump_dir: Path | None = None,
    ) -> AgentMemoryRecorder:
        """Context manager that captures every Agent created in its block.

        Dumps default to ``<workspace>/agent_logs/`` so they're cleaned up with
        the throwaway workspace. Pass ``dump_dir`` to persist them elsewhere
        (e.g. for post-mortem inspection of a failing run).
        """
        return AgentMemoryRecorder(
            dump_dir=dump_dir or self.workspace_dir / "agent_logs",
            prefix=prefix,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────


@contextlib.contextmanager
def workspace_env(
    *,
    chdir: bool = True,
    workspace_name: str = ".reme",
    load_env_file: bool = True,
) -> Iterator[WorkspaceEnv]:
    """Yield a fresh ``WorkspaceEnv`` rooted at a temp workspace.

    - Creates ``<tmp>/<workspace_name>`` eagerly so seed helpers work before
      ``make_app()``.
    - chdirs into the temp workspace (so relative workspace_dir resolves
      correctly and any helper that writes under cwd lands inside the
      throwaway tree).
    - Loads ``.env`` once (idempotent — safe to call repeatedly).

    The workspace is cleaned up automatically when the block exits. The
    caller is still responsible for ``await env.close_all()`` to release
    any apps it started.
    """
    if load_env_file:
        from reme.utils import load_env

        load_env()

    with tempfile.TemporaryDirectory() as tmp_dir:
        workspace = Path(tmp_dir).resolve()
        workspace = workspace / workspace_name
        workspace.mkdir(parents=True, exist_ok=True)
        env = WorkspaceEnv(workspace=workspace, workspace_dir=workspace)

        if chdir:
            with temp_chdir(workspace):
                yield env
        else:
            yield env


# ─────────────────────────────────────────────────────────────────────────────
# Script entry — replicate the original dreamer manual-seed CLI.
# ─────────────────────────────────────────────────────────────────────────────


# pylint: disable=missing-function-docstring
def main() -> None:
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} <workspace_dir>", file=sys.stderr)
        sys.exit(2)
    workspace = Path(sys.argv[1]).resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    env = WorkspaceEnv(workspace=workspace.parent, workspace_dir=workspace)
    removed = env.clean()
    if removed:
        print(f"cleaned {len(removed)} dir(s) under {workspace}: {', '.join(removed)}")
    seeded = env.seed_dream_workspace()
    if seeded:
        print(f"seeded {len(seeded)} file(s) under {workspace}:")
        for f in seeded:
            print(f"  + {f}")
    else:
        print(f"workspace {workspace} already seeded — no changes")
    print(f"\nDream this file:\n  {workspace}/{DREAM_INPUT_PATH}")


if __name__ == "__main__":
    main()
