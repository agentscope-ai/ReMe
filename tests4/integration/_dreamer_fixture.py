"""Fixture for the dreamer integration tests.

Seeds a vault with:

  - 4 pre-existing digest/ nodes spread across the three buckets
    (procedure / personal / wiki) — these are the **recall targets**;
    the new material partially overlaps them so Phase 2 must find
    them via search + read and decide UPDATE.
  - 4 small daily/ stubs the digest nodes already link to — they exist
    only so the seeded digest bodies don't dangle.
  - 1 NEW daily note (the file the dreamer will be invoked on).
    It exercises CREATE and UPDATE across the three buckets:

      * wiki      : UPDATE digest/wiki/jwt.md (24h rotation cadence
                    refines short-credential-compliance framing) +
                    CREATE digest/wiki/kid-versioning.md +
                    CREATE digest/wiki/soc2-30day-finding.md
                    (the OAuth2 restatement section is intentionally
                    a non-abstraction — Phase 1 should NOT emit a
                    sub-unit for it; tests Phase 1's gate-keeping)
      * procedure : UPDATE digest/procedure/key-rotation.md (24h
                    cadence + kid-versioning supersede the 30-day
                    JWKS-cache flow)
      * personal  : UPDATE digest/personal/no-trailing-summary.md
                    (extend "no trailing summary" to also forbid
                    "next steps" lists) + CREATE
                    digest/personal/small-pr.md

  Total budget per integration run: 1 Phase 1 + up-to-6 Phase 2 = up
  to 7 ReAct sessions, each with several tool turns (search +
  traverse → frontmatter_read + read → write / edit).

Idempotent: re-running does NOT overwrite existing files. To re-seed
from scratch, delete the vault and rerun.

Usage as a script:
    python tests4/integration/_dreamer_fixture.py /tmp/my-vault

Usage as a module:
    from _dreamer_fixture import clean_vault, seed_vault, INPUT_PATH
    clean_vault(Path("/tmp/my-vault"))
    seed_vault(Path("/tmp/my-vault"))
"""

import shutil
import sys
from pathlib import Path

INPUT_PATH = "daily/2026-05-28/auth-refactor/notes.md"


_FILES: dict[str, str] = {
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
    INPUT_PATH: """\
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
""",
}


_CLEAN_DIRS = ("daily", "digest", "reme_metadata")


def clean_vault(vault: Path) -> list[str]:
    """Remove fixture-managed subdirs (`daily/`, `digest/`, `reme_metadata/`)
    under `vault` so the next `seed_vault` starts from a clean slate.

    Returns the relative paths that were actually removed."""
    removed: list[str] = []
    for rel in _CLEAN_DIRS:
        target = vault / rel
        if target.exists():
            shutil.rmtree(target)
            removed.append(rel)
    return removed


def seed_vault(vault: Path) -> list[str]:
    """Write any missing fixture files under `vault`. Return relative paths
    that were actually written (skipped existing ones)."""
    seeded: list[str] = []
    for rel, body in _FILES.items():
        target = vault / rel
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")
        seeded.append(rel)
    return seeded


# pylint: disable=missing-function-docstring
def main() -> None:
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} <vault_dir>", file=sys.stderr)
        sys.exit(2)
    vault = Path(sys.argv[1]).resolve()
    vault.mkdir(parents=True, exist_ok=True)
    removed = clean_vault(vault)
    if removed:
        print(f"cleaned {len(removed)} dir(s) under {vault}: {', '.join(removed)}")
    seeded = seed_vault(vault)
    if seeded:
        print(f"seeded {len(seeded)} file(s) under {vault}:")
        for f in seeded:
            print(f"  + {f}")
    else:
        print(f"vault {vault} already seeded — no changes")
    print(f"\nDream this file:\n  {vault}/{INPUT_PATH}")


if __name__ == "__main__":
    main()
