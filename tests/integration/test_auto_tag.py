"""Real-LLM integration tests for ``auto_tag_step``.

The fixtures mimic the final Markdown written by the Auto Fin and Daily Paper
plugins, then invoke AutoTagStep through a normal application Job.  The tests
load the repository ``.env`` and therefore call the configured real LLM.
"""

import asyncio
import sys
from pathlib import Path

import frontmatter

INTEGRATION_DIR = Path(__file__).resolve().parent
REPOSITORY = INTEGRATION_DIR.parents[1]
sys.path.insert(0, str(REPOSITORY))
sys.path.insert(0, str(INTEGRATION_DIR))

# pylint: disable=wrong-import-position
from _workspace_fixture import workspace_env  # noqa: E402

from reme.utils import load_env  # noqa: E402

AUTO_TAG_JOB = {
    "integration_auto_tag": {
        "backend": "base",
        "enable_serve": False,
        "steps": [{"backend": "auto_tag_step"}],
    },
}

AUTO_FIN_REPORT = """\
---
name: auto_fin
description: 宁德时代海外电池业务与产能进展的盘后研究
kind: auto-fin-brief
---

# 宁德时代：欧洲电池业务进入产能兑现期

> 本文围绕宁德时代这一家公司，复盘其欧洲电池工厂的量产进展与客户交付节奏。

## 核心变化

宁德时代确认匈牙利工厂首条电芯产线开始试生产。管理层称，后续爬坡速度仍取决于良率、当地供应链和
客户认证。该进展可能缩短欧洲客户的交付半径，但资本开支和产能利用率仍是主要风险。

## 观察框架

后续应继续跟踪宁德时代的海外产能利用率、单位成本与订单兑现情况。本文只提供新闻研究和回顾线索，
不提供收益、目标价或买卖建议。
"""

DAILY_PAPER_NOTES = {
    "daily/2026-09-10/openai-agent-eval.md": """\
---
name: openai-agent-eval
description: OpenAI 智能体可靠性评测论文解读
kind: daily-paper-analysis
arxiv_id: "2609.10001"
---

# OpenAI 智能体可靠性评测

这篇论文以 OpenAI 为唯一机构研究对象，分析其智能体在长任务中的失败恢复机制。实验比较了重试预算、
工具错误和上下文压缩对完成率的影响，并讨论 OpenAI 对可靠性评测的设计选择。
""",
    "daily/2026-09-10/anthropic-context.md": """\
---
name: anthropic-context
description: Anthropic 长上下文研究论文解读
kind: daily-paper-analysis
arxiv_id: "2609.10002"
---

# Anthropic 长上下文研究

论文只研究 Anthropic 的长上下文模型。作者测试信息位于不同位置时的召回差异，并分析 Anthropic 模型的
注意力退化现象；上下文压缩只是实验方法，不是本文要标记的现实实体。
""",
    "daily/2026-09-10/nvidia-blackwell.md": """\
---
name: nvidia-blackwell
description: NVIDIA Blackwell 训练系统论文解读
kind: daily-paper-analysis
arxiv_id: "2609.10003"
---

# NVIDIA Blackwell 训练系统

本文的机构研究对象是 NVIDIA。论文评估 Blackwell 集群的大模型训练吞吐、故障恢复和互连扩展效率，
并给出 NVIDIA 在超大规模训练系统上的工程取舍。
""",
}

DAILY_PAPER_DIGEST = """\
---
name: 每日论文简报-2026-09-10
description: OpenAI、Anthropic 与 NVIDIA 三项研究的每日论文简报
kind: daily-paper-brief
---

# 每日论文简报：智能体、长上下文与训练系统

今天的三篇论文分别围绕三个相互独立的机构实体展开：OpenAI 的智能体可靠性评测、Anthropic 的长上下文
研究，以及 NVIDIA 的 Blackwell 训练系统。这三家机构都是本期简报的并列核心，而非顺带提及。

## 来源

- [[daily/2026-09-10/openai-agent-eval.md]]
- [[daily/2026-09-10/anthropic-context.md]]
- [[daily/2026-09-10/nvidia-blackwell.md]]
"""

GENERIC_TAGS = {
    "ai",
    "人工智能",
    "金融",
    "股票",
    "论文",
    "研究",
    "智能体",
    "长上下文",
    "训练系统",
    "电池",
    "半导体",
}


def _write(workspace: Path, relative: str, content: str) -> Path:
    target = workspace / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return target


def _post(path: Path) -> frontmatter.Post:
    return frontmatter.loads(path.read_text(encoding="utf-8"))


def _tags(path: Path) -> list[str]:
    value = _post(path).metadata.get("memory_tags")
    assert isinstance(value, list), f"memory_tags was not written as a list: {value!r}"
    assert all(isinstance(item, str) and item.strip() for item in value)
    assert len(value) <= 3
    assert not ({item.casefold() for item in value} & {item.casefold() for item in GENERIC_TAGS})
    return value


def _assert_tag(tags: list[str], expected: str) -> None:
    assert expected.casefold() in {tag.casefold() for tag in tags}, f"expected entity {expected!r}, got {tags!r}"


def _assert_any_tag(tags: list[str], expected: set[str]) -> None:
    actual = {tag.casefold() for tag in tags}
    accepted = {tag.casefold() for tag in expected}
    assert actual & accepted, f"expected one of {sorted(expected)!r}, got {tags!r}"


async def _run_tag_job(env, changes: list[dict[str, str]]):
    app = await env.make_app(jobs=AUTO_TAG_JOB)
    response = await app.run_job("integration_auto_tag", changes=changes)
    assert response.success is True, f"auto-tag failed: {response.answer!r}; metadata={response.metadata!r}"
    auto_tag = response.metadata["auto_tag"]
    assert auto_tag["processed"] == len(changes)
    assert auto_tag["succeeded"] == len(changes)
    assert auto_tag["failed"] == 0
    assert auto_tag["ignored"] == []
    return app, response


def test_auto_tag_auto_fin_report_uses_company_entity():
    """An Auto Fin report should be tagged with its company, not broad finance topics."""

    async def run():
        load_env(REPOSITORY / ".env")
        with workspace_env(load_env_file=False) as env:
            relative = "daily/2026-09-10/auto_fin.md"
            path = _write(env.workspace_dir, relative, AUTO_FIN_REPORT)
            before = _post(path)
            try:
                _app, _response = await _run_tag_job(env, [{"change": "added", "path": relative}])
                tags = _tags(path)
                _assert_any_tag(tags, {"宁德时代", "CATL"})
                assert len(tags) == 1, f"single-entity report received extra tags: {tags!r}"
                after = _post(path)
                assert after.content == before.content
                assert {key: value for key, value in after.metadata.items() if key != "memory_tags"} == before.metadata
            finally:
                await env.close_all()

    asyncio.run(run())


def test_auto_tag_daily_paper_batch_tags_analyses_and_digest():
    """The three analyses get one entity each and the digest gets all three."""

    async def run():
        load_env(REPOSITORY / ".env")
        with workspace_env(load_env_file=False) as env:
            paths = {
                relative: _write(env.workspace_dir, relative, content)
                for relative, content in DAILY_PAPER_NOTES.items()
            }
            digest_rel = "daily/2026-09-10/daily-paper-brief.md"
            paths[digest_rel] = _write(env.workspace_dir, digest_rel, DAILY_PAPER_DIGEST)
            before = {relative: _post(path) for relative, path in paths.items()}
            changes = [{"change": "added", "path": relative} for relative in paths]
            try:
                _app, response = await _run_tag_job(env, changes)
                expected = {
                    "daily/2026-09-10/openai-agent-eval.md": "OpenAI",
                    "daily/2026-09-10/anthropic-context.md": "Anthropic",
                    "daily/2026-09-10/nvidia-blackwell.md": "NVIDIA",
                }
                for relative, entity in expected.items():
                    tags = _tags(paths[relative])
                    assert len(tags) == 1, f"single-entity analysis received extra tags: {relative} -> {tags!r}"
                    _assert_tag(tags, entity)

                digest_tags = _tags(paths[digest_rel])
                assert {tag.casefold() for tag in digest_tags} == {
                    "openai",
                    "anthropic",
                    "nvidia",
                }
                for relative, path in paths.items():
                    after = _post(path)
                    assert after.content == before[relative].content
                    assert {key: value for key, value in after.metadata.items() if key != "memory_tags"} == before[
                        relative
                    ].metadata
                auto_tag = response.metadata["auto_tag"]
                assert [item["path"] for item in auto_tag["results"]] == list(paths)
            finally:
                await env.close_all()

    asyncio.run(run())


def test_auto_tag_modified_note_reuses_existing_canonical_tag():
    """A modified note should reuse an existing workspace spelling for the same entity."""

    async def run():
        load_env(REPOSITORY / ".env")
        with workspace_env(load_env_file=False) as env:
            _write(
                env.workspace_dir,
                "daily/2026-09-09/openai-history.md",
                "---\nname: openai-history\nmemory_tags: [OpenAI]\n---\n# OpenAI 历史记录\n",
            )
            relative = "daily/2026-09-10/openai-update.md"
            target = _write(
                env.workspace_dir,
                relative,
                """\
---
name: openai-update
description: OpenAI, Inc. 产品更新记录
memory_tags: [大模型]
status: reviewed
---

# OpenAI, Inc. 产品更新

这份记忆只围绕 OpenAI 公司，记录其产品发布节奏。文中的“大模型”是技术类别，不是现实实体标签。
""",
            )
            body_before = _post(target).content
            try:
                _app, response = await _run_tag_job(env, [{"change": "modified", "path": relative}])
                tags = _tags(target)
                assert [tag.casefold() for tag in tags] == ["openai"]
                post = _post(target)
                assert post.content == body_before
                assert post.metadata["status"] == "reviewed"
                assert response.metadata["auto_tag"]["results"][0]["change"] == "modified"
            finally:
                await env.close_all()

    asyncio.run(run())


if __name__ == "__main__":
    test_auto_tag_auto_fin_report_uses_company_entity()
    test_auto_tag_daily_paper_batch_tags_analyses_and_digest()
    test_auto_tag_modified_note_reuses_existing_canonical_tag()
