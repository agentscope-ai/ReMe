<p align="center">
 <img src="docs/figure/reme_logo.png" alt="ReMe Logo" width="50%">
</p>

<p align="center">
  <a href="https://pypi.org/project/reme-ai/"><img src="https://img.shields.io/badge/python-3.11+-blue" alt="Python Version"></a>
  <a href="https://pypi.org/project/reme-ai/"><img src="https://img.shields.io/pypi/v/reme-ai.svg?logo=pypi" alt="PyPI Version"></a>
  <a href="https://pepy.tech/project/reme-ai/"><img src="https://img.shields.io/pypi/dm/reme-ai" alt="PyPI Downloads"></a>
  <a href="https://github.com/agentscope-ai/ReMe"><img src="https://img.shields.io/github/commit-activity/m/agentscope-ai/ReMe?style=flat-square" alt="GitHub commit activity"></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-black" alt="License"></a>
  <a href="./README.md"><img src="https://img.shields.io/badge/English-Click-yellow" alt="English"></a>
  <a href="./README_ZH.md"><img src="https://img.shields.io/badge/简体中文-点击查看-orange" alt="简体中文"></a>
  <a href="https://github.com/agentscope-ai/ReMe"><img src="https://img.shields.io/github/stars/agentscope-ai/ReMe?style=social" alt="GitHub Stars"></a>
  <a href="https://deepwiki.com/agentscope-ai/ReMe"><img src="https://img.shields.io/badge/DeepWiki-Ask_Devin-navy.svg" alt="DeepWiki"></a>
</p>

<p align="center">
<a href="https://trendshift.io/repositories/20528" target="_blank"><img src="https://trendshift.io/api/badge/repositories/20528" alt="agentscope-ai%2FReMe | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

<p align="center">
  <strong>一个将对话和资料转化为可读、可编辑、可检索 Markdown 记忆的 Agent 记忆层。</strong><br>
</p>

> 历史版本：[0.3.x](https://github.com/agentscope-ai/ReMe/tree/reme_v3) ·
> [0.2.x](https://github.com/agentscope-ai/ReMe/tree/v0.2.0.6) ·
> [MemoryScope](https://github.com/agentscope-ai/ReMe/tree/memoryscope_branch)

🧠 ReMe 是一个面向 **AI 智能体** 的 local-first 记忆层。它把对话和资料沉淀为文件化长期记忆，并持续完成索引、链接和整理，让后续
Agent 能够可靠召回。

## ✨ 核心创新

- **Memory as File**：以带 frontmatter 和 wikilink 的 Markdown 作为记忆节点，让用户和 Agent 都能直接读写。
- **自进化知识库**：通过 Auto Memory、Auto Resource 和 Auto Dream，把对话与资料逐步加工为长期记忆，并自动建立 wikilink 关系。
- **渐进式混合搜索**：融合 wikilink、BM25 和 embedding，支持从关键词匹配到语义召回、关系扩展的混合检索。
- **Agent 友好集成**：通过 SKILL.md + CLI 接入，方便不同 Agent 读写、维护与复用记忆。

<p align="center">
  <img src="docs/figure/design-philosophy.svg" alt="ReMe 设计理念" width="92%">
</p>

## 🔭 适用场景

- **Personal assistants**：为 [QwenPaw](https://github.com/agentscope-ai/QwenPaw)、
  [OpenClaw](https://github.com/openclaw/openclaw)、[Hermes](https://github.com/nousresearch/hermes-agent)
  等个人助理提供用户可编辑的长期记忆层。
- **Coding agents**：在接入 [Claude Code](plugins/reme) 等 coding agent 时，跨会话保留代码风格、项目背景、仓库决策和流程经验。
- **LLM Wiki**：把对话、笔记和资料转化为可检索、可追溯、可链接的 Markdown 知识库，由用户和 Agent 共同维护。
- **Self-evolving agents**：帮助 Agent 从经验中学习，把成功路径、失败尝试、可复用流程和阶段性反思沉淀为记忆。

## 📰 新闻

- [2026.07] - 新增可选的 Cookbook 工作流，首个能力为 [每日论文](cookbook/daily_paper/README_ZH.md)，支持定时发现论文、
  Agent 辅助解析 PDF、沉淀可复用的 Markdown 笔记并生成五分钟简报。
- [2026.07] -
  我们的论文 [Remember Me, Refine Me: A Dynamic Procedural Memory Framework for Experience-Driven Agent Evolution](https://aclanthology.org/2026.findings-acl.829/)
  已被 Findings of ACL 2026 接收。

## 🚀 快速开始

### 安装

ReMe 要求 Python 3.11+。

从 pip 安装：

```bash
pip install "reme-ai[core]"
```

从源码安装：

```bash
git clone https://github.com/agentscope-ai/ReMe.git
cd ReMe
pip install -e ".[core]"
```

### 环境变量

如果需要 LLM 驱动的记忆演化或 embedding 检索，可以配置环境变量。embedding 默认关闭，因此默认配置不会启动
embedding 模型，也不需要 embedding API key。

```bash
cat > .env <<'EOF'
# 可选：仅在配置中显式启用 embedding 组件后使用。
# EMBEDDING_API_KEY=sk-xxx
# EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# 必须：auto_memory、auto_resource 和 auto_dream 需要 LLM。
LLM_API_KEY=sk-xxx
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
EOF
```

基础文件读写、BM25 检索、wikilink 遍历和 proactive topics 读取可以先不配置 LLM 凭证。

> [!NOTE]
> 如需启用基于 embedding 的语义检索，请取消 [`reme/config/default.yaml`](reme/config/default.yaml) 中
> `components.as_embedding` 和 `components.embedding_store` 的注释，并将
> `components.file_store.default.embedding_store` 从 `""` 改为 `default`。完整说明见
> [记忆检索文档](docs/zh/memory_search.md)。

### 启动服务

```bash
reme start
```

默认服务地址是 `127.0.0.1:2333`。如果端口被占用，可以指定其他端口：

```bash
reme start service.port=8181
# reme start workspace_dir=/tmp/reme-demo service.port=8181
```

启动后可以检查服务状态；如果使用了自定义端口，请将下面 URL 中的 `2333` 替换为对应端口。

```bash
reme version
curl -s http://127.0.0.1:2333/version -H 'Content-Type: application/json' -d '{}'
```

### 5 分钟记忆 Demo

服务运行后，可以写入一个记忆节点，让 ReMe 索引并检索它：

```bash
reme write \
  path=digest/wiki/quick-start-demo \
  name="Quick Start Demo" \
  description="第一个 ReMe 记忆节点" \
  content="# Quick Start Demo

ReMe 会把 Agent 记忆保存为可读的 Markdown。

相关链接：[[digest/wiki/memory-as-file.md]]"

reme search query="agent memory markdown" limit=5
reme read path=digest/wiki/quick-start-demo start_line=1 end_line=20
```

生成的文件是普通 Markdown，并带有 frontmatter：

```markdown
---
name: Quick Start Demo
description: 第一个 ReMe 记忆节点
---

# Quick Start Demo

ReMe 会把 Agent 记忆保存为可读的 Markdown。

相关链接：[[digest/wiki/memory-as-file.md]]
```

## 🧑‍🍳 Cookbooks

Cookbook 是由 ReMe jobs 和 steps 组装而成的可选端到端工作流。默认配置不会开启它们；启动 ReMe 时选择对应的
独立配置即可启用。后续新增的 cookbook 会继续在表格中按行追加。

| Cookbook                                  | 能力                                       |
|-------------------------------------------|------------------------------------------|
| [每日论文](cookbook/daily_paper/README_ZH.md) | 发现并排序论文，使用 Agent 解读 PDF，生成文件化论文笔记和五分钟简报。 |

## 📁 记忆系统

> Memory as File, File as Memory.

ReMe 将**记忆视为文件**，让原始对话和外部资料从 `session/`、`resource/` 渐进加工到 `daily/`，再沉淀为 `digest/`
中可长期复用的知识节点。

### 目录结构

```text
<workspace_dir>/
├── metadata/       # 系统索引、图谱、catalog 等持久状态
├── session/        # 原始对话和 Agent session
│   ├── dialog/
│   │   └── <session_id>.jsonl
│   ├── agentscope/
│   └── claude_code/
├── resource/            # 外部原始材料
│   └── YYYY-MM-DD/
│       └── <resource>.<ext>
├── daily/               # 浅加工记忆：当天事实、对话摘要、资源解读
│   ├── YYYY-MM-DD.md
│   └── YYYY-MM-DD/
│       ├── <session_event>.md
│       ├── <resource_stem>.md
│       └── interests.yaml
└── digest/              # 长期记忆：个人事实、流程经验、知识节点
    ├── personal/
    │   └── {topic/event}.md
    ├── procedure/
    │   └── {topic/event}.md
    └── wiki/
        └── {topic/event}.md
```

<p align="center">
  <img src="docs/figure/reme-overview.svg" alt="ReMe 文件化记忆系统总览" width="92%">
</p>

## 🧭 记忆设计理念

> 捕获原始对话和资料，将其整理为长期偏好、可复用经验和有价值的知识，并让结果始终能被用户和 Agent 直接编辑。

### 自动记忆流程

ReMe 遵循 capture → index → consolidate → recall 的循环。对话和资料先变成 daily 记忆卡片；后台任务保持文件可检索；
`auto_dream` 将稳定知识沉淀到 `digest/`；Agent 再通过搜索、wikilink 或 proactive topics 召回记忆。

| 能力                                          | 入口                               | 作用                                                 | 输出                                                   |
|---------------------------------------------|----------------------------------|----------------------------------------------------|------------------------------------------------------|
| [`auto_memory`](docs/zh/auto_memory.md)     | Agent hook 或 `reme auto_memory`  | 提炼有长期价值的对话事实，同时保留原始 session。                       | `session/dialog/*.jsonl`、`daily/<date>/<session>.md` |
| [`auto_resource`](docs/zh/auto_resource.md) | 资源监听或 `reme auto_resource`       | 将 `resource/<date>/` 下的文件转为带来源链接的 daily 卡片。        | `daily/<date>/<resource-card>.md`                    |
| [`auto_index`](docs/zh/memory_search.md)    | 后台监听或 `reme reindex`             | 维护 chunks、BM25 索引、wikilink 图谱及可选的 embedding 索引。    | 可检索的 `daily/`、`digest/`、`resource/` 内容               |
| [`auto_dream`](docs/zh/auto_dream.md)       | `dream_cron` 或 `reme auto_dream` | 将变化的 daily 卡片整理为长期 personal、procedure 和 wiki 记忆。   | `digest/**`、`daily/<date>/interests.yaml`            |
| [`proactive`](docs/zh/proactive.md)         | Agent 决定主动行动前调用 `reme proactive` | 读取 `auto_dream` 生成的 topics；是否以及如何提醒用户由宿主 Agent 决定。 | 来自 `daily/<date>/interests.yaml` 的结构化 topics         |

<table>
  <tr>
    <td align="center" width="50%">
      <img src="docs/figure/memory-as-file.svg" alt="Memory as File" width="92%">
    </td>
    <td align="center" width="50%">
      <img src="docs/figure/auto-memory-resource.svg" alt="Auto Memory and Resource" width="92%">
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <img src="docs/figure/auto-dream-and-proactive.svg" alt="Auto Dream and Proactive" width="92%">
    </td>
    <td align="center" width="50%">
      <img src="docs/figure/auto-index-and-memory-search.svg" alt="Auto Index and Memory Search" width="92%">
    </td>
  </tr>
</table>

## 🤝 Agent-friendly Integration

ReMe 作为本地记忆服务运行，并提供 CLI、HTTP API、MCP server 和 SDK 等多种接入方式。不同 Agent 可以选择适合自身 runtime
的路径，同时共享同一个本地 memory workspace。

| Agent                                                | 推荐接入方式                                                            | 开箱可用能力                                                          |
|------------------------------------------------------|-------------------------------------------------------------------|-----------------------------------------------------------------|
| **QwenPaw**                                          | 通过 Python SDK 嵌入 ReMe。                                            | 复用应用自身生命周期和模型配置，同时保持 memory 本地、文件化。                             |
| **Claude Code**                                      | 以 MCP service 启动 ReMe，并安装 [plugins/reme](plugins/reme)。           | MCP recall tools、`reme-memory` skill，以及自动记录会话的 Stop hook。       |
| **Other CLI-capable agents (OpenClaw/Hermes/Codex)** | 复制或安装 [skills/reme_memory/SKILL.md](skills/reme_memory/SKILL.md)。 | 通过 CLI 搜索/读取/写入记忆，并调用 `auto_memory`、`auto_dream` 和 `proactive`。 |

<p align="center"><b>集成演示</b></p>

<table>
  <tr>
    <td align="center"></td>
    <td width="45%" align="center"><b>Auto Memory</b></td>
    <td width="45%" align="center"><b>Auto Dream</b></td>
  </tr>
  <tr>
    <td align="center"><b>QwenPaw</b></td>
    <td width="45%">
      <img src="docs/figure/qwenpaw-auto-memory.gif" alt="QwenPaw Auto Memory 演示" width="100%">
    </td>
    <td width="45%">
      <img src="docs/figure/qwenpaw-auto-dream.gif" alt="QwenPaw Auto Dream 演示" width="100%">
    </td>
  </tr>
  <tr>
    <td align="center"><b>Claude Code</b></td>
    <td width="45%">
      <img src="docs/figure/cc-auto-memory.gif" alt="Claude Code Auto Memory 演示" width="100%">
    </td>
    <td width="45%">
      <img src="docs/figure/cc-auto-dream.gif" alt="Claude Code Auto Dream 演示" width="100%">
    </td>
  </tr>
</table>

## 🛠️ ReMe Operations

ReMe 通过 CLI 暴露的统一 job interface 操作 workspace。Agent 通常只需要使用检索、读取、写入、编辑和自动记忆相关命令；更底层的索引、
frontmatter 和文件操作接口主要用于维护、调试或高级集成。完整 job 列表可以运行 `reme help` 查看。

| 命令                                        | 作用                                          |
|-------------------------------------------|---------------------------------------------|
| `reme start`                              | 启动本地 ReMe 服务。                               |
| `reme version` / `reme health_check`      | 检查包版本和组件状态。                                 |
| `reme status`                             | 查看有状态数据组件的内存估算及进程 RSS。                      |
| [`reme search`](docs/zh/memory_search.md) | 默认使用 BM25 和 wikilink 检索，启用后增加向量检索。          |
| `reme read` / `reme write` / `reme edit`  | 检查和维护 Markdown 记忆文件。                        |
| `reme auto_memory`                        | 将对话 messages 转为 daily 记忆卡片；需要 LLM 凭证。       |
| `reme auto_resource`                      | 将 `resource/` 下的文件解读为 daily 资料卡片；需要 LLM 凭证。 |
| `reme auto_dream` / `reme proactive`      | 将 daily 记忆整理为长期 digest，并暴露值得关注的主题。          |
| `reme reindex`                            | 基于已有文件重建检索和 wikilink 索引。                    |

## 🤝 社区与支持

- **问题反馈与需求**：请先查看 [Open Issues](https://github.com/agentscope-ai/ReMe/issues)；如无相关讨论，可新建 Issue
  说明背景、目标行为和影响范围。
- **代码贡献**：改动前建议阅读 [贡献指南](https://docs.agentscope.io/reme/stable/zh/contributing)。架构与扩展方式以源码、schema
  和测试为准。
- **文档贡献**：用户可见文档请提交到[统一文档仓库](https://github.com/agentscope-ai/docs)的 `reme/<version>/{en,zh}/` 目录。
- **提交规范**：建议使用 Conventional Commits，例如 `feat(search): add link expansion option`、
  `docs(zh): update quick start`。
- **提交前检查**：提交 PR 前请尽量运行 `pre-commit run --all-files` 和 `pytest`；如有依赖 LLM、embedding 或外部服务的测试无法运行，请在
  PR 中说明。
- **获取帮助**：如需反馈 Bug 或功能请求，请使用 [GitHub Issues](https://github.com/agentscope-ai/ReMe/issues)；项目文档见
  [https://docs.agentscope.io/](https://docs.agentscope.io/reme/stable/zh/)。

### 贡献者

感谢所有为 ReMe 做出贡献的朋友们：

<a href="https://github.com/agentscope-ai/ReMe/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=agentscope-ai/ReMe" alt="贡献者" />
</a>

## 📄 引用

```bibtex
@software{ReMe2026,
  title = {Remember me, Refine me: Memory Management Kit for Agents},
  author = {ReMe Team},
  url = {https://reme.agentscope.io},
  year = {2026}
}
```

## ⚖️ 许可证

本项目基于 Apache License 2.0 开源，详情参见 [LICENSE](./LICENSE) 文件。
