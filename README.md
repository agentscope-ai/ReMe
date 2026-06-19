<div align="center">
  <img src="docs/figure/reme_logo.png" alt="ReMe Logo" width="420">

  <h3>Remember Me, Refine Me</h3>
  <p><strong>面向 Agent 的、文件优先的自进化记忆系统。</strong></p>

  <p>
    <a href="https://pypi.org/project/reme-ai/"><img src="https://img.shields.io/badge/python-3.11+-3776AB?logo=python&logoColor=white" alt="Python Version"></a>
    <a href="https://pypi.org/project/reme-ai/"><img src="https://img.shields.io/pypi/v/reme-ai.svg?logo=pypi&logoColor=white" alt="PyPI Version"></a>
    <a href="https://pepy.tech/project/reme-ai/"><img src="https://img.shields.io/pypi/dm/reme-ai?color=2ea44f" alt="PyPI Downloads"></a>
    <a href="https://github.com/agentscope-ai/ReMe"><img src="https://img.shields.io/github/commit-activity/m/agentscope-ai/ReMe?color=7c3aed" alt="GitHub commit activity"></a>
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-111827" alt="License"></a>
  </p>

  <p>
    <a href="https://github.com/agentscope-ai/ReMe">GitHub</a>
    ·
    <a href="https://deepwiki.com/agentscope-ai/ReMe">DeepWiki</a>
    ·
    <a href="./docs/reme_design.md">设计文档</a>
    ·
    <a href="./README_old.md">历史 README</a>
  </p>

  <p>
    <a href="https://github.com/agentscope-ai/ReMe"><img src="https://img.shields.io/github/stars/agentscope-ai/ReMe?style=social" alt="GitHub Stars"></a>
    <a href="https://trendshift.io/repositories/20528" target="_blank"><img src="https://trendshift.io/api/badge/repositories/20528" alt="agentscope-ai/ReMe | Trendshift" width="220" height="48"></a>
  </p>

  <p>
    历史版本：
    <a href="https://github.com/agentscope-ai/ReMe/tree/v0.3.1.10">0.3.x</a>
    ·
    <a href="https://github.com/agentscope-ai/ReMe/tree/v0.2.0.6">0.2.x</a>
    ·
    <a href="https://github.com/agentscope-ai/ReMe/tree/memoryscope_branch">memoryscope</a>
  </p>
</div>

---

🧠 ReMe 是一个专为 **AI Agent** 打造的记忆管理框架。它把记忆保存为 vault 目录中的普通文件，并通过 Markdown、front matter、wikilink、BM25 索引、文件图谱和后台 Agent 管线，把对话与外部资料逐步沉淀为可读、可查、可追溯的长期记忆。

它解决 Agent 记忆中的两类核心问题：**会话无状态**（新会话无法自然继承历史）和 **长期记忆不可控**（记忆被锁在黑盒数据库里，难以审查、迁移和修正）。

ReMe 的当前实现以 `reme/config/default.yaml` 为中心装配：启动 `Application` 后，默认暴露 HTTP 服务，后台监听 `daily/`、`digest/`、`resource/` 的变化，并提供检索、文件读写、daily note、自进化记忆等 Job。

<details>
<summary><b>你可以用 ReMe 做什么</b></summary>

<br>

- **个人助理**：把用户偏好、长期事实和历史上下文写入可读的 daily/digest 记忆。
- **编程助手**：沉淀项目约定、排错经验、操作流程和长期决策。
- **研究助理**：把报告、网页、日志、会议纪要等文本资源解读为 daily note。
- **知识图谱**：通过 Markdown、front matter 和 `[[wikilink]]` 维护本地知识网络。
- **主动记忆**：从 daily 输入中抽取兴趣主题，生成 `interests.yaml` 给上层 Agent 使用。
- **服务化工具**：通过 HTTP、MCP 或 CLI 调用同一组 Job。

</details>

---

## 📁 文件优先的记忆系统

> 记忆即文件，文件即记忆。

ReMe 的默认运行目录来自 `ApplicationConfig`。应用启动时会自动创建 vault 根目录和主要子目录。

```text
<vault_dir>/
├── reme_metadata/              # 索引、图谱、catalog 等持久状态
├── reme_session/               # Agent session 与原始对话
│   ├── dialog/
│   │   └── <session_id>.jsonl  # auto_memory 保存的对话消息
│   ├── agentscope/
│   └── claude_code/
├── resource/                   # 外部文本资源
│   └── YYYY-MM-DD/
│       └── <resource>.<ext>
├── daily/                      # 浅加工记忆
│   ├── YYYY-MM-DD.md           # 当天索引页
│   └── YYYY-MM-DD/
│       ├── <session_id>.md     # 对话或资源加工后的 daily note
│       └── interests.yaml      # auto_dream 产出的主动兴趣主题
└── digest/                     # 深加工记忆
    ├── personal/
    ├── procedure/
    └── wiki/
```

| 层级 | 内容 | 主要写入方 |
| --- | --- | --- |
| `resource/` | 外部文本材料，默认监听 `md/txt/json/jsonl/csv/yaml/html` | 手动同步、`resource_watch_loop` |
| `reme_session/dialog/` | 原始对话 JSONL | `auto_memory` |
| `daily/` | 当天索引页、对话笔记、资源解读、兴趣主题 | `daily_create`、`auto_memory`、`auto_resource`、`auto_dream` |
| `digest/personal/` | 用户画像、偏好、长期个人事实 | `auto_dream` |
| `digest/procedure/` | 方法论、流程、操作经验 | `auto_dream` |
| `digest/wiki/` | 通用知识、概念、决策先例 | `auto_dream` |

---

## 📝 Markdown、Front Matter 与 Wikilink

ReMe 的核心读写对象是 Markdown 文件。写入类 Job 会处理 front matter；索引时，front matter 会进入 `FileNode.front_matter`，供检索、图谱和 Agent 判断使用。

```markdown
---
name: 光伏产业链研究
description: 从硅料到组件的全链条梳理
tags: [新能源, 光伏, 产业链]
---
```

`WikilinkHandler` 是统一的 wikilink 解析和改写入口：

| 写法 | 示例 | 含义 |
| --- | --- | --- |
| 标准链接 | `[[digest/wiki/光伏.md]]` | 指向 vault-relative 目标 |
| 锚点链接 | `[[digest/wiki/钴.md#应用]]` | 指向目标章节 |
| 别名链接 | `[[digest/wiki/宁德时代.md\|宁德]]` | 显示别名，目标不变 |
| 嵌入引用 | `![[resource/2026-06-01/report.md]]` | 作为 wikilink 记录边 |
| 行级属性 | `industry:: [[digest/wiki/新能源.md]]` | 提取 predicate |
| 内联属性 | `[competitor:: [[digest/wiki/比亚迪.md]]]` | 提取 predicate |

当前实现采用**字面路径语义**：`[[X]]` 的 target 就是 `X`，不会自动补 `.md`，不会做 basename 搜索，也不会做 folder note 解析。推荐使用带扩展名的 vault-relative 路径。

---

## 🔍 索引、图谱与检索

默认 `file_store` 是 `local`，组合了：

| 子组件 | 默认后端 | 说明 |
| --- | --- | --- |
| `file_graph` | `local` | 维护文件节点、outlinks、inlinks 和 pending links |
| `keyword_index` | `bm25` | 基于 tokenizer 的全文检索 |
| `embedding_store` | 空字符串 | 代码支持向量检索，但默认配置未启用 |

因此默认行为是：**BM25 全文检索 + wikilink 图谱扩展**。如果配置了 `embedding_store`，`search_step` 会并行执行向量检索和 BM25，并用 RRF 融合结果。

### Markdown 分块

`MarkdownFileChunker` 使用 `mistletoe` 构建 Markdown AST，并按标题层级生成 chunk：

- 按 H1/H2/H3 等章节递归分块。
- 每个 chunk 带完整标题骨架，便于理解命中位置。
- 表格、代码块、列表会按结构拆分，过长片段会标记 `[Part X/N]`。
- 默认 `chunk_chars=10000`，`embed_toc=true`。
- front matter 和 wikilink 会在同一轮 chunk 中提取。

### Search

`search` Job 面向外部问答和检索：

1. 读取 `query`、`limit`、`min_score` 和可选 `search_filter`。
2. 并行调用 `vector_search` 与 `keyword_search`。
3. 默认未启用 embedding 时，实际返回 BM25 结果。
4. 如果两路都有结果，用 RRF 融合。
5. 对命中文件做 link expansion，返回 chunk 正文、行号、分数和出入链目录。

`node_search` 是 `auto_dream` 集成阶段使用的专用召回：只返回 `digest/` 下的节点，并附带 front matter 中的 `name` 和 `description`；它不返回正文，也不做 link expansion。

---

## 🧬 自进化记忆管线

ReMe 的自进化流程由默认 Job 组合完成：

```text
对话 messages ── auto_memory ──> daily/YYYY-MM-DD/<session_id>.md
外部 resource ── auto_resource ─> daily/YYYY-MM-DD/<resource_stem>.md
daily notes  ── auto_dream ────> digest/personal|procedure|wiki + interests.yaml
```

### Auto Memory

`auto_memory` 接收 `messages` 和可选 `session_id`：

1. 把输入标准化为 AgentScope `Msg`。
2. 如果有 `session_id`，保存到 `reme_session/dialog/<session_id>.jsonl`。
3. 调用 `daily_create` 创建或复用 daily note。
4. 通过 Agent 使用 `read`、`edit`、`frontmatter_update`、`write` 工具整理记忆。
5. 写入 `source_conversation` front matter，关联原始对话 JSONL。
6. 刷新当天索引页。

保存对话时会移除 base64 数据块，并把超长 tool result 截断到约 2KB。

### Auto Resource

`auto_resource` 处理 `resource/` 下的变更批次。默认后台 `resource_watch_loop` 监听：

```yaml
watch_dirs: [resource_dir]
watch_suffixes: [md, txt, json, jsonl, csv, yaml, html]
```

资源路径约定为：

```text
resource/YYYY-MM-DD/<filename>
```

`added/modified` 会读取资源文本，创建或更新同名 daily note，并让 Agent 解读内容；`deleted` 会删除对应 daily note、更新 file_store，并刷新当天索引页。

### Auto Dream

`auto_dream` 扫描指定日期的 daily 输入，将值得长期保存的内容整合进 `digest/`，并写出主动兴趣主题。

默认步骤：

```yaml
auto_dream:
  steps:
    - dream_extract_step
    - dream_integrate_step
    - dream_topics_step
    - dream_finish_step
```

| 阶段 | 实际行为 |
| --- | --- |
| Extract | 刷新当天索引页，比较 dream catalog，只处理 changed daily 输入，抽取 memory units 和 topics |
| Integrate | 对每个 unit 调用 Agent，使用 `node_search/read/frontmatter_read/write/edit/frontmatter_update` 整合进 digest |
| Topics | 写入 `daily/<date>/interests.yaml`，默认最多 3 个 topic，并参考过去 7 天去重 |
| Finish | checkpoint 成功处理的 changed paths，持久化 dream catalog，返回摘要 |

Extract 和 Integrate 需要可用 LLM；如果未配置 LLM，会返回失败。Topics 阶段在部分情况下可以退化为本地去重，但完整 `auto_dream` 仍依赖 LLM 完成抽取与整合。

### Proactive

`proactive` 读取：

```text
daily/<date>/interests.yaml
```

它返回 topics，并可按 `include_content` 返回 YAML 原文，供上层 Agent 获取当天值得主动关注的主题。

---

## ⚙️ 默认 Job

默认 Job 由 `reme/config/default.yaml` 注册。后台 Job 会随应用启动自动运行；普通 Job 会通过 HTTP/MCP/CLI 暴露。

| 类别 | Job | 说明 |
| --- | --- | --- |
| 后台索引 | `index_update_loop` | 监听 `daily/`、`digest/` Markdown 变更，增量更新 file_store |
| 后台资源 | `resource_watch_loop` | 监听 `resource/` 文本资源，更新 catalog 并触发 `auto_resource_step` |
| 后台 catalog | `digest_watch_loop` | 监听 `daily/`、`digest/`，更新 digest catalog |
| 系统 | `version`、`health_check`、`help` | 版本、健康检查、Job 列表 |
| 检索 | `search`、`node_search` | chunk 级检索、digest 节点召回 |
| 图谱 | `traverse` | 从指定 path 遍历 wikilink 图 |
| 索引维护 | `reindex` | 清空 file_store 并重建索引 |
| Daily | `daily_create`、`daily_list`、`daily_reindex` | 创建、列出、刷新 daily note |
| 文件读写 | `read`、`read_image`、`write`、`edit`、`delete`、`move`、`list`、`stat` | 操作 vault 内文件 |
| Front Matter | `frontmatter_read`、`frontmatter_update`、`frontmatter_delete` | 读取、合并更新、删除 front matter 字段 |
| 自进化 | `auto_memory`、`auto_resource`、`auto_dream` | 对话、资源、daily 输入的自动加工 |
| 主动记忆 | `proactive` | 读取 `interests.yaml` |

代码中还包含 `ingest`、`upload`、`download` 等 transfer step，但它们没有在默认配置中注册为 Job。

---

## 🚀 快速开始

### 安装

```bash
pip install reme-ai
```

从源码安装：

```bash
git clone https://github.com/agentscope-ai/ReMe.git
cd ReMe
pip install -e ".[full]"
```

### 启动服务

```bash
reme start
```

指定配置或覆盖参数：

```bash
reme start config=default service.port=8090
```

配置解析支持：

- 默认加载 `reme/config/default.yaml`。
- `config=<name-or-path>` 指定配置文件。
- dot notation 覆盖，如 `service.port=8090`。
- `${ENV_VAR:-default}` 环境变量展开。

### CLI 调用

CLI 入口在 `reme/reme.py`：

```bash
reme find_reme
reme version
reme health_check
reme search query="用户偏好" limit=5
reme daily_create date=2026-06-20
reme proactive date=2026-06-20
```

`reme start` 启动服务；`reme find_reme` 探测服务；其它 action 会作为 Job 名称，通过 client 调用已运行服务。

---

## 🌐 服务接口

### HTTP Service

默认服务后端是 FastAPI：

- 非 stream Job 注册为 `POST /<job.name>`。
- 请求体是 `Request`，响应是 `Response`。
- StreamJob 注册为 `POST /<job.name>`，返回 `text/event-stream`。
- CORS 默认开放。
- lifespan 中启动和关闭整个 `Application`。

### MCP Service

`MCPService` 使用 FastMCP：

- 非 stream Job 注册为 MCP tool。
- 支持 `stdio`、`sse`、`streamable-http`。
- StreamJob 当前不注册为 MCP tool。
- MCP 服务内置 `claim_channel` 相关通道机制，用于把 vault 变更通知发送给最近 claim 的客户端会话。

---

## 💾 持久化状态

除 vault 正文文件外，ReMe 会在 `reme_metadata/` 下保存组件状态：

| 组件 | 持久化内容 |
| --- | --- |
| `file_store` | `file_chunks_<name>_v1.jsonl.zst`，保存 chunk 元数据 |
| `file_graph` | `<name>.jsonl.zst`，保存 `FileNode` 和 links |
| `keyword_index` | `bm25_*.pkl`，保存 vocab、posting list、doc meta |
| `file_catalog` | `<catalog_name>.jsonl.zst`，保存已处理文件 mtime checkpoint |

应用关闭时会按启动顺序反向关闭组件，并触发 file_store、keyword index、file graph 和 catalog 的持久化。

---

## 🧩 关键数据模型

```text
Response
  success: bool
  answer: str
  metadata: dict

FileNode
  path: str
  st_mtime: float
  links: list[FileLink]
  chunk_ids: list[str]
  front_matter: FileFrontMatter

FileChunk
  id: str
  path: str
  start_line: int
  end_line: int
  text: str
  metadata: dict
  scores: dict[str, float]

FileLink
  source_path: str
  target_path: str
  target_anchor: str | None
  predicate: str | None
```

---

## 📄 License

This project is open-sourced under the Apache License 2.0. See [LICENSE](./LICENSE) for details.
