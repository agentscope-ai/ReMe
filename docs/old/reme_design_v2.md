# ReMe 设计文档

> 本文按 `reme/` 目录最新代码整理，重点描述当前实现，而不是历史设想。

## 整体定位

一句话总结：**面向 Agent 的、文件优先的自进化记忆系统**。

ReMe 把记忆落在一个可读、可编辑、可复制的 workspace 目录里，用 Markdown、front matter、wikilink、BM25 倒排索引和后台 Agent 管线，把原始材料逐步沉淀为可检索、可追溯、可演化的长期记忆。

核心原则：

- **文件即记忆**：长期状态主要是 workspace 下的文件和 `reme_metadata/` 中的索引快照。
- **Agent 可操作**：所有能力通过 Job 暴露，Agent 可以用 HTTP、MCP 或 Python 直接调用。
- **渐进加工**：对话和资源先进入 `daily/`，再由 `auto_dream` 提炼到 `digest/`。
- **Obsidian 兼容**：Markdown、YAML front matter、`[[wikilink]]`、Dataview 风格属性都按文本文件保存。

## 1. Workspace 与记忆分层

默认目录来自 `ApplicationConfig`：

```text
<workspace_dir>/
  reme_metadata/          # ReMe 索引、图谱、catalog 等持久状态
  reme_session/           # Agent session 与原始对话
    dialog/
      <session_id>.jsonl  # auto_memory 保存的对话消息
    agentscope/           # AgentScope wrapper session
    claude_code/          # Claude Code wrapper session
  resource/               # 外部原始材料
    YYYY-MM-DD/
      <resource>.<ext>
  daily/                  # 浅加工记忆
    YYYY-MM-DD.md         # 当天索引页
    YYYY-MM-DD/
      <session_id>.md     # 对话或资源加工后的 daily note
      interests.yaml      # auto_dream 产出的主动兴趣主题
  digest/                 # 深加工记忆
    personal/
    procedure/
    wiki/
```

分层含义：

| 层级 | 内容 | 主要写入方 | 说明 |
| --- | --- | --- | --- |
| `resource/` | 原始文本材料 | 手动、外部同步 | 当前 `auto_resource` 支持文本类资源读取：`md/txt/json/jsonl/csv/yaml/html` |
| `reme_session/dialog/` | 原始对话 JSONL | `auto_memory` | 对话消息按 `session_id` 去重、合并、持久化，并在 daily note front matter 中溯源 |
| `daily/` | 日记、资源解读、当天索引、兴趣主题 | `daily_create`、`auto_memory`、`auto_resource`、`auto_dream` | 浅加工层，保留当天发生的事实和材料 |
| `digest/personal/` | 用户画像、偏好、长期个人事实 | `auto_dream` | 深加工记忆桶之一 |
| `digest/procedure/` | 方法论、流程、操作经验 | `auto_dream` | 深加工记忆桶之一 |
| `digest/wiki/` | 通用知识、概念、决策先例 | `auto_dream` | 深加工记忆桶之一 |

启动时 `Application` 会确保 workspace 根目录和上述主要子目录存在。

## 2. Markdown 与图谱格式

### 2.1 Front Matter

Markdown 文件可带 YAML front matter：

```markdown
---
name: 光伏产业链研究
description: 从硅料到组件的全链条梳理
tags: [新能源, 光伏, 产业链]
---
```

当前 `FileFrontMatter` 约定 `name`、`description` 等字段；写入类 Job 会保留并合并 metadata。索引时，front matter 会进入 `FileNode.front_matter`，供 `node_search`、图展开和 Agent 判断使用。

### 2.2 Wikilink

`WikilinkHandler` 是系统唯一的 wikilink 解析和改写入口。支持：

| 写法 | 示例 | 含义 |
| --- | --- | --- |
| 标准链接 | `[[digest/wiki/光伏.md]]` | 指向 workspace-relative 目标 |
| 锚点链接 | `[[digest/wiki/钴.md#应用]]` | 指向目标章节 |
| 别名链接 | `[[digest/wiki/宁德时代.md\|宁德]]` | 显示别名，目标不变 |
| 嵌入引用 | `![[resource/2026-06-01/report.md]]` | 作为 wikilink 记录边 |
| 行级属性 | `industry:: [[digest/wiki/新能源.md]]` | 提取 predicate |
| 内联属性 | `[competitor:: [[digest/wiki/比亚迪.md]]]` | 提取 predicate |

当前实现采取**字面路径语义**：`[[X]]` 的 target 就是 `X`，不会自动补 `.md`，不会做 basename 搜索，也不会做 folder note 解析。推荐使用带扩展名的 workspace-relative 路径。

### 2.3 图谱边

Markdown chunker 会从正文提取 `FileLink`：

```text
source_path    # 源文件
target_path    # wikilink 里的字面目标
target_anchor  # # 后的锚点，可为空
predicate      # Dataview 风格关系名，可为空
```

`file_graph` 维护：

- 节点：`FileNode(path, st_mtime, links, chunk_ids, front_matter)`
- 正向边：文件里的 outlinks
- 反向边：谁指向当前节点
- pending 边：目标文件暂不存在时先保留为 virtual link，目标出现后自动提升为 real link

`move` 默认会调用 `WikilinkHandler.retarget_links`，把入边来源文件中的 `[[src]]` 字面链接改写为 `[[dst]]`；`delete` 会返回仍然存在的入边，提示调用方清理引用。

## 3. 语义分块与索引

### 3.1 Markdown AST 分块

`MarkdownFileChunker` 使用 `mistletoe` 构建 Markdown AST，再按标题层级折叠成树：

```text
Document AST
  -> MdNode root
    -> section H1
      -> body paragraph/list/table/code
      -> section H2
```

分块策略：

- 按 H1/H2/H3 等章节递归分块。
- 每个 chunk 默认包含完整标题骨架，检索命中后能看到片段在文档中的位置。
- 表格拆分时重复表头和分隔行。
- 代码块拆分时重复 fence opener/closer。
- 列表按 item 打包。
- 过长叶子节点按行或内部单元拆分，并加 `[Part X/N]`。
- `chunk_chars` 默认 10000，`embed_toc` 默认开启。

`DefaultFileChunker` 用于非 Markdown 的默认文本切块，默认配置中主要覆盖 `jsonl`。

### 3.2 FileStore 组合

`LocalFileStore` 是当前默认文件索引协调层，组合：

| 子组件 | 默认后端 | 功能 |
| --- | --- | --- |
| `file_graph` | `local` | 节点、wikilink 正反向图谱 |
| `keyword_index` | `bm25` | BM25 全文检索 |

默认 `reme/config/default.yaml` 中：

```yaml
file_store:
  default:
    backend: local
    keyword_index: default
    file_graph: default
```

因此最新默认行为是：**BM25 + 图谱**。

### 3.3 BM25 Index

`BM25Index` 是 numpy 实现的倒排索引：

- tokenizer 默认是 `regex`。
- 文档级 lazy delete，更新时先退休旧 doc slot，再追加新 slot。
- 持久化到 `reme_metadata/keyword_index/bm25_<name>_<tokenizer>_<fingerprint>_v1.pkl`。
- tokenizer 配置和 stopwords 指纹进入索引文件名，避免不同分词配置复用错误索引。

### 3.4 Search

`search` Job 当前实现：

1. 读取 query、limit、min_score、search_filter。
2. 默认配置下执行 `keyword_search`，返回 BM25 命中的 chunk。
3. 按 `min_score` 过滤并截断到 limit。
4. 对命中的唯一 path 做 link expansion，默认每个方向最多 10 条。
5. 返回 chunk 正文、行号、分数和出入链目录。

### 3.5 Node Search

`node_search` 是给 `auto_dream` Phase 2 使用的专用召回：

- 只返回 `digest/` 下节点。
- 以 path 聚合 chunk 结果，一篇 digest 只返回一行。
- 返回 path、score、front matter 中的 name/description。
- 不返回正文，不做 link expansion。
- 供集成 Agent 判断是 CREATE、CORROBORATE、REFINE 还是 CORRECT。

外部问答 Agent 应使用 `search`；dream 集成应使用 `node_search`。

## 4. 自进化管线

### 4.1 Auto Memory

`auto_memory` 输入对话 messages 和可选 `session_id`：

1. 把 messages 标准化为 AgentScope `Msg`。
2. 如有 `session_id`，保存到 `reme_session/dialog/<session_id>.jsonl`。
3. 调用 `daily_create` 创建或复用 `daily/<date>/<session_id>.md`，空 session 时使用 `daily/<date>.md`。
4. 通过 `agent_wrapper` 调用 LLM Agent，工具集为 `read`、`edit`、`frontmatter_update`、`write`。
5. 如果有 `session_id`，在 note front matter 写入 `source_conversation: [[reme_session/dialog/<session_id>.jsonl]]`。
6. 刷新当天索引页 `daily/<date>.md`。

保存对话时会去掉 base64 数据块，并截断超长 tool result，避免 session JSONL 过大。

### 4.2 Auto Resource

`auto_resource` 处理 `resource/` 下的变更批次。默认后台 `resource_watch_loop` 监听：

```yaml
watch_dirs: [resource_dir]
watch_suffixes: [md, txt, json, jsonl, csv, yaml, html]
```

资源路径约定为：

```text
resource/YYYY-MM-DD/<filename>
```

处理逻辑：

- `added/modified`：读取原始资源文本，创建或更新 `daily/YYYY-MM-DD/<resource_stem>.md`，再由 Agent 解读资源内容并写入 daily note。
- `deleted`：删除对应 daily note，更新 file_store，并刷新当天索引页。
- Agent session id 使用资源路径的 UUID5，保证同一资源重复处理时会话稳定。

### 4.3 Auto Dream

`auto_dream` 是最新代码中的四步 Job：

```yaml
auto_dream:
  steps:
    - dream_extract_step
    - dream_integrate_step
    - dream_topics_step
    - dream_finish_step
```

它的目标是扫描某天 daily 输入，把值得长期保留的抽象记忆写入 `digest/`，同时生成当天的 `interests.yaml`。

#### Phase 1: Extract

`dream_extract_step`：

- 刷新当天索引页。
- 扫描 `daily/<date>.md` 和 `daily/<date>/` 下文件，但排除 `interests.yaml`。
- 用 `file_catalog:dream` 对比 mtime，只处理 changed paths。
- 如果没有变化，直接结束。
- 调用 Agent 读取 changed material，输出：
  - `units`: 需要进入 digest 的抽象记忆单元。
  - `topics`: 主动兴趣主题候选。
- unit bucket 限定为 `procedure`、`personal`、`wiki`，未知 bucket 会路由到 `wiki`。

#### Phase 2: Integrate

`dream_integrate_step` 对每个 unit 独立调用 Agent：

- Agent 可用工具：`node_search`、`read`、`frontmatter_read`、`write`、`edit`、`frontmatter_update`。
- 先召回 digest 中可能相同或相关的节点。
- 决策结果是结构化 `IntegrateOutcome`：

| action | 含义 |
| --- | --- |
| `CREATE` | 新建 digest 节点 |
| `CORROBORATE` | 给已有节点追加佐证 |
| `REFINE` | 补充更精确的表述 |
| `CORRECT` | 修正旧记忆中的矛盾或过时内容 |

失败 unit 会记录到 `failed_units` 和 `failed_paths`，不会被 checkpoint，后续运行会重试。

#### Phase 3: Topics

`dream_topics_step` 写入：

```text
daily/<date>/interests.yaml
```

默认最多保留 3 个 topic，并参考过去 7 天的 `interests.yaml` 做去重，避免每天重复推送同类兴趣。若 LLM 不可用，会退化为本地去重选择。

#### Phase 4: Finish

`dream_finish_step`：

- 把成功处理的 changed paths、`interests.yaml` 和当天索引页写入 `file_catalog:dream`。
- 删除 catalog 中已经不存在的 daily 输入。
- 持久化 catalog 到 `reme_metadata/file_catalog/dream.jsonl.zst`。
- 返回本次扫描、抽取、集成、topic 和 checkpoint 的摘要。

### 4.4 Proactive

`proactive` 读取当天或指定日期的：

```text
daily/<date>/interests.yaml
```

返回 topics 和可选 YAML 原文，供调用方读取当天兴趣主题。

## 5. Job、Step 与组件架构

### 5.1 分层

```text
Service 层
  HTTP / MCP，把 Job 暴露为外部接口

Application 层
  加载配置，初始化 service、component、job，按依赖拓扑启动组件

Job 层
  BaseJob / StreamJob / BackgroundJob / CronJob，按 YAML 顺序执行 Step

Step 层
  默认 Job 使用的原子业务操作：file_io / index / evolve / common

Component 层
  可插拔基础设施：store、graph、index、catalog、LLM、agent wrapper、tokenizer
```

### 5.2 Registry 与依赖注入

所有后端通过全局注册表 `R` 注册：

```python
@R.register("local")
class LocalFileStore(BaseFileStore):
    ...
```

配置中的 `backend` 会通过 `(ComponentEnum, backend)` 找到类。组件依赖通过 `BaseComponent.bind(name, BaseClass)` 声明，`Application` 会按依赖拓扑顺序启动组件，并在关闭时反序关闭。

### 5.3 Job 类型

| Job 后端 | 类 | 行为 |
| --- | --- | --- |
| `base` | `BaseJob` | 请求触发，按步骤顺序执行，返回 `Response` |
| `stream` | `StreamJob` | SSE/流式输出 chunk |
| `background` | `BackgroundJob` | 应用启动后后台运行，失败时可 supervisor 重启 |
| `cron` | `CronJob` | 按 cron 表达式定时执行步骤 |

后台 Job 强制 `enable_serve=False`，不会暴露成 HTTP endpoint 或 MCP tool。

### 5.4 默认 Job 列表

默认配置中的主要 Job：

| 类别 | Job | 说明 |
| --- | --- | --- |
| 后台索引 | `index_update_loop` | 监听 `daily/`、`digest/` 的 Markdown 变更，增量更新 file_store |
| 后台资源 | `resource_watch_loop` | 监听 `resource/` 文本资源，更新 resource catalog 并触发 `auto_resource_step` |
| 后台 catalog | `digest_watch_loop` | 监听 `daily/`、`digest/`，更新 digest catalog 并记录变更 |
| 系统 | `version` | 返回包版本 |
| 系统 | `health_check` | 返回组件健康快照 |
| 系统 | `help` | 列出已注册 Job |
| 检索 | `search` | chunk 级 BM25 检索和 link expansion |
| 检索 | `node_search` | digest 节点级召回，供 dream 集成使用 |
| 图谱 | `traverse` | 从指定 path 遍历 wikilink 图 |
| 索引维护 | `reindex` | 清空 file_store 并从文件重新建索引 |
| 日记 | `daily_create` | 幂等创建当天 day-level 或 session-level note |
| 日记 | `daily_list` | 列出某天 daily notes |
| 日记 | `daily_reindex` | 重建当天索引页 |
| 文件读写 | `read` | 读取 workspace 内 Markdown 文件，可指定行号 |
| 文件读写 | `read_image` | 读取图片为 base64，默认上限 5MB |
| 文件读写 | `write` | 写 Markdown 文件和 front matter |
| 文件读写 | `edit` | 全量 find-and-replace |
| 文件读写 | `delete` | 删除文件或目录，返回残留入边 |
| 文件读写 | `move` | 移动或重命名文件，默认改写入边 wikilink |
| 文件读写 | `list` | 列目录 |
| 文件读写 | `stat` | 返回路径元信息 |
| Front Matter | `frontmatter_read` | 读取 front matter |
| Front Matter | `frontmatter_update` | 合并更新 front matter |
| Front Matter | `frontmatter_delete` | 删除 front matter 字段 |
| 自进化 | `auto_memory` | 对话写入 daily note |
| 自进化 | `auto_resource` | 资源文件解读为 daily note |
| 自进化 | `auto_dream` | daily -> digest + interests.yaml |
| 主动记忆 | `proactive` | 读取 `interests.yaml` |

## 6. 服务、客户端与 CLI

### 6.1 HTTP Service

`HttpService` 使用 FastAPI：

- 非 stream Job 注册为 `POST /<job.name>`，请求体是 `Request`，响应是 `Response`。
- StreamJob 注册为 `POST /<job.name>`，返回 `text/event-stream`。
- CORS 默认开放。
- lifespan 中启动/关闭整个 `Application`。

### 6.2 MCP Service

`MCPService` 使用 FastMCP：

- 非 stream Job 注册为 MCP tool。
- 支持 `stdio`、`sse`、`streamable-http` 等 transport。
- StreamJob 当前不注册为 MCP tool。

### 6.3 Client 与 CLI

入口是：

```bash
reme start
reme find_reme
reme <job_name> key=value ...
```

行为：

- `reme start`：加载 `.env`，解析配置，启动服务。
- `reme find_reme`：从环境或默认地址探活。
- 其他 action：通过 client 调用已运行服务，默认 HTTP，也可指定 `backend=mcp`。

配置解析支持：

- 默认加载 `reme/config/default.yaml`。
- `config=<name-or-path>` 指定配置文件。
- dot notation 覆盖，如 `service.port=8090`。
- `${ENV_VAR:-default}` 环境变量展开。

服务启动后会把地址写到环境变量 `REME_SERVICE_INFO`，HTTP client 会优先使用显式 host/port，其次使用该环境变量，最后回落到默认 host/port。

## 7. 默认组件后端

默认配置中的组件：

| ComponentEnum | 名称 | 后端 | 说明 |
| --- | --- | --- | --- |
| `service` | - | `http` | 默认服务协议 |
| `tokenizer` | `default` | `regex` | BM25 分词 |
| `as_llm` | `default` | `${LLM_BACKEND:-openai}` | OpenAI 兼容 LLM，默认模型 `qwen3.7-plus` |
| `agent_wrapper` | `default` | `agentscope` | AgentScope ReAct wrapper |
| `agent_wrapper` | `claude_code` | `claude_code` | Claude Code wrapper |
| `file_graph` | `default` | `local` | 纯 Python 图谱 |
| `file_catalog` | `default/resource/digest/dream` | `local` | JSONL.zst catalog |
| `file_chunker` | `markdown` | `markdown` | Markdown AST chunker |
| `file_chunker` | `default` | `default` | 默认文本 chunker |
| `keyword_index` | `default` | `bm25` | numpy BM25 |
| `file_store` | `default` | `local` | graph + keyword |

## 8. 持久化状态

除 workspace 正文文件外，ReMe 会在 `reme_metadata/` 下保存组件状态：

| 组件 | 持久化内容 |
| --- | --- |
| `file_store` | `file_chunks_<name>_v1.jsonl.zst`，保存 chunk 元数据 |
| `file_graph` | `<name>.jsonl.zst`，保存 `FileNode` 和 links |
| `keyword_index` | `bm25_*.pkl`，保存 vocab、posting list、doc meta |
| `file_catalog` | `<catalog_name>.jsonl.zst`，保存已处理文件 mtime checkpoint |

`Application.close()` 会反序关闭组件，`LocalFileStore.close()` 会触发 chunk、keyword index、file graph dump。后台 catalog/dream finish 也会按需 dump catalog。

## 9. 关键数据模型

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
  id: str                  # hash(path, start_line, end_line, text)
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

DreamState
  date / changed_paths / unchanged_paths / deleted_paths
  units / topics
  integrate_results / failed_units / failed_paths
  interests_path / topics_written
  checkpoint_paths / errors / summary
```
