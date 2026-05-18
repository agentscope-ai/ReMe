# 🧠 AI 记忆系统

## 核心架构理念:Memory as Code (记忆即代码)

1. **文件即真理 (SSOT):** 物理存在的 Markdown 文件(及其 YAML 头部)是系统中**唯一**存储记忆原始状态的地方。
2. **只读投影 (Read-only Projections):** 图数据库 (Graph DB) 和向量数据库 (Vector DB) 不再是独立存储介质,而是文件系统的"下游索引/缓存",完全由文件 `Change` 事件被动驱动刷新。

> **状态约定:** 各章节标题后括号 `[已实现 / 部分实现 / 未实现]` 反映 reme4 当前代码。未实现部分保留设计原文,后续讨论再固化。

---

## 全局架构蓝图 (Global Architecture Blueprint)

系统按职责自上而下分 **5 层** + **1 跨切面 (Schema)**:

| Layer | 模块 | 角色 | 是否含 LLM |
|---|---|---|---|
| 1 | Agent (Claude Code) | 上层应用,持 Hot Context / Warm Summary / Loaded Memory | LLM (主) |
| 2 | Plugin / MCP 接口层 | reme-service 与 reme-expert 两个 tier 暴露 MCP 工具 + 钩子 + 子代理 | (子代理含 LLM) |
| 3 | 服务层 (`reme4/steps/jobs/`) | LLM-driven 复合工作流:Synchronizer / Distiller / Maintainer [未] | LLM (内部 ReAct) |
| 4 | 原子工具层 (`reme4/steps/{crud,property,tags,lint,graph,daily,common}/`) | 无 LLM 的纯文件/索引/工作区操作,**被 Layer 2 / Layer 3 / Layer 2 子代理三方共享** | — |
| 5 | 核心引擎 (`reme4/components/`) | file_store (MFS) + watcher + parser + 投影 (vector/keyword/graph),域无知 | — |
| ▭ | Schema (`reme4/steps/jobs/protocol.md` + `lint/schema.py` + `schema/file_front_matter.py`) | 跨切面契约:4 轴 + R-M-W 决策树 + 后置 validator + 开放 dict 模型 | — |

```plain
==============================================================================
                  【 Layer 1: Agent (Claude Code / 上层应用) 】
==============================================================================
   [ Session 工作台 ]
   - Hot Context: 原始对话滑动窗口
   - Warm Summary: Agent 后台维护的认知进度
   - Loaded Memory: 检索注入的底层快照
                            ↕  通过 MCP 唯一接触面
==============================================================================
              【 Layer 2: Plugin / MCP 接口层 (双 tier) 】
==============================================================================
  ┌──── reme-service ─────────────┐  ┌──── reme-expert ────────────────┐
  │ 26 MCP tools:                  │  │ 24 MCP tools (仅原子,无 L3):    │
  │   ┌ [L3] synchronizer          │  │                                 │
  │   └ [L3] distiller             │  │                                 │
  │   ├ [L4] 24 shared atomic      │  │   ┌ [L4] 24 shared atomic       │
  │   └                            │  │   └                             │
  │                                │  │                                 │
  │ Hooks: PreCompact / SessionEnd │  │ Hooks: 同左                     │
  │        Stop (active_daily_check)│ │                                 │
  │                                │  │ Subagents (LLM 跑在 Claude Code,│
  │ (无 subagent / slash)          │  │  通过 MCP 调 L4):                │
  │                                │  │   reme-distiller (R-M-W loop)   │
  │                                │  │   reme-curator   (lint sweep)   │
  │                                │  │                                 │
  │                                │  │ Slash: /reme-distill            │
  │                                │  │         /reme-recall            │
  │                                │  │         /reme-clean             │
  └────────────────────────────────┘  └─────────────────────────────────┘
         │ MCP 直调 L3 + L4               │ subagent 通过 MCP 调 L4
         ▼                                ▼
==============================================================================
       【 Layer 3: 服务层 (reme4/steps/jobs/) — LLM-driven 复合工作流 】
==============================================================================
  ┌─────────────────┐  ┌─────────────────┐  ┌────────────────────────┐
  │ Synchronizer    │  │ Distiller       │  │ Maintainer  [未实现]   │
  │ (热写 / Log)    │  │ (冷写 / Distill)│  │ (治理:Decay/Merge/Split)│
  │ ─────────       │  │ ─────────       │  │ ─────────              │
  │ in: messages    │  │ in: daily_paths │  │ in: cron / threshold   │
  │ 内部 ReActAgent │  │ 内部 ReActAgent │  │ out: 覆写 MFS          │
  │ + toolkit       │  │ + toolkit       │  │                        │
  │ → daily/<date>/ │  │ → knowledge/    │  │                        │
  │   <slug>/.md    │  │   <slug>/.md    │  │                        │
  │                 │  │ + status flip   │  │                        │
  │                 │  │ + lint:schema   │  │                        │
  │                 │  │   自查           │  │                        │
  └─────────────────┘  └─────────────────┘  └────────────────────────┘
         │ ReActAgent 内部 toolkit 调 L4      │ (未来)
         ▼                                    ▼
==============================================================================
       【 Layer 4: 原子工具层 (reme4/steps/{crud,property,tags,lint,...}/) 】
                            无 LLM,纯文件 / 索引操作
==============================================================================
  ┌──────────┬──────────────┬──────────────┬──────────┬──────────────────┐
  │ Retrieve │ Read         │ Write        │ Tags     │ Lint             │
  │ ──────── │ ──────────   │ ──────────   │ ──────── │ ──────────       │
  │ search   │ list         │ write        │ tags_list│ lint_dangling    │
  │ graph_   │ read         │ property_    │ tags_stat│ lint_orphans     │
  │  traverse│ stat         │   update     │          │ lint_collisions  │
  │          │ property_read│ property_    │          │ lint_schema      │
  │          │              │   delete     │          │                  │
  ├──────────┴──────────────┴──────────────┴──────────┴──────────────────┤
  │ File ops:  move    copy    delete    upload    download              │
  │ Daily:     daily:read    daily:list    daily:write    daily:status    │
  └──────────────────────────────────────────────────────────────────────┘
         │ 所有原子工具走 file_store / file_graph / indexes
         ▼
==============================================================================
       【 Layer 5: 核心引擎 (reme4/components/) — 域无知 】
==============================================================================
  ┌──────────────┐   ┌──────────────────┐   ┌────────────────────────┐
  │ file_store   │ → │ file_watcher +    │ → │ Projections           │
  │ (MFS, SSOT)  │   │ file_parser       │   │ (只读,可整体重建)      │
  │ ─────────    │   │ ─────────         │   │ ─────────              │
  │ Local/...    │   │ lite watcher      │   │ embedding_model        │
  │ markdown +   │   │  + md parser      │   │   (向量索引)            │
  │ YAML +       │   │ AST 分块 + hash   │   │ keyword_index           │
  │ wikilinks    │   │ diff → 仅 dirty   │   │   (BM25 + tokenizer)    │
  │ 接受所有写入  │   │  block 触发投影   │   │ file_graph              │
  │              │   │ 快路线 [√]        │   │   (Local/Nx/Neo4j)      │
  │              │   │ 慢 LLM IE [未]    │   │                        │
  └──────────────┘   └──────────────────┘   └────────────────────────┘

══════════════════════════════════════════════════════════════════════════════
            ◄═══════ Schema (跨切面契约,非"层") ═══════►
══════════════════════════════════════════════════════════════════════════════
  reme4/steps/jobs/protocol.md          — 4 轴 (lifecycle/scope/source/role)
                                          + R-M-W 决策树 + wikilink 约定 (单一源)
         │
         ├── Layer 2 transclude:  plugin SKILL.md  /  subagent agent.md
         ├── Layer 3 inject:      distiller.yaml 内 {protocol} 占位符
         └── Layer 4 enforce:     lint:schema 后置校验 (必填 4 key + status enum)

  reme4/schema/file_front_matter.py     — Pydantic 开放 dict 模型 (extra=allow)
  Python preset 类 (EVENT_PRESET / ...)  [未实现] — 等 vault 域稳定再固化
```

---

## 层间数据流 (Read Path / Write Path)

```
写路径 (例: Distill phase, service tier)
─────────────────────────────────────────
  L1 Agent
    └─ MCP call distiller(daily_paths=[...])               [Layer 2 入口]
        └─ Layer 3 Distiller.execute
            └─ 内部 ReActAgent 通过 toolkit
                ├─ L4 read / list / graph_traverse  (调研)
                ├─ L4 write / property_update       (落盘)
                └─ L4 lint_schema                   (自查)
                    └─ Layer 5 file_store          (MFS 物理写)
                        └─ Layer 5 file_watcher    (扫到 mtime/hash 变化)
                            └─ Layer 5 投影刷新     (vector/keyword/graph)

读路径 (例: Recall phase)
─────────────────────────────────────────
  L1 Agent
    └─ MCP call search(query=...) / graph_traverse(path=...) [Layer 2 入口]
        └─ Layer 4 search_step / graph:traverse
            └─ Layer 5 Projections (查 embedding + BM25 + file_graph)
                ▲
                │ 原文按需 crud:read 回 Layer 5 file_store
                │
        ◄─ FileChunk[] 返回 Agent
```

**关键不变量:** Agent 永远只看到 Layer 2 (MCP);Layer 5 永远只看到文件;Layer 3 和 Layer 4 之间用相同的 toolkit 抽象 — Layer 3 是 LLM 编排,Layer 4 是 LLM 自动机里的"动作"。

---

## 核心约束

1. **MFS 即真理 (Layer 5 不可绕过)** — Vector / Keyword / File Graph 都是 Layer 5 的下游投影,可整体重建;任何写都必须先落 MFS,再由 Watcher 异步重建投影。
2. **引擎域无知 (Layer 5 不依赖 Schema)** — frontmatter 在 Layer 5 永远是开放 dict (`FileFrontMatter.model_config = extra="allow"`)。换 vault 形态 (代码笔记 / 日记 / 文献库) = 换 Schema 协议 + Layer 3 服务,Layer 5 一行不动。
3. **Schema 是单一权威 (跨切面)** — 4 轴在 protocol.md 文字约定;新增记忆形态 = 在 protocol.md 加新组合,行为代码 (`lint:schema` / Distiller 决策树) 从轴上读取。Python preset 类未实现,以文字约定先行。
4. **Layer 4 是共享子结构** — 同一批原子工具 (`search` / `graph_traverse` / `crud:*` / `property:*` / `tags:*` / `lint:*`) 被三方消费:Layer 3 ReAct 内部 toolkit、Layer 2 MCP 直接暴露、Layer 2 expert 子代理通过 MCP。**不存在为某一方独立实现**。
5. **Layer 2 双 tier 分裂只在 Layer 3 暴露与否** — reme-service 把 Layer 3 当 MCP 工具直接暴露 (LLM 跑在 reme4),reme-expert 不暴露 Layer 3,改用子代理跑同一套 Layer 4 (LLM 跑在 Claude Code)。Layer 4 / 5 / Schema 完全相同。

---

## Layer 详解

### Layer 1: Agent (Claude Code) [外部]

主 LLM 应用,持有会话工作台。reme 不感知其内部状态;仅通过 Layer 2 MCP 接收读/写请求。

会话工作台三种内存槽:
- **Hot Context** — 原始对话滑动窗口 (主 LLM 自管)
- **Warm Summary** — Agent 自行维护的进度摘要 (主 LLM 自管)
- **Loaded Memory** — 通过 Layer 2 检索回填的底层快照 (frontmatter + graph edges + blocks)

---

### Layer 2: Plugin / MCP 接口层 [已实现,双 tier]

唯一 Agent 接触面。两个 tier 共享 Layer 4,只差是否暴露 Layer 3 + 是否带子代理:

| 维度 | reme-service | reme-expert |
|---|---|---|
| MCP 工具数 | 26 (2 L3 + 24 L4) | 24 (仅 L4) |
| Layer 3 暴露 | ✅ synchronizer + distiller | ❌ |
| 子代理 | (无) | reme-distiller / reme-curator |
| Slash | (无) | /reme-distill /reme-recall /reme-clean |
| Hooks | PreCompact / SessionEnd / Stop | 同左 |
| LLM 跑在 | reme4 (Layer 3 内部 ReActAgent) | Claude Code (子代理) |
| 认知负担 | 低 (复合工作流交给 L3) | 高 (子代理用 L4 拼装 R-M-W) |

详见 `reme-plugin/README.md` 与各 plugin 的 SKILL.md。

---

### Layer 3: 服务层 (reme4/steps/jobs/) [部分实现]

LLM-driven 复合工作流。每个 job 在内部 `ReActAgent` 里编排 Layer 4 原子工具,完成单次"高语义"操作。

#### Synchronizer (热写 / Log phase) [已实现]

- **输入:** `messages: list[Msg|dict]`, `workspace: str?`
- **职责:** 从对话切片中决定 slug、写 `daily/<YYYY-MM-DD>/<slug>/<slug>.md` summary note + sibling materials;同 slug 复用 = upsert 累积。
- **关键不变量:** 不写 `knowledge/`;不调 `lint:schema` 自查 (热路径,延迟敏感)。

#### Distiller (冷写 / Distill phase) [已实现]

- **输入:** `daily_paths: list[str]`, `hint: str?`, `flip_status: bool = True`
- **职责:** 把 daily 蒸馏成 knowledge node,完成 R-M-W 闭环。详见下一节。

#### Maintainer (治理) [未实现]

整服务尚未建模。设计意图见 [后台自演化算法](#后台自演化算法-maintenance--evolution-未实现) 章节。占位的物理路径已统一:写入仍走 Layer 5 file_store,行为对齐 Layer 3 其他 job 的 ReAct 模式。

---

### Layer 4: 原子工具层 (reme4/steps/{crud,property,tags,lint,graph,daily,common}/) [已实现]

无 LLM,纯文件/索引操作。所有 step 经 `R.register(...)` 注册,可被 Layer 2 暴露为 MCP 工具、被 Layer 3 ReActAgent 内部 toolkit 调用、或被 Layer 2 子代理通过 MCP 调用 — **同一份实现,三方共享**。

| 类 | 注册名 | 主要 step |
|---|---|---|
| Retrieve | `search_step` `graph:traverse` | RRF 混合检索 / 1-hop 图扩展 |
| Read | `list` `read` `stat` `property:read` | 文件读 + 属性读 |
| Tags | `tags:list` `tags:stat` | 标签聚合 / 按标签拉文件 |
| Write | `write` `property:update` `property:delete` | 文件写 + 属性更新 |
| File ops | `move` `copy` `delete` `upload` `download` | 文件搬移 / 复制 / 删除 + vault ↔ 本地 fs 双向传输 |
| Daily | `daily:read` `daily:list` `daily:write` `daily:status` | 工作区 CRUDS:单个 workspace 一次读完 / 按日期+状态列表 / upsert 写入 (默认 frontmatter 自动) / 状态机迁移 |
| Lint | `lint:dangling` `lint:orphans` `lint:collisions` `lint:schema` | 4 种诊断 |

详见 [检索与召回算法](#检索与召回算法-retrieval--recall) 章节 (search + graph_traverse 的现状与未实现 RAG pipeline)。

---

### Layer 5: 核心引擎 (reme4/components/) [已实现]

域无知。任意 markdown vault 都能跑,不知道 Daily / Knowledge 是什么。

```
file_store (MFS)  →  file_watcher + file_parser  →  Projections
─────────────       ─────────────────────────────    ──────────
Local/...           lite watcher                     embedding_model (vector)
markdown + YAML     md parser (wikilink/Dataview)    keyword_index (BM25)
SSOT, 接受所有写    AST 分块 + hash diff             file_graph (Local/Nx/Neo4j)
                    快路线 [√] / 慢 LLM IE [未]
```

详见 [存储与投影算法](#存储与投影算法-storage--projection) 章节。

---

### Schema (跨切面) [部分实现]

横跨 Layer 2 / 3 / 4 的契约。文字优先,Python 类后置。

| 资产 | 路径 | 角色 |
|---|---|---|
| `protocol.md` (单一源) | `reme4/steps/jobs/protocol.md` | 4 轴 + R-M-W 决策树 + wikilink 约定 |
| Pydantic 模型 | `reme4/schema/file_front_matter.py` | 开放 dict (`extra=allow`),仅 title/description/tags 类型化 |
| 后置 validator | `reme4/steps/lint/schema.py` | 必填 4 key (`title/lifecycle/scope/source/role`) + status enum |
| Python preset 类 [未实现] | — | EVENT_PRESET / PROFILE_PRESET 等;等 vault 域稳定再固化 |

`protocol.md` 三方消费:
- **Layer 2 transclude:** plugin SKILL.md 与 subagent agent.md 用 `@../../protocol.md`
- **Layer 3 inject:** `reme4/steps/jobs/distiller.yaml` 把它注入 `{protocol}` 占位符 → 进 ReActAgent sys_prompt
- **Layer 4 enforce:** `lint:schema` 在写后扫描,Distiller 自调修复

**4 个驱动行为的轴:**

| key | 取值 |
|---|---|
| `lifecycle` | `streaming` (会衰减,daily) / `evolving` (持续编辑,knowledge) / `frozen` (不可改) |
| `scope` | `instance` / `class` |
| `source` | `auto` (机器) / `curated` (人/LLM) / `derived` (推算) |
| `role` | `observation` / `claim` / `question` / `profile` / `concept` / `method` / `reference` / `fundamentals` |

**Role-conditional 字段 (protocol.md 约定,目前 lint:schema 未全部强制):**
- `role=claim` → `confidence` (✅ / ⏳ / ❌) 推荐
- `lifecycle=streaming` → `status` (active / distilled / archived) 强制
- `source=auto` → `originSessionId` [未实现]

---

## Distiller 的 R-M-W 认知循环 [已实现]

> 本节为 Layer 3 中 Distiller 的实际行为。Distiller 在 reme4 中曾叫 Ingestor,已重命名以对齐"知识蒸馏"语义。

Distiller 接收到 `daily_paths` 后,不是直接写文件,而是严格执行一个**闭环的 LLM 推理过程**:

### 读阶段 (Read - 上下文感知)
- Distiller 拿到 `daily_paths` (由 caller 显式传入,**不自动扫描**全 vault)。
- 对每个 `daily/<date>/<slug>/`,优先读 summary note (`<slug>.md`);material 文件 (pdf/csv/...) 由 LLM 按需 read。
- **不预扫现有 knowledge** — LLM 在 Modify 阶段按需用 Layer 4 的 `list` / `graph_traverse` / `tags_list` 查找相关知识节点。

### 修阶段 (Modify - LLM 认知裁决)
- Distiller 将【打包好的 daily 内容】+【protocol.md】+【caller hint】一并提交给 ReActAgent。
- **LLM 的核心任务:状态对齐与消除冗余**
    - **发现冲突:** `knowledge/项目X.md` 写着"使用 JS 开发",新 daily 说"用 TS 重构"。LLM 覆写该状态,而不是在文件末尾追加矛盾的话。
    - **合并同类项:** `knowledge/张三/张三.md` 有"擅长前端",LLM 将新事件提炼为"主导项目X的 TS 重构",归入"项目经验"段落下。
- **决策树:**
    - no hit            → CREATE `knowledge/<slug>/<slug>.md`
    - exact role match  → UPDATE: read → merge → write overwrite=True
    - role mismatch     → CREATE 新 knowledge node,与原节点 cross-link
    - 仅关系值得记录    → LINK: 追加 wikilink 到现有 knowledge body

### 写阶段 (Write - 物理落盘)
- LLM 直接调 Layer 4 的 `write` / `property_update` 工具,逐条落盘。
- **写后自查:** Distiller 让 LLM 主动调 `lint:schema` 拿违规清单,用 `property:update` 修。Distiller 本身**不内嵌** Schema validator。
- **闭环 flip:** 所有 writes 成功的 daily,LLM 调 `property:update status=distilled` 翻状态。
- 落盘完成,触发 Layer 5 的 Watcher 去更新向量和图谱投影。

### 与 Synchronizer (热写) 的分工

| 维度 | Synchronizer (热写 / Log) | Distiller (冷写 / Distill) |
|---|---|---|
| 触发 | 每次会话 / 任务结束 | 显式调用,batch 处理 |
| 输入 | 对话 messages | `daily_paths: list[str]` |
| 输出 | `daily/<YYYY-MM-DD>/<slug>/` workspace | `knowledge/<slug>/` knowledge node |
| 改 daily? | 创建 / 更新 daily summary | 只翻 `status: distilled` |
| Schema 校验 | 无 (待补 pre-write hook) | LLM 自查 `lint:schema` |

---

## 存储与投影算法 (Storage & Projection)

> 对应 **Layer 5** 实现细节。被动投影,核心算法的重点在 **File Parser** 上。

### 向量的增量投影 (Incremental Embedding) [已实现]
+ **AST Markdown 语义分块:**

当文件变更时,Parser 通过解析 Markdown 抽象语法树 (AST),以标题 `##`、段落 `\n\n` 为界提取 Block。

+ **Hash Diff 计算算法:**
    1. 系统为文件的每一个 Block 计算哈希值 (如 MD5)。
    2. 对比变更前后文件的 Block Hash 列表。
    3. **仅对 Hash 发生改变或新增的 Block (Dirty Blocks)** 调用 Embedding 模型。
    4. 将新向量同步至 Vector DB,记录元数据 `{"file_id": ".../张三.md", "block_id": "b_123"}`。若某 Hash 消失,则发送 `DELETE` 指令清理 Vector DB 中的孤儿向量。

### 图谱关系的双态抽取 (Dual-state Relation Extraction)
文件更新后,必须提取概念间的关联以更新 Graph DB。分为快慢两条路线:

+ **快路线 (显式抽取):基于规则的解析** [已实现]
    - **触发:** 文件中包含标准双链语法 (`md` parser 支持三种形式):
        - bare:                   `See [[张三]]`
        - line-level Dataview:    `colleague:: [[李四]]`
        - inline-bracketed:       `主导 [负责:: [[项目X]]] 的重构`
    - **算法:** Parser 直接使用正则提取 `FileLink(source_path, target_path, target_anchor, predicate)`,写入 `file_graph` 投影 (LocalFileGraph / NxFileGraph / Neo4jFileGraph 任选)。延迟接近 0。
+ **慢路线 (隐式抽取):基于 LLM 的 IE (信息抽取)** [未实现]
    - **触发 (设计意图):** 文件新增了自然语言段落 (如 daily summary 写入的:"2026-05: 张三与李四发生技术冲突")。
    - **算法 (设计意图):**
        1. 提取器截取该新增 Block,传入轻量级本地 LLM (如 Llama-3-8B)。
        2. 强制 JSON 输出提取的三元组。
        3. **实体对齐 (Entity Resolution):** 将字符串 "李四" 通过极速检索匹配到 `knowledge/li-si/li-si.md`。
        4. **物理回写 (Write-back 关键步):** 为了保证 SSOT,后台将提取出的隐式关系转换为隐形元数据或双链语法,**追加回 Markdown 文件末尾**,随后再同步给 Graph DB。
    - 当前由 Distiller 在 LLM agent 中显式产出 wikilink 代替;独立的"隐式抽取 + write-back"后台进程未建。

---

## 后台自演化算法 (Maintenance & Evolution) [未实现]

> 对应 **Layer 3 Maintainer**。整服务尚未在 reme4 中建模,下文为设计意图。

记忆不能无限膨胀,系统通过独立的后台守护进程 (The Maintainer) 执行拓扑运算,由两种信号唤醒:

+ **周期性 Cron Job (定时):** 例如每天凌晨 2 点,扫描全局节点计算衰减公式 (Decay),或者计算全局 Embedding 相似度矩阵寻找可以合并 (Merge) 的节点。
+ **Watcher 警报 (阈值):** 当 Parser 在处理文件时,发现 `张三.md` 已经超过了 8000 Tokens。Watcher 会向 Maintainer 发送一个"超载预警"。

### 实体消歧与合并 (Hierarchical Clustering Merge) [未实现]
每天夜间低峰期,系统自动清理冗余概念文件 (如 `AI.md` 和 `人工智能.md`)。

+ **算法:**
    1. 提取所有文件 YAML 头部的全局摘要 Embedding。
    2. 计算全量节点对的余弦相似度矩阵。
    3. 使用 **Union-Find (并查集)** 算法,以 $Similarity > 0.95$ 为阈值,找出所有连通分量 (疑似同义词集合)。
    4. 触发物理合并:将较新文件的内容 Append 到较早创建的主文件中,较新文件转化为 Symlink (软链接)。
    5. 触发文件修改事件 -> Watcher 自动更新 Graph DB,将所有指向旧文件的边重定向至主文件。

### 记忆节点的细胞分裂 (K-Means Split) [未实现]
防止单一概念文件成为包含几万字、语义混杂的"大垃圾桶"。

+ **算法:**
    1. 监控指标:当单一 `.md` 文件的总 Token 数或内部 Block 数量超过设定阈值 $T_{max}$。
    2. **特征提取:** 取出该文件内所有 Block 的 Embedding。
    3. **自适应聚类:** 运行 K-Means 聚类,通过计算**轮廓系数 (Silhouette Coefficient)** 自动寻找最优的分类数 $K$。
    4. 调用 LLM 为这 $K$ 个簇生成新的文件名 (如从 `编程.md` 拆分为 `前端.md` 和 `后端.md`)。
    5. 执行物理拆分,创建新文件。Watcher 随后根据每个 Block 的新归属,自动重构 Graph 中的连线。

### 拓扑与时间联合遗忘算法 (Topo-Temporal Decay) [未实现]
冷门无用的临时记忆必须被清理,但不纯依赖时间。

+ **算法计算公式:**

$$Score_i = (I_i \times e^{-\lambda \Delta t}) + \alpha \log(1 + D_{in}^{(i)})$$

- $I_i$: 节点的初始重要性 (1-10 分)。
- $\Delta t$: 距离最后一次被 Agent 访问的天数。
- $\lambda$: 遗忘速率常数。
- $D_{in}^{(i)}$: 该节点在 Graph DB 中的入度 (被引用的次数)。
- $\alpha$: 图拓扑权重系数。
+ **执行逻辑:** 当 $Score_i < 阈值$ 时,不执行硬删除,而是修改文件的 YAML 头 `status: archived`,并将其移动至 `/Archive/` 目录。Graph DB 随即挂起 (Suspend) 其所有相关连线。

> 现状:`lint:schema` 中已经接受 `status: archived` 作为合法值,但触发归档的 Decay 逻辑未建。

---

## 检索与召回算法 (Retrieval & Recall)

> 对应 **Layer 4 Retrieve 类** + 设计意图中的统一 Retriever。Retriever 作为统一类未建,reme4 当前以 `search_step` (RRF 融合) + `graph:traverse` (1-hop BFS) + `crud:read` 等 Layer 4 原子手动组合。意图路由 / Rerank pipeline 未建。

### 意图路由 (Semantic Soft Routing — Retriever 内部子策略) [未实现]
避免每次检索前都调 LLM 算意图。

+ **机制 (设计意图):** 在向量空间中预埋 3 个核心意图锚点向量 ($V_{实体}, V_{事实}, V_{关系}$)。
+ **算法 (设计意图):** 用户的 Query 转换为向量 $V_q$,计算其与 3 个锚点的余弦距离,基于最近邻 (KNN) 确定检索策略。
+ **位置:** 跑在 Retriever 入口处,决定走哪条召回管线 (实体定位 / 三路融合 / 图谱扩展)。
+ **现状:** 调用方 (主 Agent) 显式选 `search` 或 `graph_traverse`,无自动路由。

### 图谱增强三路召回 (Graph-RAG Recall Pipeline) [部分实现]
当策略判定需要深挖细节时,执行以下三步走:

1. **种子 Block 召回 (Vector + BM25):** [已实现 — `search_step`]

使用倒数秩融合 (RRF) 算法合并稠密向量与稀疏词频的打分:

$$RRF\_Score = \frac{vector\_weight}{60 + Rank_{vector}} + \frac{text\_weight}{60 + Rank_{BM25}}$$

(reme4 实现在 `reme4/steps/common/search.py::SearchStep._rrf_merge`, 常量 `_RRF_K = 60`, 默认 vector_weight=0.7。)

选出分数最高的 Top-K Blocks,并上溯找到其对应的 $N$ 个**种子文件节点 (Seed Nodes)**。

2. **图谱游走扩展 (1-Hop Graph BFS):** [已实现 — `graph:traverse`]

以这 $N$ 个节点为起点,在 file_graph 中执行深度为 1 的广度优先遍历。

    - **剪枝:** 当前按 `predicate` / `depth` 过滤;`is_active` / 时间戳剪枝 [未实现]。
    - **种子来源:** 需要 caller 把 `search` 结果手动喂进 `graph_traverse`,**没有自动串联的 pipeline 步骤**。

3. **动态组装与裁剪 (Rerank & Prompt Assembly):** [未实现]

将底层拿到的冷数据转换为 Agent 的高质量提示词片段。

```plain
[🎯 核心实体背景: knowledge/zhang-san/zhang-san.md]
(来自文件 YAML: 状态活跃, 前端架构师)

[🕸️ 认知关系网络]
- 负责 -> [[knowledge/项目X]] (2026-05)
- 冲突对象 -> [[knowledge/li-si]]

[🔍 命中事实切片]
"2026-05-06: 张三与李四发生技术冲突,引入了 React。" (相关度: 0.92)
```

> 现状:`search` 返回 FileChunk 列表 (含分数与片段);Schema-aware rerank + 上述格式化模板未实现。

---

## 落地状态总览

按 5-layer + Schema 分组:

| Layer | 模块 | 状态 | 备注 |
|---|---|---|---|
| **L1** | Agent (Claude Code) | — | 外部 |
| **L2** | reme-service plugin | ✅ 已实现 | 24 MCP tools + Hooks; SessionEnd 走 distiller |
| **L2** | reme-expert plugin | ✅ 已实现 | 22 MCP tools + 2 subagents (distiller/curator) + 3 slash |
| **L2** | Hooks (PreCompact/SessionEnd/Stop) | ✅ 已实现 | `active_daily_check.py` 扫 daily/ |
| **L3** | Synchronizer | ✅ 已实现 | 热写 daily workspace; `_coerce_messages` 支持 dict/Msg 双输入 |
| **L3** | Distiller (原 Ingestor) | ✅ 已实现 | 冷写 knowledge;R-M-W + lint:schema 自查 + status flip |
| **L3** | Maintainer (Merge/Split/Decay) | ❌ 未实现 | 服务未建;`status: archived` 已是合法值但无触发逻辑 |
| **L4** | search (RRF 混合召回) | ✅ 已实现 | `reme4/steps/common/search.py` |
| **L4** | graph_traverse (1-hop BFS) | ✅ 已实现 | `reme4/steps/graph/traverse.py` |
| **L4** | crud (list/read/write/stat/property_*) | ✅ 已实现 | `reme4/steps/crud/` |
| **L4** | File ops (move/copy/delete/upload/download) | ✅ 已实现 | `reme4/steps/crud/{move,copy,delete,upload,download}.py` |
| **L4** | Daily (daily:read/daily:list) | ✅ 已实现 | `reme4/steps/daily/` — 工作区 CRUD |
| **L4** | tags (tags_list/tags_stat) | ✅ 已实现 | `reme4/steps/tags/` |
| **L4** | lint (dangling/orphans/collisions/schema) | ✅ 已实现 | `reme4/steps/lint/` |
| **L4** | Retriever 统一类 (intent routing + rerank) | ❌ 未实现 | search + graph_traverse 散步式实现 |
| **L4** | 三路融合 pipeline (search → graph → rerank) | ❌ 未实现 | 需要 caller 手动串联 |
| **L5** | file_store (MFS) | ✅ 已实现 | LocalFileStore / 可扩展 |
| **L5** | file_watcher + file_parser | ✅ 已实现 | lite watcher + bare/default/md parser |
| **L5** | Projections (vector/keyword/graph) | ✅ 已实现 | embedding + BM25 + Local/Nx/Neo4j file_graph |
| **L5** | 隐式 LLM IE 关系抽取 (慢路线) | ❌ 未实现 | 当前由 Distiller 在 prompt 内显式产出 wikilink 代替 |
| **Schema** | protocol.md (4 轴文字契约) | ✅ 已实现 | `reme4/steps/jobs/protocol.md` |
| **Schema** | lint:schema (后置 validator) | ✅ 已实现 | 必填 4 key + status enum |
| **Schema** | file_front_matter (开放 dict 模型) | ✅ 已实现 | `extra=allow` |
| **Schema** | Python preset 类 | ❌ 未实现 | 等 vault 域稳定再固化 |
