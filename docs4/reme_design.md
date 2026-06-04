# 新版ReMe 设计文档

## 整体定位

> 一句话总结：自进化的个人知识库

## 特性1：记忆分层

记忆按"原始 → 浅加工 → 深加工 "三层组织：

### 1.1 目录结构示例

```
- resource【原始素材】                        # 外部渠道摄入/手动摄入
  - YYYY-MM-DD/                                # 按接收日期归档，含 meta.json 索引
    - session_{session_id}.jsonl             # 对话原始记录（msg_to_dict 序列化）
    - {channel}_{xxxx}.html                   # 网页抓取、邮件等 HTML 原件
    - {channel}_{xxxx}.txt                    # 文本类资料
    - {channel}_{xxxx}.md                     # Markdown 类资料
- daily【日记，浅加工】                        # auto-memory 写入
  - YYYY-MM-DD.md 索引                          # 当天主索引页，汇总所有事件 [[wikilink]]
  - YYYY-MM-DD/                                
    - session_{session_id}.md 日志本           # 分sessionid
    - resource_{resource_id}.md               # 对 resource 素材的加工笔记
- digest【深加工】                             # auto-dream 持续打磨，可反复消费的精华层
  - personal/ 个性化信息                       # 用户偏好、习惯、身份特征
    - xxx.md
  - procedural/ 程序化记忆                     # 方法论、步骤
    - xxx.md
  - wiki/ 知识型记忆                           # 通用知识
    - xxx.md                                  
```

### 1.2 分层详解

| 目录                  | 内容性质                    | 写入方/触发               | 生命周期     | 典型例子                              |
|---------------------|-------------------------|----------------------|----------|-----------------------------------|
| `resource/`         | 外部渠道摄入的原始文件（研报、网页、邮件附件） | ingest step / 手动放入   | 只增文件不删文件 | PDF 研报、HTML 网页抓取、对话 JSONL         |
| `daily/`            | 每天的事件性记忆，按对话 session 拆分 | auto-memory          | 只增文件不删文件 | "调试登录页面 CSS"、"与 Alice 周末聚餐"       |
| `digest/procedure/` | 方法论：步骤、工作流              | 任务完成后归纳              | 持续打磨演化   | "webpack 编译卡死的排查路径"               |
| `digest/personal/`  | 用户画像：偏好、习惯、身份特征         | 用户纠正 / 偏好表达          | 持续打磨演化   | "用户不爱写注释"、"用户喜欢 pnpm"             |
| `digest/wiki/`      | 通用知识：定义、原则、决策先例         | 主题对话 / auto-dream 归档 | 持续打磨演化   | "光伏产业链"、"React Server Components" |

`resource/` 和 `daily/` 是只增不删的"流水帐"——前者保留原始事实、后者保留事件级现场；`digest/` 下三个桶是被反复消费的精华层，各桶有独立的
auto-dream 整合 prompt。

## 特性2：Obsidian 兼容的 Markdown 格式

```
┌─────────────────────────────────────────────────────────────┐
│  一个 .md 文件的完整结构                                       │
├─────────────────────────────────────────────────────────────┤
│  ---                                                        │
│  name: 宁德时代                     ← YAML front matter      │
│  description: 全球动力电池龙头                                │
│  tags: [新能源, 电池]                                        │
│  ---                                                        │
├─────────────────────────────────────────────────────────────┤
│  所属行业:: [[新能源]]               ← 语义化链接（Dataview）   │
│  竞争对手:: [[比亚迪]]                                        │
│                                                             │
│  # 基本面                            ← Markdown 正文         │
│  全球动力电池出货量第一，核心技术为                              │
│  [[CTP]] 和 [[钠离子电池]]……         ← 标准 wikilink         │
│                                                             │
│  参考 ![[2026Q1调研纪要]]            ← 嵌入引用               │
├─────────────────────────────────────────────────────────────┤
│          ↓ AST 语义分块 ↓                                    │
│  chunk 1: [标题骨架] + 正文片段                               │
│  chunk 2: [标题骨架] + 正文片段                               │
└─────────────────────────────────────────────────────────────┘
```

### 2.1 YAML front matter

每个笔记的元数据存储在 YAML front matter 中：

```markdown
---
name: 光伏产业链研究
description: 从硅料到组件的全链条梳理
tags: [新能源, 光伏, 产业链]
---
```

`name` / `description` 是约定字段（`FileFrontMatter` schema），其余键值对作为 extras 全部保留。
frontmatter 使用开放模型（`extra="allow"`），不会丢弃任何自定义字段。

### 2.2 四种 wikilink 写法

| 写法   | 示例             | 语义         |
|------|----------------|------------|
| 标准链接 | `[[光伏产业链]]`    | 指向目标文件     |
| 锚点链接 | `[[钴#应用]]`     | 指向文件中的特定章节 |
| 别名链接 | `[[宁德时代\|宁德]]` | 自定义显示文本    |
| 嵌入引用 | `![[钴]]`       | 内联嵌入目标内容   |

### 2.3 语义化链接（Dataview 风格）

普通 wikilink 只表达"A 提到了 B"，语义化链接额外表达"A 和 B 是什么关系"：

```markdown
<!-- 例：在「宁德时代.md」笔记中 -->
所属行业:: [[新能源]]            ← 行级属性（独占一行）
总部:: [[宁德]]
[竞争对手:: [[比亚迪]]]          ← 内联属性（嵌入正文中）
```

`WikilinkHandler` 是全系统唯一的 wikilink 语法真理源，负责正则解析四种写法 + Dataview 谓词推断，确保解析规则在parser、graph、search
各层完全一致。

### 2.4 AST 感知的语义分块

传统 RAG 用固定 token 长度 + overlap 切片，经常切坏文档结构。ReMe 提供Markdown AST 语义分块：

- 使用 mistletoe 解析为 `MdNode` 树（root / section / body 三种节点），按 H1/H2/H3 章节嵌套
- 递归分块：先试整子树；超限则遍历子节点——同级 body 贪心打包，子 section 递归处理
- **每个 chunk 携带完整标题骨架**：当前内容前后的章节标题作为 TOC 保留，让检索到的片段一眼看出"这段在哪个章节、什么主题下"
- 叶子节点智能分割：表格重复表头、代码块重复 fence 标记、列表按项打包、段落按行贪心
- 多片段输出添加 `[Part X/N]` 标记

```
某 chunk 实际内容示例：
─────────────────────
# 光伏产业链
## 上游：硅料
### 多晶硅工艺
[chunk 正文]          ← 当前 chunk 的实际内容
## 中游：硅片          ← 后续章节骨架（只有标题，无正文）
## 下游：组件
─────────────────────
```

## 特性3：自进化

> **ReMe 的记忆不是被动存的，是主动长成知识图谱的。**

```
用户正常对话 / 外部素材流入
        │
        ▼
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  auto-resource   │    │   auto-memory    │    │   auto-dream     │
│  监控 resource/  │    │  对话 → daily/   │    │  daily/ → digest/│
│  新文件 → 解析    │    │  LLM 自主记录    │    │  提炼 + 建图谱   │
└────────┬─────────┘    └────────┬─────────┘    └────────┬─────────┘
         │                       │                       │
         ▼                       ▼                       ▼
    resource/                 daily/                  digest/
    (原始素材)             (事件日记)             (知识卡片 + wikilink 图谱)
```

不需要用户手工整理，Agent 在后台让笔记自己长出结构。

### 3.1 auto-resource

自动检测 `resource/` 目录下新增的原始素材（邮件附件、网页抓取、文档等），触发解析和整理：

- 监控 `resource/` 目录变化，发现新文件后自动解析内容
- 将解析结果整理为结构化笔记，写入 `daily/YYYY-MM-DD/` 下
- 整理完成后推送通知给上游 Agent，由 Agent 决定后续动作

> 注：实际监控范围和推送方式取决于上游 Agent 的能力集成。

### 3.2 auto-memory

Agent 对话进行时，除了Agent自主会记笔记到daily/YYYY-MM-DD.md，ReMe 在后台把上下文按session自动拆分写入 daily 笔记。

**工作流**：

1. 调用 `daily_create` 幂等获取当天笔记路径
2. 构建 ReAct Agent，配备 `read` / `edit` / `frontmatter_update` / `write` 工具
3. Agent 根据对话历史（messages），自主决策在每日笔记中记录什么、保留什么

**设计要点**：不是简单的对话摘要，而是让 LLM Agent 拥有完整的文件读写能力，自行决策如何组织信息。Agent
可以读取已有笔记内容、决定合并还是新增、选择合适的标题和标签。

### 3.3 auto-dream + auto-link：睡眠式记忆整理

人白天经历事情，睡眠时大脑把碎片巩固为长期知识。auto-dream 就是这个过程——把日记和素材提炼成可复用的知识卡片，并自动织出图谱关系。

```
                        ┌───────────────────────────┐
                        │  daily/2026-05-28/xxx.md  │  ← 一篇日记或素材
                        └─────────────┬─────────────┘
                                      │
                    ╔═════════════════════════════════════╗
                    ║  Phase 1 — Extract（一个 Agent）     ║
                    ║  "这份材料教了什么道理？"               ║
                    ║                                     ║
                    ║  输出 N 个抽象单元，各带 bucket 标签    ║
                    ║  (空 → 结束，没东西值得记)              ║
                    ╚══════════╤══════════╤═══════════════╝
                               │          │
                 ┌─────────────┘          └──────────────┐
                 ▼                                       ▼
  ╔══════════════════════════════╗     ╔══════════════════════════════╗
  ║  Phase 2 — Integrate        ║     ║  Phase 2 — Integrate        ║
  ║  (每个 unit 独立一个 Agent)   ║     ║  (每个 unit 独立一个 Agent)   ║
  ║                              ║     ║                              ║
  ║  1. search + traverse 召回   ║     ║  1. search + traverse 召回   ║
  ║  2. 决策: CREATE / UPDATE    ║     ║  2. 决策: CREATE / UPDATE    ║
  ║  3. 写入 + 自动织链接         ║     ║  3. 写入 + 自动织链接         ║
  ╚══════════════╤═══════════════╝     ╚══════════════╤═══════════════╝
                 │                                     │
                 ▼                                     ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  digest/                                                     │
  │    procedure/key-rotation.md  ←─ derived_from:: [[daily/..]] │
  │    wiki/credential-compliance.md ─ relates_to:: [[...]]      │
  │    personal/user-pr-pref.md                                  │
  └──────────────────────────────────────────────────────────────┘
                         知识图谱自动生长
```

**要点 1：Phase 1 筛选——"有什么道理值得记？"**

多个事实说明同一个道理就合并为一个 unit，分到三个桶：`procedure`（怎么做）/ `personal`（用户偏好）/ `wiki`（通用知识）。没东西值得记则流程结束。

**要点 2：Phase 2 先搜后写——不重复造卡片**

先搜已有 digest，再决策：新建（CREATE）、追加佐证（CORROBORATE）、补充精度（REFINE）、修正矛盾（CORRECT）。

**要点 3：auto-link 是写入的副产品**

写 digest 时自动加 `derived_from:: [[素材]]` 溯源 + `relates_to::` 概念互联，图谱随每次 dream 自动变密。

**要点 4：CronDreamer 定时批跑**

每天扫描当天所有 daily + resource 文件，逐个执行上述管线。

### 3.4 三者协同

```
[实时]                       [离线]                          [持续]
对话 + 外部资产               空闲触发                         后台监控
    │                          │                              │
    ├─ auto-memory ──► daily/  │                              │
    ├─ ingest ──────► resource/│                              │
    │                          ├─ auto-dream ──► digest/      │
    │                          │    Phase 1: 提取抽象单元      │
    │                          │    Phase 2: 按桶整合写入      │
    │                          │       └─ auto-link ──► [[wikilink]]
    │                          │                              │
    │                          │                     auto-resource ──► 索引更新
    └──────────────────────────┴──────────────────────────────┘
                              知识图谱自然生长
```

## 特性4：混合索引+渐进式展开

```
用户提问: "钴在锂电中的应用？"
         │
         ├──────────────────────┬──────────────────────────┐
         ▼                      ▼                          │
  ┌─────────────────┐   ┌──────────────────┐              │
  │ 全文倒排索引     │   │ 向量索引          │              │
  │ (numpy + jieba) │   │ (faiss)          │              │
  │                 │   │                  │              │
  │ "钴" 精确命中    │   │ "锂电正极原料"    │              │
  │  → rank 1,2,3  │   │  语义近似命中      │              │
  └────────┬────────┘   └────────┬─────────┘              │
           │  text_weight=0.3    │  vector_weight=0.7      │
           └──────────┬──────────┘                         │
                      ▼                                    │
              ┌───────────────┐                            │
              │  RRF 融合排序  │                            │
              │  score = Σ(w/(k+rank))                     │
              └───────┬───────┘                            │
                      ▼                                    │
  ┌──────────────────────────────────────┐                 │
  │  第一跳：Top-K chunk 全文 + 评分       │                 │
  └───────────────────┬──────────────────┘                 │
                      ▼                                    │
  ┌──────────────────────────────────────┐                 │
  │  第二跳：邻居目录（只有标题，不展开正文）│  ← wikilink 图谱 │
  └───────────────────┬──────────────────┘                 │
                      ▼                                    │
  ┌──────────────────────────────────────┐                 │
  │  第 N 跳：Agent 按需追问，展开正文     │                 │
  └──────────────────────────────────────┘                 │
```

### 4.1 混合索引构建

两套索引并行维护，各擅其长：

- **全文倒排索引**（基于numpy）：精确匹配专有名词，搜"宁德时代"必须命中。增量更新，纯 sqlite/chromadb依赖。
- **向量索引**（基于faiss）：语义相似度，搜"锂电正极原料"能命中"钴"。基于 embedding 模型生成稠密向量。

### 4.2 基于 RRF 的混合检索

两条通路并行跑（`asyncio.gather`），结果用 RRF（Reciprocal Rank Fusion）融合：

```
融合分 = Σ( weight_i / (k + rank_i) )    k=60, vector_weight=0.7, text_weight=0.3
```

纯向量容易把名词错配（"苹果公司"≈"水果"），纯关键词抓不到同义改写——两路融合互补盲区。

### 4.3 渐进式链接展开

传统 RAG 一次性把 Top-K 全塞进上下文，token 浪费且噪音多。ReMe 分跳展开，按需深入：

**第一跳** — 返回命中 chunk 全文 + 分数明细（vector / keyword / fused）

**第二跳** — 对每个命中文件，展开 wikilink 邻居的"目录"（只有标题，不展开正文）：

```
========== digest/wiki/宁德时代.md:5-22 [score=0.0247 vector=0.0156 keyword=0.0091] ==========
# 宁德时代
全球动力电池出货量第一，核心技术为 CTP（Cell to Pack）和钠离子电池……

  outlinks (2):
    → digest/wiki/磷酸铁锂.md  name="磷酸铁锂正极路线"  description="磷酸铁锂与三元路线对比"  via predicate=相关技术
    → digest/wiki/固态电池.md  name="固态电池技术路线"  description="全固态与半固态进展"  via predicate=技术演进
  inlinks (2):
    ← daily/2026-03-18/宁德调研.md  name="宁德时代调研纪要"  description="2026Q1产能与订单跟踪"  via plain
    ← digest/wiki/新能源产业链.md  name="新能源产业链全景"  description="从锂矿到整车的全链条"  via predicate=下游应用
```

**第 N 跳** — Agent 看过"目录"后，自己决定哪些邻居值得展开正文，再发起 read_file 拿详情。

二跳目录每条只占一行（最多 10 outlink + 10 inlink），Agent 拥有全局视野却不撑爆上下文。

## 特性5：多 Agent 框架集成

ReMe 不做独立 Agent 产品，而是作为**能力层**被任意 Harness 调用：

| 集成路径                | 适用对象                 | 方式                                                |
|---------------------|----------------------|---------------------------------------------------|
| SDK 深度集成            | AgentScope / Qwenpaw | 通过 middleware 注册 tools + prompt，通过 hook 注册 auto-* |
| MCP Tool + skill.md | Claude Code          | 通过 MCP 注册 Tool，配 skill.md 开箱即用，通过 hook 注册 auto-*  |
| HTTP API + CLI      | 通用方案                 | skill.md + CLI 调用                                 |

---

# 二、工程架构

```
┌─────────────────────────────────────────────────────────────────┐
│  Service 层（HTTP / MCP 双协议）                                  │
│  FastAPI + FastMCP，同一套 Job 同时暴露为 REST 和 MCP Tool         │
├─────────────────────────────────────────────────────────────────┤
│  Application 层                                                  │
│  配置加载 → 组件初始化 → Job 注册 → start() / close() 生命周期     │
├─────────────────────────────────────────────────────────────────┤
│  Job 层（编排）                                                   │
│  每个 Job = 一组 Step 的有序管线，YAML 声明式配置                   │
├─────────────────────────────────────────────────────────────────┤
│  Step 层（业务逻辑）                                              │
│  原子操作单元，按功能域分组：file_io / index / evolve / common      │
├─────────────────────────────────────────────────────────────────┤
│  Component 层（可插拔基础设施）                                    │
│  统一注册表 R，一行配置切换实现                                     │
│  file_store / embedding / keyword_index / llm / file_graph       │
└─────────────────────────────────────────────────────────────────┘
```

## 2.1 服务层（非 SDK集成）

每个 Job 同时暴露为两种协议，无需重复开发：

| 协议            | 传输方式                          | 适用场景                         |
|---------------|-------------------------------|------------------------------|
| HTTP（FastAPI） | JSON POST / SSE               | 通用 REST 调用、Web 前端            |
| MCP（FastMCP）  | stdio / SSE / streamable-http | Claude Code、Cursor 等 MCP 客户端 |

通过 `REME_SERVICE_INFO` 环境变量广播绑定地址，客户端自动发现。

- **按需拉起**：Agent 检测到 ReMe 服务未运行时，可自动后台拉起，用户无感知
- **服务发现**：`find_reme` 一键探活，避免端口冲突；多个 ReMe 实例共存时也能精准定位

## 2.2 组件系统（Component）

ComponentRegistry 单例 `R`，二级注册 `(类型, 名称) → 实现类`，一行配置切换后端：

| 组件类型            | 职责           | 可选后端                  |
|-----------------|--------------|-----------------------|
| file_store      | 文件存储 + 索引协调  | local                 |
| file_graph      | wikilink 双向图 | local / nx / neo4j    |
| keyword_index   | 全文倒排索引       | bm25（numpy + jieba）   |
| embedding_store | 向量存储 + 检索    | local（faiss）          |
| embedding       | 文本向量化        | openai 兼容接口           |
| llm             | 大模型调用        | anthropic / openai 兼容 |
| tokenizer       | 分词器          | regex / jieba         |

## 2.3 Job 列表

**Job** 是 ReMe 暴露给外部的操作单元——同一个 Job 可以作为 Python 函数直接调用、作为 MCP Tool 被 Agent 使用、也可以作为 CLI
命令行执行。YAML 声明式配置，每个 Job 内部由一组 Step 串联：

| 类别   | Job                       | 功能                               |
|------|---------------------------|----------------------------------|
| 检索   | `search`                  | 混合检索（向量 + BM25 + RRF）+ 渐进式图展开    |
| 检索   | `traverse`                | 从指定路径遍历 wikilink 图谱              |
| 文件读写 | `read`                    | 读取 markdown 文件内容                 |
| 文件读写 | `read_image`              | 读取图片文件（base64）                   |
| 文件读写 | `write`                   | 新建或覆写 markdown 文件（含 frontmatter） |
| 文件读写 | `edit`                    | 文件内查找替换                          |
| 文件读写 | `delete`                  | 删除文件，返回残留入边                      |
| 文件读写 | `move`                    | 移动/重命名文件，自动重写入边 wikilink         |
| 文件读写 | `list`                    | 列出目录下文件                          |
| 文件读写 | `stat`                    | 文件元信息（大小、时间、是否存在）                |
| 文件读写 | `frontmatter_read`        | 读取文件 frontmatter                 |
| 文件读写 | `frontmatter_update`      | 合并更新 frontmatter 字段              |
| 文件读写 | `frontmatter_delete`      | 删除 frontmatter 字段                |
| 日记管理 | `daily_create`            | 幂等创建当天日记文件                       |
| 日记管理 | `daily_list`              | 列出某天的所有日记                        |
| 日记管理 | `daily_reindex`           | 重建当天日记索引页                        |
| 索引维护 | `reindex`                 | 清空并全量重建索引                        |
| 索引维护 | `update_store_index_loop` | 后台持续监听文件变更，增量更新索引                |
| 自进化  | `auto_memory`             | 将对话记录写入当天日记（LLM Agent）           |
| 自进化  | `dream`                   | 单文件记忆提炼到 digest（LLM Agent）       |
| 自进化  | `auto-dream`              | 批量扫描当天文件，逐个 dream                |
| 系统   | `health_check`            | 组件健康检查                           |
| 系统   | `version`                 | 返回版本号                            |
| 系统   | `help`                    | 列出所有已注册 Job                      |


