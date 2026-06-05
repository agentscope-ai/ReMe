# ReMe：把本地 Markdown 自进化成知识图谱的个人记忆引擎

> 面向 Leader / 决策者的能力报告
> 关键词：个人记忆 · 记忆自进化 · 多模检索 · Agent 接入 · 本地优先

---

## 一、引言：为什么要做新版本

### 1.1 ReMe 的一句话定位

> **ReMe 是一个把本地 Markdown 自进化成知识图谱的个人记忆引擎。**

- **记忆分层（形态的物化）** → 记忆按"原始 → 加工"两层组织：`resource/`（原始素材）、`daily/`（日记事件）是只增不删的流水帐；`digest/` 是加工层，下分 `personal/`（个性化）、`knowledge/`（主题知识）、`procedural/`（Agent 任务经验）、`proactive/`（主动洞察）四个固定子目录，写入策略和检索权重各有差异。
- **本地 Markdown（载体）** → 所有记忆都是 Obsidian 兼容的 .md 文件——YAML front matter、四种 wikilink（`[[X]]` / `[[X#anchor]]` / `[[X|alias]]` / `![[X]]`）、Dataview 风格 `predicate:: [[X]]` 语义关系全部沿用社区约定。用户可读、可备份、可迁移，对抗黑盒。
- **自进化（机制）** → 不需要用户手工整理，Agent 在后台让笔记自己长出结构。这一点把 ReMe 同时与"手动建图的 Obsidian"和"扁平存储的 Mem0"拉开。【event/trace】
- **知识图谱（结果）** → 自进化的产出物不是一堆扁平笔记，而是一张可被**多模 + 渐进式检索**消费的图：向量 + 关键词（中文 BM25）+ 图谱三路 RRF 融合，返回时通过 1-hop 邻居 meta 让 Agent"先看目录、再决定要不要展开正文"，不像传统 RAG 那样一次性把 top-K 切片塞进上下文。
- **被集成而非内置（分发形态）** → ReMe 不做独立 Agent 产品，而是作为**能力**被任意 Harness 调用：SDK 深度集成（qwenpaw / AgentScope）、MCP Tool（Claude Code / Cursor / Cherry Studio）、CLI + skill.md 三条路径并行，记忆跟着用户走，不绑定任何上层框架。

支撑这一切的是**自研轻量索引内核**——纯 Python(后期可以使用rust重写索引内核) + 文件持久化，无 sqlite / chroma
等原生扩展依赖，老旧 Linux/Win 也能稳跑，这是 ReMe 能"被部署到大量异构用户机器"的工程前提。

---

## 二、产品全景图

### 2.1 一张图看 ReMe

```
┌─────────────────────────────────────────────────────────┐
│  外部 Agent / Harness                                   │
│  qwenpaw · Claude Code · Cursor · 其它                  │
└──────────┬──────────────┬───────────────┬───────────────┘
           │ SDK          │ MCP Tool      │ CLI / skill.md
┌──────────▼──────────────▼───────────────▼───────────────┐
│                ReMe Service Layer                        │
│   HTTP / MCP / CLI · 服务发现 · 进程托管                 │
├─────────────────────────────────────────────────────────┤
│                ReMe Job/Step 编排                        │
│   search · auto_memory · auto_dream · auto_link …       │
├─────────────────────────────────────────────────────────┤
│        Markdown 知识内核（本地文件即数据库）              │
│   FileChunker · FileStore · FileGraph · FileWatcher      │
│   BM25 倒排 · 向量索引 · Wiki Link 图谱                 │
├─────────────────────────────────────────────────────────┤
│        文件目录约定                                      │
│   resource/ · daily/ · digest/{personal,knowledge,       │
│                                procedural,proactive}/    │
└─────────────────────────────────────────────────────────┘
```

四层结构从上到下：

- **接入层**：让任何 Agent 框架都能用。
- **服务层**：HTTP / MCP / CLI 三种协议同时暴露，按 Harness 需求选用。
- **编排层**：把能力拆成 Job/Step，可组合、可流式。
- **内核层**：Markdown 解析、存储、图谱、监听一体化。
- **目录层**：用户最直观看到的文件夹结构，本身就是 ReMe 的"产品形态"。

### 2.2 三个最直观的故事场景

**场景一：金融分析师的产业链知识库**
盘后，分析师和 Agent 对话讨论今天看到的几条新能源新闻。第二天打开知识库，发现昨天的对话已经被自动拆分成「钴价波动」「下游电池厂动向」「上游矿企并购」三条事件笔记，分别归档到
`digest/knowledge/financial/产业链/` 下相应主题；笔记之间通过 `[[钴]]` `[[宁德时代]]` 这样的 wikilink 互相串联。下周再问"钴的下游应用"，ReMe
沿着图谱渐进展开，从一个节点跳到相关的全部上下文。

**场景二：个人工作 & 生活第二大脑**
日常对话、会议讨论、学习笔记都被主 Agent 实时写入 daily 笔记。夜晚 Agent 空闲时，ReMe 在后台把零散的 daily 内容按"
客户/项目/学习/生活"主题重新整理到 knowledge 库，并自动补全笔记之间的链接。三个月后，用户拥有一份完全属于自己的、可视化的"
第二大脑"，可以用 Obsidian 直接打开浏览。

**场景三：Agent 框架开箱接入**
开发者在 qwenpaw 或 Claude Code 中安装 ReMe，不需要改 Agent 一行代码 —— Agent 立刻拥有"长期记忆 + 个人知识检索 +
自动整理"三项能力。SDK 集成是无感的：每次对话自动落入 daily，每次检索自动走多模融合，每天空闲自动整理归档。

---

## 三、记忆模型：ReMe 把什么存下来

### 3.1 两层记忆结构：原始素材 + 加工记忆

ReMe 不把对话一股脑塞进数据库，而是按"原始 → 加工"分两层组织：

| 层次       | 目录          | 存什么                              | 典型场景                          |
|----------|-------------|----------------------------------|-------------------------------|
| **原始素材** | `resource/` | 上传文件、抓取网页、PDF 研报、邮件附件            | 溯源 / 审计 / 二次加工                |
| **日记流水** | `daily/`    | 每天的事件性记忆，主 Agent 实时写入            | "我今天和谁聊了什么"、"今天工作内容"          |
| **加工记忆** | `digest/`   | 经 Auto-Dream 整理后的长期记忆，下分四类（见 3.2） | "光伏产业链"、"webpack 排查路径"、"早间洞察" |

`resource/` 和 `daily/` 是只增不删的"流水帐"——前者保留原始事实、后者保留事件级现场；`digest/` 才是被反复消费的精华层，也是 Auto-Dream / Auto-Link 持续打磨的主要产物。

### 3.2 digest 下的四种记忆

`digest/` 固定划分四个子目录，正交覆盖"用户 / 知识 / Agent / 主动"四个维度：

| 子目录                    | 内容性质                | 写入触发                 | 检索权重        | 典型例子                                  |
|------------------------|---------------------|----------------------|-------------|---------------------------------------|
| **digest/personal/**   | 个性化（用户偏好、习惯、人物档案）   | 用户纠正 / 偏好表达          | 全场景常驻       | "用户不爱写注释"、"用户喜欢 pnpm"                 |
| **digest/knowledge/**  | 知识类（客观知识、领域参考）      | 主题对话 / Auto-Dream 归档 | 主题相关性匹配     | "光伏产业链"、"React Server Components" |
| **digest/procedural/** | 程序化（Agent 任务经验）     | 任务完成后归纳              | 任务相似度高时唤起   | "webpack 卡死的排查路径"                     |
| **digest/proactive/**  | 主动推送（Agent 输出的分析）   | 定时 / 触发式生成           | 时效衰减        | 早间洞察、周复盘、热点跟踪                         |

四类的"形状"刻意不同：

- `digest/knowledge/` 由用户自定义二级分类（如 `work/`、`financial/`、`life/`），下面层级随意展开。
- `digest/personal/`、`digest/proactive/` 保持扁平，便于全量加载或时序浏览。
- `digest/procedural/` 软链到 Harness Agent 的 skill 目录（P0），让 Agent 沉淀的"经验"直接成为可复用的 skill。

### 3.3 目录约定（用户可读、可备份、可迁移）

skills 建议使用index 看 whole picture~

```
~/reme_workspace/
├── resource/                    # 原始素材（按日期归档）
│   └── 20260518/
│       ├── 1430_xueqiu_comment.html       # 雪球评论抓取
│       └── 1620_research_report.pdf       # 研报原件
├── daily/
│   ├── 20260518.md              # 当天主索引（兼容 write/edit）
│   └── 20260518/
│       ├── meeting-with-alice.md
│       ├── debug-login.md
│       └── reading-paper.md
└── digest/                      # 固定四个子目录：personal / knowledge / procedural / proactive
    ├── personal/                # 个性化记忆：偏好、习惯、个人事件
             .change_log.md difference
             _moc.md
    │   └── 用户偏好.md
    ├── knowledge/               # 知识类记忆：用户自定义二级目录（work / financial / ...）
    │   ├── work/
    │   ├── financial/
    │   │   ├── 光伏产业链.md
    │   │   └── 钴.md
    │   └── ...
    ├── procedural/              # 程序化记忆：软链到 harness agent 的 skill 目录（P0）
    │   └── xxxx.md              # Agent 完成任务沉淀的经验
    └── proactive/               # 主动推送：各家协议不同
        └── 20260518.md          # Agent 主动产出的建议
```

> `digest/` 下固定为 **personal / knowledge / procedural / proactive** 四个子目录；其中 `knowledge/` 内部由用户自行扩展二级分类（如 `work/`、`financial/`），其余三类保持扁平结构。

**所有记忆都是普通 Markdown 文件**，用户随时可以：

- 用 Obsidian / Typora / VSCode 打开浏览编辑
- 用 Git / iCloud / 网盘做版本控制和跨设备同步
- 迁移到任何机器，复制目录即可
- **没有黑盒数据库，没有产品锁定**

这一点对 Leader 视角尤其重要：用户对自己数据的掌控感，是所有"个人记忆"产品的信任基础。

## 四、Markdown 内核：把文件当数据库

### 4.1 Obsidian 兼容的 Markdown 格式

ReMe 没有发明新格式，而是完全复用 Obsidian 生态的约定：

- **YAML front matter**：标题、标签、描述、自定义字段。

```markdown
---
title: 光伏产业链研究
description: 从硅料到组件的全链条梳理
tags: [新能源, 光伏, 产业链]
parent: 新能源
author: 张三
updated。: 2026-05-19
---

# 正文从这里开始

`title` / `description` / `tags` 是约定字段（参见 `reme4/schema/file_front_matter.py`），其余键值对作为 extras
全部保留，可被检索和图索引消费。
```

- **4 种 wikilink 写法**：
    - `[[X]]`：标准链接
    - `[[X#anchor]]`：链接到文件中的章节
    - `[[X|alias]]`：自定义显示文本
    - `![[X]]`：嵌入引用
- **Dataview 风格语义关系**：`predicate:: [[X]]`，例如 `parent:: [[新能源]]`、`founder:: [[张三]]`，把 link 升级为带类型的"
  边"。
- 标准 `[text](xxx.md)` 链接也会被识别为图边。

**意义**：用户的知识库可以直接用 Obsidian 打开做可视化浏览，可以用 Obsidian 插件做扩展。ReMe 不是替代 Obsidian，而是**给
Obsidian 加上一个会自己写笔记的 Agent**

### 4.2 比 RAG 更聪明的切片

传统 RAG 用固定 token 长度 + overlap 切片，经常切坏文档结构。ReMe 用 Markdown AST 切片：

- 解析为章节嵌套树（按 H1/H2/H3 分层）。
- 按章节边界递归切分，保留语义完整性。
- **每个 chunk 自带完整的标题骨架（TOC）**：检索回来的片段一眼就能看出"这段在哪个章节、什么主题下"。

```
某 chunk 实际内容长这样：
─────────────────────
# 光伏产业链
## 上游：硅料
### 多晶硅工艺
[chunk 正文]
## 中游：硅片
## 下游：组件
─────────────────────
```````

Agent 拿到这个 chunk，立刻知道层级位置，不会断章取义。

### 4.3 Graph 索引：双向链接

定位句里"自进化成**知识图谱**"的物理形态，就落在这一节——每个文件参与两套索引：

- **正向（outlinks）**：A → B（A 引用了 B）
- **反向（inlinks）**：B ← {A, C, D}（谁引用了 B）

**反向链接**是知识库可用性的关键 —— 让你站在任意一个概念上，看到"还有哪些地方提到过我"。

ReMe 提供三种 graph backend，按规模和需求切换：

- **本地 dict + JSONL**：轻量、零依赖、适合个人规模。
- **NetworkX + pickle**：方便做图算法分析。
- **Neo4j**：企业规模、Cypher 查询、可视化丰富。

切换只需配置一行。


## 五、记忆的自进化（核心差异化）

> 这是 ReMe 最重要的能力，也是和市面所有「记忆即数据库」产品的根本分野。
>
> **ReMe 的记忆不是被动存的，是主动长成知识图谱的。**

定位句中"自进化成知识图谱"的具体路径，由下面三件套共同承担：**auto-memory** 在前线把对话拆成事件，**auto-dream**
在空闲时把事件归档成主题，**auto-link** 把这一切用 wikilink 串成图。三者协作，daily 流水最终被织成一张越用越密的个人知识图谱。

### 5.1 Auto-Memory：实时拆事件

主对话进行时，ReMe 在后台把上下文按"事件"自动拆分：

- 用户和 Agent 的连续对话，被识别为若干个独立事件（一次会议、一次 debug、一次学习）。
- 每个事件成为一个独立的 `daily/YYYYMMDD/{event}.md` 笔记。
- 同时在 `daily/YYYYMMDD.md` 维护主索引，所有事件可被反向追溯。

**用户体验**
：不需要手动整理。打开当天主索引，事件已经分章节列好，每条都能跳转到独立笔记。这就像有一个秘书在你说话的同时帮你做"
会议纪要的分章节"。

### 5.2 Auto-Dream：空闲整理

借鉴人在睡眠中"记忆巩固"的机制：

- Agent 检测到空闲（夜晚、用户离开、长时间无交互）时触发。
- 把若干天的 daily 笔记按主题、实体重新组织到 `digest/knowledge/{domain}/` 下；同时把对话里反复出现的偏好沉淀到 `digest/personal/`，把 Agent 完成任务的经验固化到 `digest/procedural/`。
- 抽取共性、合并重复、生成总结。

**用户体验**：第二天打开 `digest/`，会发现昨天散落在不同对话里的内容已经按"客户/项目/学习"自动归档到 `digest/knowledge/`，关键概念被抽成独立的主题笔记。

这是 ReMe 区别于"对话历史搜索"的关键 —— **它会自己整理**。

### 5.3 Auto-Link：自动建图

后台任务自动从正文里识别实体、候选链接，把隐式关系写回 wikilink：

- 在「光伏产业链」笔记里提到「隆基」，ReMe 自动补 `[[隆基]]` 链接到对应主题笔记。
- 在 daily 事件里提到「Alice」，自动链到 `[[Alice]]` 个人档案。
- 生成的 link 是可见的、可编辑的（写在 Markdown 文件里），用户随时可以修正。

**用户体验**：知识库随时间自然"越长越密"。浏览时可以从任意一处跳转到相关全部上下文，类似于在自己的脑子里"联想"。

### 5.4 三者协同：从对话到知识图谱的自然演化

```
[实时]                    [离线]                       [持续]
原始对话  ─Auto-Memory─►  daily 事件  ─Auto-Dream─►  digest/{personal,knowledge,procedural}
                                            │
                                       Auto-Link
                                            │
                                            ▼
                                       知识图谱
```

整个过程**不需要用户操心**。用户只需要正常和 Agent
对话，三个月后回头看，就有了一张按主题组织、互相关联、可视化浏览的个人知识图谱——这就是一句话定位里"自进化成知识图谱"的物理产物。


## 六、检索体验：多模检索 + 渐进式展开

### 6.1 三路融合的混合检索

ReMe 同时跑三种检索通路，结果通过 RRF（Reciprocal Rank Fusion）排序融合：

- **向量检索** —— 捕捉语义相似度（"钴" ≈ "锂电正极原料"）
- **关键词检索（BM25）** —— 精确匹配，对中文友好（"宁德时代" 一定要命中）
- **图谱检索** —— 通过 wikilink 邻居展开（找到"钴" → 自动带上"刚果(金)"、"嘉能可"）

单一通路都有盲区：

- 纯向量 → 名词术语容易错配。
- 纯关键词 → 同义改写抓不到。
- 纯图谱 → 起点选错就全盘错。

三路融合让检索像"三个人各自查一遍再开会确认"，结果鲁棒得多。

### 6.2 渐进式展开

传统 RAG 是一次性把 top-K 切片塞进上下文，token 利用率低，而且经常带进不相关的噪音。ReMe 的检索（
`reme4/steps/common/search.py`）是**分跳**的，且每一跳的"信息密度"刻意不同：

- **第一跳：直接命中的切片**——返回 chunk 全文 + 章节骨架。
- **第二跳：1-hop 邻居**——只返回邻居的 path + meta（name/description）+ 边的语义（predicate/anchor），**不展开正文**。
- **第 N 跳：Agent 主动追问**——基于二跳的"目录"，挑出真正相关的邻居，再发起新一次 search 拿正文。

**一个具体例子：分析师查询"钴的下游应用"**

第一跳直接命中 `digest/knowledge/financial/产业链/钴.md` 的某一段切片，answer 里这一段长这样：

```
========== digest/knowledge/financial/产业链/钴.md:42-78 [score=0.0234 vector=0.0123 keyword=0.0111] ==========
# 钴
## 应用
钴是锂电正极材料的关键原料，主要用于动力电池、消费电子和储能……

  → outlinks (3):
    → digest/knowledge/financial/矿产/刚果(金).md  name="刚果(金) - 钴矿主产区"
        via predicate=producer, anchor=#钴矿带
    → digest/knowledge/financial/公司/嘉能可.md  name="嘉能可 Glencore"
        via plain
    → digest/knowledge/financial/产品/三元正极.md  name="三元正极材料"
        via predicate=downstream
  ← inlinks (2):
    ← digest/knowledge/financial/产业链/锂电产业链.md  name="锂电产业链总览"
        via predicate=upstream
    ← daily/20260318/宁德调研纪要.md  name="宁德时代调研纪要"
        via plain
```

注意第二跳的信息只有"路径 + 名称 + 边的 predicate/anchor"，**没有邻居正文**。这是关键设计：

- 一次检索就让 Agent 看到"这个主题周围长什么样"——上游是刚果(金)、嘉能可，下游是三元正极，被锂电产业链当作 upstream 引用，最近还在
  3 月 18 日的宁德调研里被提到。
- Agent 可以基于这份"目录"判断哪个邻居才是用户真正想要的，再调一次 search 拉对应文件的正文（比如挑 `三元正极.md` 的细节）。

**为什么不一次把邻居正文也带回来**

如果第二跳直接返回正文，三跳网络很容易把上下文撑爆。当前实现里 `max_links_per_direction` 默认 10，单跳最多吐出 10 个
outlink + 10 个 inlink 的 meta，每条只占一行，**整张二跳目录的成本不到一个 chunk 的 token**。

**工程层面的关键参数**

- `candidate_multiplier=3.0`：候选池预拉 `limit*3` 条（最多 200），给 RRF 融合留余量。
- `min_score`：过低分切片直接丢弃，避免噪音。
- `expand_links=True`：开关二跳展开；关闭则退化为传统 RAG。
- `max_links_per_direction=10`：单方向（出/入）最多展示几个邻居，防爆。

**用户体验**：检索像"翻知识网络"——先看一眼周边目录，再决定要不要深入某一条线，而不是"拉一坨切片塞进上下文"。
**工程价值**：上下文窗口永远只装最相关的部分，token 成本可控；Agent 也能更精确地解释"我为什么知道这个"——因为它能引用
predicate=upstream、anchor=#应用 这种带语义的边。

### 6.3 关键词索引的工程价值

很多人忽视：**做中文知识库，关键词检索比向量更重要**。

ReMe 自研增量 BM25 倒排索引，配合 jieba 中文分词：

- 增量更新：新增/删除文件无需重建全索引。
- 跨平台：纯 Python + 文件落盘，没有 sqlite/chroma 这类原生扩展。
- 这一点直接解决了老版本在 qwenpaw 等老旧 Linux/Win 系统上的 core dump 兼容问题。

---

## 七、工程架构：可扩展、可替换、可演进

### 7.1 Component 框架

ReMe 把所有能力封装为 Component：

```
embedding · file_store · file_graph · file_chunker · file_watcher
tokenizer · keyword_index · LLM 适配 · service · client
```

每个 Component 都可以：

- **Backend 热切换**：`local` ↔ `nx` ↔ `neo4j` 一行配置改完。
- **生命周期托管**：start / close / restart 全自动，幂等保护。
- **依赖声明**：组件间相互调用，按依赖图拓扑排序自动启动。
- **持久化钩子**：dump/load 标准接口。

这意味着 ReMe 有非常强的**可演进性** —— 当某个 backend 不够用了（比如个人 Neo4j 改用云上 Neo4j），换的成本极低。

### 7.2 Job / Step 编排（借鉴 GitHub Actions）

- **Step**：最小执行单元，做一件具体的事（如检索、解析、调 LLM）。
- **Job**：steps 的有序组合，可复用、可流式。
- **对外**：每个 Job 同时暴露为 HTTP API / MCP Tool / CLI 命令，无需重复开发。

新增一个能力的标准动作是：

1. 写一个 Step（继承 BaseStep，实现 execute）。
2. 在配置里把它组合进 Job。
3. 自动获得 HTTP / MCP / CLI 三种调用方式。

### 7.3 配置即应用

一份 `default.yaml` 描述完整应用：service / components / jobs。

```yaml
service:
  backend: http
components:
  file_store:
    backend: local
  file_graph:
    backend: local
jobs:
  search:
    steps:
      - search_step
```

替换 backend、增删 Job、调整依赖，全部通过配置完成，部署上线无需改代码。

---

## 八、生态接入：ReMe 如何被使用

### 8.1 三种集成路径

| 路径                      | 适用对象                                                | 体验                                                                 |
|-------------------------|-----------------------------------------------------|--------------------------------------------------------------------|
| **SDK 集成**              | qwenpaw / AgentScope 等深度合作框架                        | 直接调用 `AgentscopeTools`，无感拥有 auto-memory / auto-dream / auto-search |
| **MCP Tool + skill.md** | 任何支持 MCP 的客户端（Claude Code / Cursor / Cherry Studio） | 配 skill.md，开箱即用                                                    |
| **CLI + skill.md**      | 通用方案，兜底所有 Harness                                   | 一条命令调用，shell 友好                                                    |

三条路径的设计哲学是：**不强迫任何 Agent 框架做 ReMe-specific 的改造**。

- 对深度合作方，给最丝滑的 SDK。
- 对支持 MCP 的产品，靠 MCP 标准协议。
- 对什么都不支持的环境，CLI + skill.md 兜底。

### 8.2 服务托管

- **按需拉起**：Agent 检测到 ReMe 服务未运行时，可以自动后台拉起，用户无感知。
- **服务发现**：`find_reme` 一键探活，避免端口冲突；多个 ReMe 实例共存时也能精准定位。

### 8.3 ReMe 的边界

> **ReMe 专注于知识加工，不做知识获取。**

- **数据采集** —— 网页抓取、邮件接入、Slack 同步、文件上传 —— 由上游 Agent 完成。
- **ReMe 负责** —— 把这些资料消化、整理、链接、检索、自进化。

这个边界划得清楚的好处：

- ReMe 不和上游的数据接入工具竞争。
- ReMe 不需要为每种数据源写适配，专注做记忆引擎本职。
- 让 ReMe 在"被集成"路线上更纯粹、更通用。

---

## 九、应用场景

### 9.1 金融场景：产业链知识库

**主角**：王分析师，新能源行业研究员，每天要处理 10+ 篇研报、数十条产业新闻、若干场公司调研。
**痛点**：信息散落在飞书文档、PDF 研报、微信群消息、调研纪要里，"上次调研宁德时代时聊到的钴价话题"再也找不回来。

#### 一周内 ReMe 自动织出的产业链图谱

> **关键边界**：Auto-Dream 只对**已存在的 daily 事实**做聚合，不会凭空梦出"产业链总览"这种结构性概念。总览级笔记的诞生依赖**用户主动 query**，下面会分两个阶段展示。

##### 阶段一：Auto-Memory + Auto-Dream（事实层聚合）

**Day 1（周一）盘后**：王分析师把今天看到的 3 篇研报扔给 Agent，又口述了对刚果(金)矿权变更的看法。

```
对话原文(片段)：
> 今天嘉能可发了三季报，钴产量同比下滑 18%……
> 刚果(金)那边的政策变化，对洛阳钼业 KFM 矿的影响要重点跟……
> 下游三元正极厂商已经开始转向高镍低钴方案……
```

ReMe 当晚 Auto-Memory 拆事件：

```
daily/20260518/
├── 嘉能可三季报点评.md       ← Auto-Memory 拆出的事件 1
├── 刚果金矿权政策跟踪.md      ← 事件 2
└── 三元正极高镍化趋势.md      ← 事件 3
```

**Day 2-3**：王分析师又陆续聊了宁德调研、亿纬电话会、嘉能可后续公告，daily 里「钴」「嘉能可」「三元正极」「宁德时代」反复出现。

**Day 3（周三）夜间 Auto-Dream**：把多天 daily 里**反复出现的实体**聚合成实体笔记——只做归并，不做总览。

```
digest/knowledge/financial/公司/
├── 嘉能可.md                 ← 新建：聚合 day1/day2 提到嘉能可的 4 个事件
├── 洛阳钼业.md               ← 新建
└── 宁德时代.md               ← 已有，本次新增「高镍化决策」一节
digest/knowledge/financial/原料/
└── 钴.md                     ← 新建：聚合 3 天里所有提到「钴」的内容
digest/knowledge/financial/产品/
└── 三元正极.md               ← 新建：技术路线变化
```

注意：**Auto-Dream 没有生成「锂电产业链.md」**——产业链是结构性概括，不在 daily 事实里，凭空生成就是幻觉。

##### 阶段二：用户 query 触发检索 + 合成（结构层）

**Day 5（周五）**：王分析师准备组会要讲新能源板块，主动问 Agent：

> **"分析锂电相关上下游"**

这一句 query 触发了**检索 → 合成 → 落盘**的完整闭环：

###### Step 1：渐进式检索返回多节点 + 关系骨架

ReMe 走多模融合（向量 + BM25 + 图谱），命中 3 天来 Auto-Dream 已经聚合好的实体节点，并展开 1-hop 邻居 meta：

```
========== 第一跳：直接命中（5 个节点）==========

digest/knowledge/financial/原料/钴.md:42-78  [score=0.0234]
# 钴 / ## 应用
钴是锂电正极材料的关键原料，主要用于动力电池……
  → outlinks (4):
    → digest/knowledge/financial/产品/三元正极.md     via predicate=downstream
    → digest/knowledge/financial/公司/嘉能可.md       via predicate=producer
    → digest/knowledge/financial/公司/洛阳钼业.md     via predicate=producer
    → daily/20260518/三元正极高镍化趋势.md            via plain
  ← inlinks (2):
    ← daily/20260318/宁德调研纪要.md  via plain
    ← daily/20260512/亿纬电话会.md    via plain

digest/knowledge/financial/产品/三元正极.md:15-44  [score=0.0211]
# 三元正极 / ## 高镍低钴路线
2025 年起主流厂商加速 8 系/9 系产品……
  → outlinks (2):
    → digest/knowledge/financial/公司/宁德时代.md     via predicate=used_by
    → digest/knowledge/financial/原料/钴.md           via predicate=upstream

digest/knowledge/financial/公司/宁德时代.md:88-120  [score=0.0193]
# 宁德时代 / ## 高镍化决策
本季度切换到 9 系三元为主……

digest/knowledge/financial/公司/嘉能可.md:5-30      [score=0.0167]
digest/knowledge/financial/公司/洛阳钼业.md:1-22    [score=0.0152]

========== 第二跳：Agent 主动展开邻居 meta（不取正文）==========
共 9 个邻居节点，按预测相关度排序：
  - digest/knowledge/financial/公司/亿纬锂能.md     ← 三元正极 used_by 反链
  - daily/20260512/亿纬电话会.md                    ← 宁德时代 inlinks
  - daily/20260318/宁德调研纪要.md                  ← 钴 inlinks
  ...
```

Agent 不需要把这些邻居正文都拉回来——光看"路径 + name + predicate"就足够拼出上下游骨架。

###### Step 2：Agent 基于检索结果合成总览，写回知识库

```
digest/knowledge/financial/产业链/
└── 锂电产业链.md             ← 由 Day 5 query 触发合成
                                  内容来源：上一步检索命中的 5 个节点 + 关系
                                  不引入任何 daily 之外的"想象"
```

`锂电产业链.md` 正文：

```markdown
---
name: 锂电产业链总览
source: query-synthesized                  ← 标记来源是 query 合成
trigger_query: "分析锂电相关上下游"
generated_at: 2026-05-22
generated_from:
  - [[钴]]
  - [[嘉能可]]
  - [[洛阳钼业]]
  - [[三元正极]]
  - [[宁德时代]]
---

## 上游 · 资源
- 钴矿：[[嘉能可]] / [[洛阳钼业]]（刚果金为主产区，详见 [[钴]]）

## 中游 · 材料
- 正极：[[三元正极]]（高镍化趋势，详见同名笔记）

## 下游 · 电池厂
- [[宁德时代]]（[[daily/20260318/宁德调研纪要]] 中已确认 9 系切换）
- [[亿纬锂能]]（[[daily/20260512/亿纬电话会]] 中提到产能规划）

> 本笔记由 query 触发合成；下次再问"锂电上下游"会直接命中此文件，
> 后续 Auto-Link 会在新 daily 事件出现相关实体时增量补连接。
```

###### Step 3：Agent 同步给出组会答复

Agent 拿这份合成结果，给王分析师的回复直接带**上下游骨架 + 公司归属 + 历史调研引用**：

> "锂电产业链分三段：上游钴矿（嘉能可、洛阳钼业，刚果金集中）、中游三元正极（高镍化加速）、下游电池厂（宁德/亿纬）。这周您提到的事件分别落在：嘉能可三季报 → 上游产能；高镍化趋势 → 中游路线切换；宁德 9 系切换 → 下游产品验证。详细引用见 [[digest/knowledge/financial/产业链/锂电产业链]]。"

**关键差异**：传统 RAG 会一股脑塞 5 个文件正文进上下文；ReMe 是"先看 5 个节点的目录骨架 → 拼出总览 → 落盘成可被反复消费的笔记"。下次再问"锂电下游有谁"，直接命中这份总览，不用再走一遍合成。

##### Day 7：图谱已经长出层次

```
                          ┌─────────────┐
                ┌────────►│ 锂电产业链   │◄────────┐
                │         └──────┬──────┘  ← Day 5 query 合成
                │  upstream      │                │ upstream
                │                │ contains       │
        ┌───────┴──────┐         ▼         ┌──────┴──────┐
        │   钴         │   ┌─────────┐     │   锂        │
        │ (刚果金产区) │◄──┤  原料   ├────►│ (盐湖产区)  │
        └───────┬──────┘   └────┬────┘     └─────────────┘
                │ producer      │ downstream
                ▼               ▼
        ┌──────────────┐   ┌──────────────┐
        │   嘉能可     │   │  三元正极    │◄── 高镍化趋势
        │   洛阳钼业   │   └──────┬───────┘
        └──────────────┘          │ used_by
                                  ▼
                          ┌──────────────┐
                          │   宁德时代   │  ← daily/0318 调研纪要
                          │   亿纬锂能   │  ← daily/0512 电话会
                          └──────────────┘
        【实体层 · Auto-Dream 聚合产生】 │
                                         │
                                  【结构层 · query 合成产生】
```

每条边都对应文件里的一句 `predicate:: [[X]]`，每个节点点开就是 Markdown 笔记，每段笔记都能反向追溯到原始 daily 事件——**没有任何节点是凭空"梦"出来的**。

#### proactive：主动洞察推送

每天早上 9:00，ReMe 在 `digest/proactive/20260519.md` 里写：

```markdown
---
name: 早间洞察 · 2026-05-19
---

## 与您近期关注主题相关的事件

- **嘉能可宣布刚果(金) Mutanda 矿复产** ← 关联 [[钴]] / [[嘉能可]]
  上周您在 [[daily/20260518/嘉能可三季报点评]] 中标注「关注复产节奏」。
  → 复产对钴价的边际影响估计 -5% 到 -8%，可能影响 [[三元正极]] 成本。

- **宁德时代发布麒麟电池新版本** ← 关联 [[宁德时代]] / [[三元正极]]
  上次调研（[[daily/20260318/宁德调研纪要]]）中提到的高镍方案已落地。
```

**这就是金融场景下 ReMe 的核心价值**：分析师只负责"看 + 说"，知识图谱自己长出来；当行业事件发生时，ReMe 主动把"新事件 ↔ 旧上下文"的连线送到分析师面前。

---

### 9.2 个人工作 & 生活第二大脑

**主角**：李工，前端工程师 + 业余跑者 + 有娃奶爸。每天和 Agent 聊工作 bug、读论文、讨论小孩教育、规划周末徒步路线。
**目标**：让所有这些零散的对话沉淀成一份"自己的"知识库，三个月后能用 Obsidian 直接打开浏览。

#### 时间线：从空目录到第二大脑

```
Day 1                    Day 7                    Day 30                   Day 90
  │                        │                        │                        │
  ▼                        ▼                        ▼                        ▼
[空目录]            [daily 流水开始堆积]      [knowledge 主题浮现]    [图谱密集成网]
                                                                          │
  ──────────────────────── Auto-Memory ─────────────────────────────►     │
                ──────────── Auto-Dream（每晚） ──────────────────►      │
                          ────── Auto-Link（持续） ─────────────►        │
                                                                          ▼
                                                                  Obsidian Graph
                                                                  打开是密集网状
```

#### Day 7：daily 流水

```
daily/
├── 20260513.md
├── 20260513/
│   ├── 调试登录页面 CSS 问题.md       ← 工作
│   ├── 读《深度工作》第三章.md         ← 学习
│   └── 周末徒步路线讨论.md             ← 生活
├── 20260514.md
├── 20260514/
│   ├── 团队周会决定切换到 pnpm.md     ← 工作
│   ├── 给宝宝挑选英语启蒙绘本.md      ← 育儿
│   └── 5km 配速训练记录.md             ← 跑步
└── ……
```

打开 `daily/20260513.md`（主索引）：

```markdown
---
name: 2026-05-13
---

## 今日事件

- 09:30 [[daily/20260513/调试登录页面 CSS 问题]] · #工作 #前端
- 14:20 [[daily/20260513/读《深度工作》第三章]] · #阅读
- 21:10 [[daily/20260513/周末徒步路线讨论]] · #生活 #徒步
```

#### Day 30：Auto-Dream 已经把主题归档好了

```
digest/knowledge/
├── life/                       ← 用户自定义二级分类：生活
│   ├── 跑步训练日志.md          ← 把 1 个月的「配速训练」事件聚合
│   ├── 阅读笔记/
│   │   ├── 深度工作.md          ← 全书要点（从多次 daily 阅读片段汇总）
│   │   └── 给孩子的诗.md
│   └── 徒步路线/
│       ├── 莫干山线.md
│       └── 千岛湖环湖线.md
├── work/                       ← 用户自定义二级分类：工作
│   ├── 前端调试技巧.md          ← 「调试登录页面 CSS」「修复 z-index」等事件聚合
│   ├── 包管理工具切换决策.md     ← 「pnpm vs npm」讨论沉淀
│   └── 团队会议纪要/
└── parenting/                  ← 用户自定义二级分类：育儿
    ├── 英语启蒙书单.md
    └── 与孩子沟通技巧.md
```

李工没有手动建过任何一个 `digest/knowledge/` 下的文件——它们都是 Agent 在他睡觉时从 daily 里"梦出来"的。同时 Agent 在 `digest/personal/` 沉淀着对李工的画像（"偏前端"、"早睡"、"周末徒步"），但这一层用户不会主动浏览。

#### Day 90：Obsidian Graph View 打开是这样

```
                                    [深度工作]
                                       ▲
                              引用      │  应用到
                                       │
   [跑步训练日志] ◄── 借鉴方法 ── [前端调试技巧] ──► [包管理工具切换决策]
        │                          ▲   ▲
        │ 关联                      │   │ 提到
        ▼                          │   │
   [徒步路线]                  [周会纪要] [团队成员]
        │                                       │
        │                                       │ mentions
        ▼                                       ▼
   [莫干山线] ───── 同行 ────► [Alice] ◄── 育儿讨论 ── [给孩子的诗]
                                                          │
                                                          ▼
                                                   [英语启蒙书单]
```

**这张图是李工的"第二大脑"**——工作、跑步、阅读、育儿、社交关系全部交织在一起，能从「Alice」一路联想到「莫干山徒步」，再跳到「孩子的英语书单」，因为某次徒步同行时聊到过这个话题。

#### 一次具体的"联想式回忆"

某天李工问："**上次和 Alice 一起聊过的那本书叫什么？**"

```
========== 第一跳：daily 命中 ==========
daily/20260420/与 Alice 周末聚餐.md
> ……Alice 推荐了一本讲注意力的书，标题里有「深度」两个字……

  ← inlinks:
    ← digest/knowledge/life/阅读笔记/深度工作.md  via plain
  → outlinks:
    → digest/personal/Alice.md  via mention

========== 第二跳：Agent 顺藤摸瓜 ==========
digest/knowledge/life/阅读笔记/深度工作.md
> 卡尔·纽波特，2016 年出版……
```

Agent 回："**《深度工作》，卡尔·纽波特著。** 您在 4/20 周末聚餐时 Alice 推荐的，您后来在 5/13 读了第三章并做了笔记。"

模糊回忆 → 精确召回 → 上下文重建——这是普通"对话历史搜索"做不到的，因为它只会按时间顺序往回翻，而 ReMe 沿着图谱"联想"。

---

### 9.3 Agent 长期陪伴：跨会话的程序化记忆

**主角**：张研发，长期使用 Claude Code 做日常开发。希望 Agent "越用越懂我"——记得我的代码风格偏好，记得过去踩过的坑，记得未完成的任务。

#### 跨会话的记忆生命周期

```
[第 1 次会话]                [第 N 次会话，N 周后]
   │                                │
   │ 用户在编辑器报错               │ 用户遇到类似报错
   ▼                                ▼
┌─────────────┐                ┌─────────────┐
│ Agent 排查  │                │ Agent 检索  │
│ 试 A 方案 ❌│   ── ReMe ──►  │ ReMe 召回    │
│ 试 B 方案 ✅│                │ 上次的 B 方案│
└──────┬──────┘                └──────┬──────┘
       │ 写入                          │ 直接套用
       ▼                              ▼
digest/                           跳过踩坑，1 步解决
├── procedural/
│   └── webpack 编译卡死.md       ← Agent 任务经验
└── personal/
    └── 代码风格.md               ← 用户偏好画像
```

#### 程序化记忆的真实例子

**第 1 次会话（2026-03-10）**：webpack 编译突然卡死。

```
对话过程(摘要)：
- 用户："npm run build 卡在 92% 不动了"
- Agent 试方案 A：清缓存 → 没用 ❌
- Agent 试方案 B：升级 terser-webpack-plugin → 没用 ❌
- Agent 试方案 C：发现是 fork-ts-checker 的 OOM，加 --max-old-space-size=8192 → ✅ 成功
```

ReMe 的 Auto-Dream 当晚把这次会话沉淀到：

```markdown
# digest/procedural/webpack 编译卡死.md
---
name: webpack 编译卡死的排查路径
type: programmatic
---

## 症状
build 卡在 92%（chunk asset optimization 阶段）。

## 排查顺序（按命中率）
1. ✅ **检查 Node 内存**：`NODE_OPTIONS=--max-old-space-size=8192`
   → fork-ts-checker 在大型 TS 项目里很容易 OOM
2. ❌ 清 .cache 缓存（多数情况下无效）
3. ❌ 升级 terser-webpack-plugin（仅特定版本相关）

## 关联
- 用户偏好：[[digest/personal/代码风格#TypeScript]] 启用了 strict 模式，
  类型检查负担大，更易触发 OOM
- 历史事件：[[daily/20260310/webpack 卡死排查]]
```

**第 2 次会话（2026-04-22，6 周后）**：用户在另一个项目又遇到类似问题。

```
用户："vite build 也卡在打包阶段，是同一类问题吗？"

Agent 内部检索 ReMe：
========== 命中：digest/procedural/webpack 编译卡死.md ==========
关键词匹配：编译卡死、build 阶段
向量相似：vite build ≈ webpack build

Agent 回复：
"根据您 3/10 在另一个项目遇到的同类问题，建议优先检查 Node 内存。
vite 的 esbuild 阶段也吃内存，先试 NODE_OPTIONS=--max-old-space-size=8192。
（上次清缓存和升级 plugin 都没用，可以跳过）"
```

**节省的不是时间，是踩坑路径**——这就是程序化记忆的价值。

#### 用户偏好的"自我画像"

`digest/personal/` 下随时间长出的画像：

```markdown
# digest/personal/代码风格.md
---
type: personalization
---

## 命名
- 倾向：函数 camelCase、类型 PascalCase、常量 SCREAMING_SNAKE
- 来源：[[daily/20260215]] 多次纠正 Agent 的命名建议

## 注释
- 倾向：**不写无意义注释**，只在 WHY 不显然时写
- 来源：[[daily/20260301]] 用户原话："don't comment what the code already says"

## 错误处理
- 倾向：边界处校验、内部代码相信调用方
- 来源：[[daily/20260408]] 用户拒绝在内部函数加 try/catch 时的解释

## 测试组织
- 倾向：tests4/unittest 按基类组织（来自项目 CLAUDE.md）
```

**意义**：这不是 Agent 在 system prompt 里写死的"用户喜欢简洁"，而是**从用户实际行为里被动观察到的、可追溯到具体对话的偏好画像**。每条偏好都有 `[[daily/...]]` 反向链接，用户可以审视、可以修正。

#### 三类记忆在 Agent 陪伴里的分工

回到 [3.2](#32-digest-下的四种记忆) 中 `digest/` 的四个子目录，本场景主要由其中三类共同支撑（proactive 已在 9.1 主动推送场景演示）：

| digest 子目录                  | 写入触发           | 检索权重     | 实际表现                                  |
|----------------------------|----------------|----------|---------------------------------------|
| **digest/personal/**       | 用户纠正 / 偏好表达    | 全场景常驻    | "我懂你不爱写注释"                            |
| **digest/procedural/**     | 任务完成后归纳成功/失败路径 | 任务相似度高时高 | "上次这类 bug 你这样解决过"                     |
| **digest/knowledge/**      | 学习对话、文档阅读      | 主题相关时高   | "你之前学过的 React Server Components"      |

三类共同织成一个"懂用户 + 会做事 + 有知识"的长期陪伴 Agent——**差异化体验来自 ReMe 维护的个人记忆，而不是模型本身**。换言之，同一个 Claude / Qwen 模型，套上不同用户的 ReMe，会变成完全不同的 Agent。

---

## 十、性能与稳定性

### 10.1 自研轻量内核

新版本重写了记忆引擎的核心模块：

- **file chunker** —— Markdown AST + 章节切片 + wikilink 抽取
- **file store** —— 内存 chunk 字典 + JSONL 持久化
- **file graph** —— 双向链接索引，多 backend
- **file watcher** —— 基于 watchfiles 的轻量监听
- **keyword index** —— 自研增量 BM25 倒排，原生支持中文

整体**纯 Python + 文件持久化**，无 sqlite/chroma 等三方原生依赖。

### 10.2 跨平台稳定性

老版本在 qwenpaw 等低版本 Linux/Win 环境会出现 sqlite 段错误、chroma core dump，这些问题在新版本完全规避：

- 没有 native 扩展依赖。
- 老旧 glibc / 老旧 Python 版本也能跑。
- 安装简单，不需要 cmake、build-essential。

这对一个**要被部署到大量异构用户机器**的产品至关重要。

### 10.3 未来：Rust / C++ 高性能内核

- 当前 Python 版本已能覆盖个人规模知识库（万级文件）。
- 规划用 Rust / C++ 重写关键路径（BM25 索引、文件解析、向量计算），支撑：
    - 更大规模（十万级文件）
    - 更低延迟（亚秒级冷启动）
    - 更小内存
- 上层 API 不变，对用户和 Agent 接入方完全透明。

---

## 十一、Roadmap

| 阶段        | 关键里程碑                                                                                                  |
|-----------|--------------------------------------------------------------------------------------------------------|
| **Now**   | 组件框架、Job/Step、Markdown 内核、混合检索（向量+BM25+图谱）、HTTP / MCP / CLI 三协议服务                                      |
| **Next**  | auto-memory / auto-dream / auto-link 全套自进化能力；记忆类型分层；resource / proactive 目录；skill.md 模板；qwenpaw SDK 集成 |
| **Later** | 多跳渐进检索 API、领域 demo（金融产业链）、个人场景模板包、Rust 高性能内核、可视化管理面板                                                   |

每个阶段都有清晰的对外可演示成果：

- Now → 可以现场演示 ReMe 检索 + Agent 集成。
- Next → 可以演示"今天聊的内容明天自动整理好"。
- Later → 可以演示十万级知识库下的亚秒检索 + 主动推送闭环。

---

## 十二、结语：ReMe 想成为什么

> ReMe 不止是「记忆库」。
>
> 它的目标是：**让每个用户拥有一张由本地 Markdown 自进化而成、可携带、可被任意 Agent 调用的个人知识图谱。**
>
> 当 Agent 时代真正到来时，差异化的不是模型，而是「这个 Agent 是不是了解我」。
>
> ReMe 想做的，就是这份「了解」的载体——一张属于用户自己、Agent 可读可写、会自己生长的图谱。

**三个判断**：

1. 个人记忆是 Agent 时代必然出现的基础设施 —— 不是 ReMe 不做就没人做，而是早做的人有先发优势。
2. **本地 Markdown + 自进化 + 知识图谱**（叠加被集成路线）—— 这套组合在当下市场是空缺的。
3. ReMe 的工程内核已经就位，剩下是**自进化能力 + 生态集成 + 场景模板**的三件套加固，路径明确。
