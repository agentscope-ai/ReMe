# reme 系统架构 — 设计文档

## 文档定位

本文档定义 reme 的**架构设计**:概念边界、数据流契约、职责划分。

- 不涉及代码路径 / 实现进度 / API 具体形态
- **执行栈与架构角色**(分层、模块切分、职责划分)在第 6-7 节;源码映射与落地状态见 `docs4/reme_report.md` 与源码
- 不规定怎么做,只规定**是什么、谁负责、输入输出**

> 阅读顺序:**第 1 节**给出完整的架构总览(数据视角 + 运行时视角 + 核心机制 + 不变量 + 导航);**第 2-5 节**逐层展开三层存储 / 6 类 L4 动作语义 / 反向回流 / 触发节奏;**第 6 节**描述底层执行栈(L0→L5);**第 7 节**把 6 类 Action 落到 L4 实现模块;**第 8 节**讲跨切面 schema 契约;**第 9 节**列出明确不属于本架构的反例。

---

## 1. 架构总览

本章给出 reme 完整的设计骨架,后续 §2-§9 逐项展开细节。

### 1.1 解决什么问题

reme 是 agent 的**长期记忆系统**。它把 agent 的工作过程沉淀为可检索、可演化的知识结构。

设计要解耦两件事:

| 关注点 | 由谁负责 |
|---|---|
| **agent 写什么 / 读什么** | agent 的工作流自决 |
| **workspace 自身如何健康演化** | reme 自治,agent 不感知 |

实现方式:三层存储拓扑为"两路并行写入(原始资料 / 任务过程) → 双源合流到沉淀知识层",reme 提供这三层的容器、动作语义、反向检索与自治维护。

### 1.2 数据视角:并行起点 + 双源合流 + 反向回流

数据有**两个并行起点** —— **External source**(webhook/upload/pull)与 **Agent**(外部主体,自身任务驱动)。两条独立通道各自落地到**并行的材料层**:External 经 `ingest` 沉到 `resource/`,Agent 经 `sync` 写到 `daily/`。两层材料**合流**到 `digest/`,由 reme 通过 `digest` 动作完成"消化"。`digest/` 自身由 `maintain` 做 in-place 重组。Agent 通过 `retrieve` 从 resource + daily + digest **三层并行**回流。`notify` 是 reme 跨过 workspace 直接提醒 Agent 的**虚边**(控制信号,不写任何文件)。每段路径都对应一个 L4 动作语义(完整动作详见 §5.2 / §7):

| 路径 | L4 动作 | 说明 |
|---|---|---|
| External source → resource             | **ingest**   | 外部信源落到 workspace |
| External source ╌╌► Agent              | **notify**   | reme 推送通知(虚边,只走 L2 推送队列,不写文件) |
| Agent → daily                          | **sync**     | Agent 写 daily(响应 notify 或自身任务驱动) |
| resource + daily → digest              | **digest**   | reme 内部 LLM 抽取沉淀(双源合流) |
| digest → digest                        | **maintain** | reme 内部 LLM 折叠重组(in-place, fold-only) |
| resource + daily + digest → Agent      | **retrieve** | 三层并行回流(state / semantic / topological 三种正交问法) |

```
                                notify(虚边,控制信号)
External source ╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌►  AGENT
(webhook/upload/pull)                                     (外部主体/自身任务)
        │                                                    │    ▲
        │ ingest                                             │    │ retrieve
        │                  ┌────── sync ────────────────────-┘    │ (state /
        ▼                  ▼                                      │  semantic /
   ┌──────────┐       ┌──────────┐                                │  topological)
   │resource/ │       │  daily/  │                                │
   │ 原始资料  │       │任务工作区  │───────── retrieve ─────────────┤
   │ 不可变    │       │ 半可变    │                                │
   └───┬──┬───┘       └────┬─────┘                                │
       │  │                │                                      │
       │  │ digest  digest │                                      │
       │  └────────┐  ┌────┘                                      │
       │           ▼  ▼                                           │
       │       ┌────────────────────┐                             │
       │       │      digest/       │ ◄──╮                        │
       │       │     沉淀知识        │    │ maintain               │
       │       │  可重组(语义索引)    │ ───╯ (in-place, fold-only)  │
       │       └──────────┬─────────┘                             │
       │                  │ retrieve                              │
       │                  └─────────────────────────────────────-─┤
       │                                                          │
       │ retrieve                                                 │
       └──────────────────────────────────────────────────────────┘
```

**模型要点**:

- **两个起点平行,不存在主从** —— External 与 Agent 各自独立驱动;Agent 既可响应 `notify` 也可由自身任务直接 `sync`。
- **两层材料平行,不存在传递** —— resource 与 daily 是**两条独立的写入通道**,不互相穿越:Agent 不写 resource,ingester 不写 daily。
- **digest 是双源合流的产物** —— `digest` 动作的输入是 resource + daily 的组合(不是仅 daily);相应地,digest 节点的 provenance 可同时指向 resource 与 daily。
- **digest 自循环** —— `maintain` 在 digest 内部做密度折叠,不与上游材料层交互。
- **notify 是虚边** —— reme 用它提醒 Agent "有新 resource 值得看",但不落任何文件;Agent 的响应通过 `sync` 落 daily(并可选地用 wikilink 引 resource)。

retrieve 三种问法正交:

| 问法 | 工具 | 主要看哪层 |
|---|---|---|
| **state**(谁在 / 是什么状态) | `list` / `frontmatter` | 各层平等 |
| **semantic**(我想到一个意思) | `search` | digest > daily > resource(默认权重) |
| **topological**(从一个点向外摸) | `traverse` | 沿 wikilink 跨层平等 |

### 1.3 运行时视角:六层执行栈 + 双进程

reme 的功能不是堆在一层,而是从文件系统底层往上栈式堆叠。顶层 Service 与 Runtime 是同一套 workspace 上的两个进程角色,共享 L0-L4 全栈(详见 §5.3 / §6)。

```
   ┌─────────────────────┐      ┌─────────────────────┐
   │  L5  Service         │      │  L5  Runtime         │
   │  (HTTP / MCP)        │      │  (scheduler 自治)    │
   └──────────┬───────────┘      └──────────┬──────────┘
              │                              │
              └─────────────┬────────────────┘
                            ▼
   ┌──────────────────────────────────────────────────┐
   │  L4   6 类 Action(动作语义)                     │
   │  ingest  notify  sync  retrieve  digest  maintain│
   └──────────────────────┬───────────────────────────┘
                          ▼
   ┌──────────────────────────────────────────────────┐
   │  L3   原子工具                                   │
   │  ┌─────────────────────┐  ┌──────────────────┐   │
   │  │ 基础工具             │  │ 高级工具          │   │
   │  │ create/append/edit/  │  │ search           │   │
   │  │ read/write/move/     │  │ traverse         │   │
   │  │ delete/list/stat     │  │ frontmatter      │   │
   │  └──────────┬──────────┘  └────────┬─────────┘   │
   └─────────────│──────────────────────│─────────────┘
                 │ 直读 / 直写            │ 走索引读
                 │ (eventual,有滞后)    │
                 │                       ▼
                 │       ┌────────────────────────────┐
                 │       │  L2   文件状态              │
                 │       │  · file_store(chunk+vec)   │
                 │       │  · file_graph(node+link)   │
                 │       │  · 自治状态(scheduler 用): │
                 │       │    - resource: 入流批次/    │
                 │       │      未消化(orphan)        │
                 │       │    - daily: 任务索引        │
                 │       │      (进行中/stale/完成)   │
                 │       │    - digest: 密度水位/      │
                 │       │      断链(broken wikilink) │
                 │       │  · 推送队列(notify):       │
                 │       │    pending/notified/        │
                 │       │    acknowledged             │
                 │       └─────────────▲──────────────┘
                 │                     │ 派生 / 更新
                 │       ┌─────────────┴──────────────┐
                 │       │  L1   file_watcher          │
                 │       │  fs event → state delta     │
                 │       │  (唯一 fs→state 桥)         │
                 │       └─────────────▲──────────────┘
                 │                     │ 监听
                 ▼                     │
   ┌──────────────────────────────────────────────────┐
   │  L0   workspace 文件系统                             │
   │  resource/    daily/    digest/                  │
   └──────────────────────────────────────────────────┘
```

Service 与 Runtime 是同一份 workspace 上的两个进程角色:

| 进程 | 触发源 | 时延敏感 | 典型动作 |
|---|---|---|---|
| **Service** | 外部 push / 外部 pull / agent 同步请求 / **MCP 推送通道** | 是 | ingest / sync / retrieve / **notify-out(MCP)** |
| **Runtime** | scheduler 周期 + L2 自治状态阈值 | 否(eventual) | **notify 决策** / digest / maintain |

### 1.4 核心机制总览

| 机制 | 一句话 | 详见 |
|---|---|---|
| 三层存储 | resource(冷) / daily(温) / digest(冷,组织化) | §2 |
| 6 类 L4 动作 | ingest / notify / sync / retrieve / digest / maintain | §1.2 / §3 |
| notify+sync 链 | reme 主动从 L2 资源自治状态选候选,经 MCP 推给 agent;agent sync 落 daily | §3.3-3.4 / §7.1 |
| 反向 retrieval | state / semantic / topological 三种正交问法 | §4 |
| 触发四源 | 外部 push / 外部 pull / agent on-demand / reme 后台 | §5.1-§5.2 |
| Service + Runtime | 双进程角色,共享 L0-L4,职责按时延分 | §5.3 |
| 执行栈(L0-L5) | filesystem → file_watcher → 文件状态 → 原子工具 → Action → Service/Runtime | §6 |
| 原子工具:基础 vs 高级 | 基础直 fs;高级走 L2 索引(eventual) | §6.4 / §5.5 |
| Action 模块映射 | 6 类 Action 由 5 个模块实现(retrieve 直走原子工具) | §7 |
| Schema 跨切面 | name+description 是核心强约束,其余 opinionated default 可重载 | §8 |

### 1.5 核心不变量速览

写入拓扑(架构脊梁,来自 §2.3):两路并行写入(External→resource、Agent→daily)→ 双源合流到 digest;resource 与 daily 之间互不写入;任何一层都不能反向改写它的上游。

| 不变量 | 内容 | 来源 |
|---|---|---|
| **I-1** | agent 不直接写 digest(digest 写权只属 dreamer / maintainer) | §2.4 |
| **I-2** | daily folder 单作者(同 folder 不并发改) | §2.4 |
| **I-3** | resource 内容不可变,只允许 metadata appendable | §2.4 |
| **I-4** | 三层共用同一套 wikilink 索引,跨层引用全靠 wikilink | §2.4 |
| **R-1** | retrieve 三种问法分立,不合并为单一 read verb | §4.3 |
| **M-1** | Maintainer 只做一件事:密度折叠(把碎片叶子折叠到新的中间节点下) | §7.3 |
| **F-1** | L1 `file_watcher` 是 L0→L2 的唯一派生桥 | §6.5 |
| **F-2** | L3 基础工具直接读写 L0;高级工具只走 L2 | §6.5 |
| **F-5** | L5 Service / Runtime 共享 L0-L4,不直接通信 | §6.5 |
| **F-6** | L0↔L2 存在 eventual 窗口,agent 上下文承担近期信息 | §5.5 / §6.5 |

### 1.6 文档导航

| 想了解… | 看 |
|---|---|
| 三层各自的定位、不变量 | §2 |
| 6 类 L4 动作语义(notify / sync / digest / maintain 的输入产出不变量) | §3 |
| Retrieval 的三种问法与跨层语义 | §4 |
| 谁来触发、什么节奏、为什么分两个进程 | §5 |
| 系统从文件系统到 Service 的分层(底层基础) | §6 |
| L4 五个模块的对称结构与 Maintainer 折叠设计 | §7 |
| Schema 协议与重载机制 | §8 |
| 哪些设计不属于本架构(反例与边界) | §9 |
| 术语回查 | 附录 |

---

## 2. 三层存储

### 2.1 一句话定位

| 层 | 一句话 |
|---|---|
| **resource/** | 外部原始资料的**不可变快照**。reme 是容器,不是作者。 |
| **daily/** | agent 的**任务工作区**。folder 是单位,以"日 + 任务"为索引。 |
| **digest/** | 跨任务沉淀的**有组织知识**。以语义(概念/实体/方法)为索引,与时间无关。 |

### 2.2 五维度对照

| 维度 | resource/ | daily/ | digest/ |
|---|---|---|---|
| **组织主轴** | 时间(`<date>/<name>`) | 时间 + 任务(`<date>/<slug>/`) | 语义(`<slug>/<subslug>/...`,任意嵌套) |
| **写权归属** | 入流通道唯一(webhook / upload / pull) | agent(写入任务过程) | dreamer / maintainer(无 agent 直写) |
| **可变性** | 不可变,只追加新文件 | folder 内可反复更新 | 单节点可演化,可被合并/拆分/移动 |
| **不变量** | 写入即冻结,原文永不变 | folder 名 = summary note 名(可移动单元);同 slug 同日只一份 | slug 全局唯一;每 folder 有 canonical entry;wikilink 全路径 |
| **谁在用** | agent(查原文)、dreamer(双源输入之一) | agent(自己的工作记录)、dreamer(双源输入之一) | agent(召回主目标)、maintainer(自维护对象) |

### 2.3 写入纪律:并行写入 + 双源合流

```
   External source                       Agent
         │                                 │
         │ ingest                          │ sync
         ▼                                 ▼
    ┌──────────┐                      ┌──────────┐
    │resource/ │                      │  daily/  │
    │ 不可变   │                      │ 半可变   │
    │(ingester)│                      │ (agent)  │
    └─────┬────┘                      └─────┬────┘
          │                                 │
          │ digest                  digest  │
          └──────────────┐    ┌─────────────┘
                         ▼    ▼
                    ┌──────────────┐  ◄──╮
                    │   digest/    │     │ maintain
                    │  可重组      │ ────╯ (in-place,
                    │ (dreamer +  │       fold-only)
                    │  maintainer) │
                    └──────────────┘
```

写权按这个**两层并行 → 单层合流**的拓扑分配:resource 写权专属 ingester(外部入流通道),daily 写权专属 agent(sync 落入,可响应 notify 或自身任务驱动),digest 写权专属 dreamer + maintainer。**resource 与 daily 之间互不写入**(agent 不动 resource,ingester 不动 daily);任何一层都不能反向改写它的上游。这是整个架构的脊梁。

### 2.4 不变量(永远成立)

| # | 不变量 | 否则后果 |
|---|---|---|
| **I-1** | agent 不直接写 digest | digest 的"有组织"性失守,沉淀质量退化 |
| **I-2** | daily folder 单作者(同 folder 不并发改) | 任务边界模糊,sync/digest 竞态 |
| **I-3** | resource content immutable,只允许 metadata appendable | 原文可能消失/被覆写,citation 不可信 |
| **I-4** | 三层共用同一套 wikilink 索引,跨层引用全靠 wikilink | 引入第二套引用机制 → 索引重建复杂 / 跨层关系不可达 |

---

## 3. L4 动作语义详解

§1.2 给出了 6 个 L4 动作在数据视角下的整体形态。本节按动作逐个展开输入 / 产出 / 不变量 / 反例。`ingest`(外部→resource,机械)和 `retrieve`(三层并行回流,只读)分别在 §5/§7.4 与 §4 详述,本节聚焦四个**写动作**:`notify` / `sync` / `digest` / `maintain`。

### 3.1 统一原则

四个写动作都遵守:

| 原则 | 内容 |
|---|---|
| **Monotonic content** | 上游内容不可变,下游只能新建节点或加链接,不能改写上游 |
| **Provenance 必须可达** | 任何下游节点必须能通过 wikilink 反查到上游来源 |
| **Wikilink 是新结构的唯一载体** | 跨层关系靠 wikilink,不靠内容拷贝 |

它们都不是"数据搬家",而是"在下游新生成有引用关系的节点"。

### 3.2 四个写动作的本质对照

| 动作 | 上游 → 下游 | 性质 | 上游变化 | 下游变化 |
|---|---|---|---|---|
| **notify** | resource → Agent | **Attention**(推送注意力) | 不变 | 不写 workspace;仅入 L2 推送队列 |
| **sync** | Agent → daily | **Reference**(引用落地) | 不变 | daily 中新增工作记录 + 对 resource 的 wikilink |
| **digest** | resource + daily → digest | **Crystallize**(双源合流结晶) | 不变 | digest 新增节点,wikilink 反指上游来源(daily 与/或 resource) |
| **maintain** | digest → digest | **Reorganize**(重组) | 结构变,内容守恒 | fold-only:引入子中间节点搬叶子,改变拓扑 |

> `notify` 与 `sync` 共同实现"resource 中的候选被 Agent 看见并织入 daily"这条**Reference 链**;它们是两个独立的 L4 动作,主体不同(notify 由 Reme 自治触发,sync 由 Agent 触发)。

### 3.3 notify:reme → Agent 推送候选

| 维度 | 内容 |
|---|---|
| 主体 | Reme Runtime(`notifier` 模块) |
| 输入 | L2 资源自治状态:orphan(无 inbound wikilink)/ 入流批次 / 未消化老于 N |
| 产出 | L2 推送队列条目;通过 Service MCP **server-initiated notification** 推到 Agent;**不写任何 workspace 文件** |
| 不变量 | resource 原文 0 修改;**完全单向**,不维护任何反向元数据;`notify` 决策在 Runtime,Service 只作 MCP transport |
| Acknowledge 机制 | L1 watcher 检测到 daily→resource 新 wikilink → L2 推送状态 `notified` → `acknowledged`,避免重复推送 |
| 反例 | (a) Service 自决推什么 notify(✗-17);(b) notify 写入 workspace(✗-16);(c) Agent 主动调 notify(✗-18) |

### 3.4 sync:Agent → daily 落地

| 维度 | 内容 |
|---|---|
| 主体 | Agent(`synchronizer` 模块在 Service 内编排) |
| 输入 | Agent 当前事件流(响应 `notify` 的候选,**或**自身任务直接驱动) |
| 产出 | daily folder 内的工作叙事;可选地用全路径 wikilink 引 resource(agent 自决,reme 不强制) |
| 不变量 | resource 原文 0 修改;daily 单作者(I-2);folder 名 = summary note 名(可移动单元) |
| Provenance | daily → resource 可达(通过 daily body 中的 wikilink) |
| 反例 | "agent 把 resource 内容拷进 daily" —— 不允许,daily 只持有引用 + 自己的工作记录 |

### 3.5 digest:resource + daily → digest 双源合流结晶

| 维度 | 内容 |
|---|---|
| 主体 | Reme Runtime(`dreamer` 模块,LLM-driven) |
| 输入 | 一组待蒸馏的 daily folder + 相关 resource(双源合流;通常以 daily 任务为线索,顺着 wikilink / 同主题搜索拉入相关 resource 原文) |
| 产出 | digest 中 0~N 个新节点 或 已有节点的更新;新节点必须用 wikilink 反指至少一个上游来源 |
| 不变量 | resource / daily 正文 0 修改;digest 新节点必须 wikilink 反指上游(provenance);digest 节点遵守第 2.4 节列的不变量 |
| Provenance | digest → daily / resource 双源链条可达(资料源是 resource 时直接反指,任务过程是 daily 时反指 daily 进而可达 resource) |
| 反例 | dreamer 改写 resource;dreamer 改写 daily 正文 |

### 3.6 maintain:digest → digest 折叠

| 维度 | 内容 |
|---|---|
| 主体 | Reme Runtime(`maintainer` 模块,LLM-driven,**fold-only**) |
| 输入 | digest/ 当前整体状态 |
| 产出 | 同一 digest/ 树的**密度折叠**(fold):某中间节点下叶子过多时,引入子中间节点把相关叶子归簇 + 写"高密度摘要" |
| 不变量 | 树**只向下生长**(从不反向);叶子内容 0 修改,只被搬位置;新中间节点 = 一个高密度摘要文件;任何节点移动**原子重写所有入边**(retarget);slug 全局唯一在折叠后仍成立 |
| Provenance | digest → daily 的反指链接在折叠后仍有效(retarget 保证) |
| 反例 | merge / move / promote / demote 等改写既有拓扑的操作;改写既有叶子内容 |

> 折叠操作的承诺与决策点见 §7.3。

### 3.7 链接重定向例外

`maintain` 的 retarget 会改写其它节点中指向被移动节点的 wikilink。从字面看,这违反了"上游内容不可变"。

实际上这是 wikilink 系统的**机械性副作用**,不算下游写上游:

| 字面 | 实质 |
|---|---|
| daily 里的 `[[digest/old.md]]` 被改成 `[[digest/new.md]]` | 作者意图("我引用了 X 这个 digest 节点")没变,只是 X 的物理位置变了 |

只要 retarget 保持 wikilink 的**目标语义不变**,就允许它作为机械维护副作用穿越层界。这是这条规则的唯一例外。

---

## 4. Retrieval 反向回流

Retrieval 是把 §3 几条正向写动作反着读:agent 站在结果端,沿 wikilink 反查源头。

### 4.1 三种问法

按 agent 意图分,有三类完全不同的读需求,**正交**,各自独立:

| 问法 | 例子 | 本质 |
|---|---|---|
| **状态问** (State) | "我有哪些 in-progress 的任务?""哪些 resource 还没被引用?" | 在某层做 list + frontmatter 过滤 |
| **语义问** (Semantic) | "关于 auth 重构我知道什么?" | 跨层全文/向量检索 |
| **拓扑问** (Topological) | "auth 概念周围都连了什么?" | 从某节点沿 wikilink 走 |

### 4.2 三层 × 三问法 矩阵

| 问法 | resource/ | daily/ | digest/ |
|---|---|---|---|
| **状态问** | "未处理 resource 清单" | "active / pending-digest 清单" | "孤儿节点 / canonical 缺失 清单"(给 maintain 用) |
| **语义问** | 兜底(原文,信噪比低) | 次优(最新,但未沉淀) | **首选**(沉淀过,信噪比高) |
| **拓扑问** | 通常是叶子(被指向) | daily → resource / digest | digest 内部连接最密 |

### 4.3 设计原则

| # | 原则 | 含义 |
|---|---|---|
| **R-1** | 三种问法分立,不合并为单一 "read" verb | 不同问法的索引、过滤、排序逻辑完全不同 |
| **R-2** | 语义问的默认权重 `digest > daily > resource`,**可被显式覆盖** | 默认体现"沉淀质量",但 agent 可指定单层或调权 |
| **R-3** | 拓扑问与层无关 | traverse 沿 wikilink 走,天然跨三层(I-4) |
| **R-4** | Provenance expansion 默认 lazy,eager 是上层便利封装 | 原子 retrieval 不自动展开;agent 需要时再 traverse |
| **R-5** | Cold start 不是新的 retrieval mode | 只是状态问 + 语义问的组合,reme 不为它单设 verb |

### 4.4 在主干图里的位置

Retrieval 不引入新存储,不引入新层。它是 **agent 与三层存储之间的读视图**,通过三种正交问法暴露,共享同一套 wikilink 索引(I-4)。

---

## 5. 节奏与触发

锁定"谁推动每个动作发生"。这一步定 reme 是纯被动 service 还是带后台 runtime。

### 5.1 触发源四分类

| 触发源 | 性质 | 例子 |
|---|---|---|
| **External push** | 外部事件主动推 | webhook / 用户 upload |
| **External pull** | reme 主动去外部拉 | scheduled fetcher(RSS / 邮件 / API 轮询) |
| **Agent on-demand** | agent 在请求里显式调用 | "sync 我的对话" / "搜 X" |
| **Reme background** | reme 自己的 watcher / scheduler | file_watcher / cron-like |

### 5.2 六个动作的触发归属

| 动作 | 主触发 | 备用触发 | 备注 |
|---|---|---|---|
| **ingest** | External push / pull | — | 外部→resource,机械 |
| **notify** | **Reme background**(cron + L2 资源自治状态阈值) | — | Runtime 决策,Service MCP 推送 |
| **sync** | Agent on-demand | — | agent→daily;notify 的响应也走这里 |
| **digest** | **Reme background** | Agent 显式(后门) | resource + daily 双源合流到 digest |
| **maintain** | **Reme background**(cron + threshold) | Agent / 人工 显式(后门) | digest 内部折叠 |
| **retrieve** | Agent on-demand | — | 三层只读 |

**关键定性**:`notify` / `digest` / `maintain` 三个 reme 自治动作的主控制权**在 reme,不在 agent**。Agent 只负责"响应 notify + 写 daily(响应或自身任务驱动)+ 主动读";不需要记得"该看哪些 resource""该蒸了""该整理了"。

### 5.3 Service + Runtime 双进程结构

把 5.2 的归属直接推出 reme 的基本架构。两进程共享 L0-L4 全栈,只在 L5(进程入口)分叉:

```
┌──────────────────────────────────────────────────────────┐
│                      Reme System                         │
│                                                          │
│  ┌────────────────────┐      ┌────────────────────────┐ │
│  │  L5 Service        │      │  L5 Runtime            │ │
│  │  (HTTP / MCP)      │      │  (scheduler 自治)       │ │
│  │  服务 agent 请求:   │      │  服务 workspace 健康:        │ │
│  │  · ingest          │      │  · notify(决策)        │ │
│  │  · sync            │      │  · digest              │ │
│  │  · retrieve        │      │  · maintain            │ │
│  │  · notify(MCP 推送)│      │                        │ │
│  └─────────┬──────────┘      └───────────┬────────────┘ │
│            │                              │              │
│            └──────────────┬───────────────┘              │
│                           ▼                              │
│   ┌──────────────────────────────────────────────────┐   │
│   │  L4 Action / L3 原子工具                          │   │
│   │  Action 编排 → 基础工具 + 高级工具                  │   │
│   └──────────────────────┬───────────────────────────┘   │
│                          ▼                               │
│   ┌──────────────────────────────────────────────────┐   │
│   │  L2 文件状态(file_store + file_graph + 自治)       │   │
│   └──────────────────────▲───────────────────────────┘   │
│                          │ 派生                          │
│   ┌──────────────────────┴───────────────────────────┐   │
│   │  L1 file_watcher(fs → state 的唯一桥)             │   │
│   └──────────────────────▲───────────────────────────┘   │
│                          │ 监听                           │
│   ┌──────────────────────┴───────────────────────────┐   │
│   │  L0 workspace filesystem                             │   │
│   │  resource/    daily/    digest/                  │   │
│   └──────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────┘
```

两进程职责正交、共享 L0-L4 基础设施(详见 §6):

| 维度 | L5 Service | L5 Runtime |
|---|---|---|
| 触发方式 | 请求-响应 + MCP server-initiated 推送 | 周期 + L2 自治状态阈值 |
| 服务对象 | agent | workspace 自身 |
| 暴露给 agent | 是 | 否(agent 不感知) |
| 主要动作 | ingest / sync / retrieve / **notify 推送通道**(MCP) | **notify 决策** / digest / maintain |
| 与对端的耦合 | 通过 L2 推送队列读 notifier 产出 | 通过 L2 推送队列写,**不直接调 Service** |

### 5.4 节奏(latency tolerance)

| 动作 | 节奏 | latency 容忍 |
|---|---|---|
| retrieve | request-driven | sub-second |
| ingest | event-driven | seconds |
| sync | agent on-demand | seconds |
| **notify** | reactive(L2 资源状态变化后) | seconds ~ minutes |
| digest | reactive(状态变化后) | minutes ~ hours(eventual consistency) |
| maintain | periodic | days(无紧迫) |

实时性需求差三个数量级。这是 digest / maintain 必须放后台异步的根本原因 —— 不能阻塞 agent 的 retrieve / sync 请求。

### 5.5 排序约束与一致性模型

部分动作对**不能并发**,background runtime 内部要保证排序:

| 约束 | 原因 |
|---|---|
| **sync(同一 daily)→ digest(同一 daily)** | digest 不能看到 sync 半成品 |
| **digest(同一 scope)→ maintain(同一 scope)** | maintain 重组的拓扑不应被 digest 中途插入 |

ingest / retrieve 跟所有动作都可并发(纯入流 + 纯读)。

**一致性模型 = Eventual consistency on digest/maintain**。Agent 不能依赖"我刚写完 daily 就能查到对应 digest"。digest / maintain 都是后台异步,有可见的延迟窗口。

**watcher 滞后契约(L0 ↔ L2)**:基础工具直接写 L0 文件系统,L2 文件状态由 L1 file_watcher 派生,二者之间存在 eventual 窗口 —— 写完一份 daily 后,search / traverse 这类走 L2 索引的高级工具不一定立刻能看到。这是设计意图,不是 bug:agent 本身有上下文窗口,近期信息靠 agent 自带的对话上下文承接,不依赖 reme 索引立即可见。需要"写后立刻可读"的场景请用基础工具(read 直接读 fs)。

---

## 6. 执行栈:六层结构

L3 原子工具、L4 Action、L5 进程都不直接操作文件系统。它们坐在 L0-L2 的**底层基础**上 —— 这套基础是 reme 的"动力源",决定了为什么上层能解耦成 Service + Runtime 两进程,且两者既正交又共享状态。

### 6.1 概念分层(L0 → L5)

```
┌──────────────────────────────────────────────────────────┐
│  L5  Service  ‖  Runtime                                 │
│        进程入口:Service 服务 agent;Runtime 自治维护     │
├──────────────────────────────────────────────────────────┤
│  L4  Action(6 类动作语义)                              │
│        ingest / notify / sync / retrieve / digest / maintain │
├──────────────────────────────────────────────────────────┤
│  L3  原子工具                                            │
│        基础工具(直接 fs) + 高级工具(走 L2 索引)      │
├──────────────────────────────────────────────────────────┤
│  L2  文件状态                                            │
│        file_store + file_graph + 自治状态(scheduler 用) │
├──────────────────────────────────────────────────────────┤
│  L1  file_watcher                                        │
│        fs event → state delta(唯一 fs→state 桥)         │
├──────────────────────────────────────────────────────────┤
│  L0  workspace filesystem                                    │
│        resource/ + daily/ + digest/                      │
└──────────────────────────────────────────────────────────┘

数据流(主要关系):
  · L3 基础工具    ──写──►  L0
  · L3 基础工具    ──读──►  L0(无需经 L2)
  · L0 变化        ──►  L1 监听到  ──派生──►  L2 state delta
  · L3 高级工具    ──读──►  L2(走索引)
  · L4 Action      ──编排──►  L3 工具组合
  · L5 进程        ──触发──►  L4 Action
```

| 层 | 角色 | 关键约束 |
|---|---|---|
| L0 | workspace 文件系统 | 唯一真相源;任何 L2 状态都可由 reindex 从 L0 重建 |
| L1 | `file_watcher` | **唯一**与 fs 事件直接耦合的组件;fs→state 的唯一派生桥 |
| L2 | 文件状态(`file_store` + `file_graph` + 自治状态) | 高级工具的读视图;由 L1 单向更新,L3+ 只读不写 |
| L3 | 原子工具(基础 / 高级) | 基础直读写 L0;高级只走 L2 |
| L4 | Action(6 类语义动作) | N:M 编排 L3 工具;不直接碰 L0 / L2 |
| L5 | Service / Runtime 进程 | 共享 L0-L4 全栈,**不直接通信**,只通过 L0 / L2 状态间接耦合 |

### 6.2 L1 file_watcher:fs → state 的唯一桥

`file_watcher` 是 reme 唯一与 OS filesystem 事件直接耦合的组件。它把 fs 变化翻译为 L2 文件状态的 delta,承担**双重职责**:

```
filesystem events (create / modify / move / delete)
       │
       ▼
   file_watcher ─┬─► 索引同步:写完文件,L2 file_store/file_graph 自动更新
                 │     (L3 基础工具不需要显式调用"入索引")
                 │
                 └─► 自治状态派生:维护 scheduler 用的可推导状态
                       · resource: 入流批次 / 未消化(orphan) /
                                  推送状态(pending/notified/acknowledged)
                       · daily:   任务索引(进行中 / stale / 完成)
                       · digest:  密度水位 / 断链(broken wikilink)
```

**关键设计**:可派生的状态由 watcher 在外部索引中维护,**不写回 frontmatter**。L3 工具只管写内容文件,状态由 watcher 独立派生。这是 ✗-14 反例(L3 写 `status` 字段)成立的基础。

**Acknowledge 派生例**:`notify` 的"推送状态"由 watcher 维护 —— 当 watcher 检测到一条新 wikilink 从 daily 指向某 resource,即把该 resource 的推送状态从 `notified` 改为 `acknowledged`。notifier / Service 都不需要显式 ack。

### 6.3 L2 文件状态

| 组件 | 职责 | 由谁更新 |
|---|---|---|
| `file_store` | chunk 分块 + 向量持久化,提供 search / read API | L1 watcher 派生 |
| `file_graph` | wikilink 有向图,提供 upsert / traverse(双向)API | L1 watcher 派生 |
| **自治状态** | scheduler 自治决策的输入(入流批次 / 任务索引 / 密度水位 / 断链) | L1 watcher 派生 |
| **推送队列** | `notify` 的待推送 / 已推送 / 已确认条目 | notifier 写 pending;Service 推送后置 notified;L1 watcher 检测到 ack 后置 acknowledged |

L2 只关心"workspace 当前是什么样",**无业务语义** —— 不知道 daily / digest / 动作语义的存在。L3+ 只读 L2,不写(**例外**:notifier 写推送队列,这是 Runtime 与 Service 之间唯一的间接耦合通道,见 F-5)。

> **现状提示**:当前实现中 file_store / file_graph 之外的"自治状态"和"推送队列"尚不完整,这是 L1 watcher 与 notifier 待补齐的能力。完整化后 scheduler 才能从"周期扫描"切换为"事件驱动",`notify` 才能从隐式变为显式。

### 6.4 L3 原子工具:基础 vs 高级

两组工具的切分依据只有一条:**是否必须经过 L2 索引**。

| 组别 | 工具 | 数据通路 | 一致性 |
|---|---|---|---|
| **基础工具** | create / append / edit / read / write / move / delete / list / stat | 直接对接 L0 | 写后立即可读(同一工具) |
| **高级工具** | search / traverse / frontmatter | 必须走 L2 索引 | 受 watcher 滞后影响(eventual) |

**写路径全部走基础工具**(L4 Action 编排基础工具完成写入)。高级工具是**只读**的索引查询入口。

**eventual 窗口**:基础工具写 L0 后,L2 索引由 L1 watcher 异步追平。在窗口内,高级工具看到的是滞后的视图。详见 §5.5"watcher 滞后契约"。

### 6.5 设计含义

| # | 不变量 | 推论 |
|---|---|---|
| **F-1** | L1 `file_watcher` 是 L0 → L2 的**唯一**派生桥 | 状态一致性是 L1 的事;L3 工具不要"自己更新索引" |
| **F-2** | L3 基础工具直接读写 L0;L3 高级工具只走 L2 | 写路径无需"先 reindex";读路径接受 eventual |
| **F-3** | L0 是唯一真相源 | 任何 L2 状态都可由 reindex 从 L0 重建,L2 是缓存而非数据库 |
| **F-4** | L4 Action 与 L3 工具是 N:M 编排关系 | Action 不直接碰 L0 / L2 |
| **F-5** | L5 Service 与 Runtime 共享 L0-L4 全栈,**不直接通信** | 只通过 L0 / L2 状态间接耦合;一边崩了不影响另一边的读 |
| **F-6** | L0 与 L2 之间存在 eventual 窗口 | agent 上下文承担近期信息,不依赖 L2 立即可见(详见 §5.5) |

---

## 7. L4 Action 模块映射

第 5 节列了 6 类 Action 及其触发源,本节把这些 Action 落到 **L4 实现模块**(架构角色,不指代源码路径);触发机制(on-demand 路径与 background 路径)见 §5.3。

### 7.1 五个 L4 模块

| 模块 | 实现动作 | 触发 | LLM-driven | 单一职责 |
|---|---|---|---|---|
| **ingester** | ingest | External push / pull | × | 原样落 resource + 抽 frontmatter + 入索引 |
| **notifier** | notify | Reme background(cron + L2 资源自治状态阈值) | × | 从 L2 资源自治状态选候选 → 写 L2 推送队列;Service MCP 拿走 |
| **synchronizer** | sync | Agent on-demand | ✓ | 把当下事件织入 daily 工作叙事 |
| **dreamer** | digest | Reme background | ✓ | resource + daily 双源合流成 digest 长期条目 |
| **maintainer** | maintain | Reme background | ✓ | digest topic tree 的**密度折叠**(fold-only) |

`retrieve` 不构成独立 L4 模块,理由见 §7.4。

**三个 reme 自治模块**:notifier(机械)、dreamer(LLM)、maintainer(LLM)。三者都由 scheduler 触发,都消费 L2 自治状态,但只有 notifier 是机械的 —— 候选选择不需要 LLM,LLM 决策在 agent 侧的 sync。

### 7.2 对称结构

```
                     跨表征层翻译                  结构性纪律
                     (LLM-driven)                 (机械)

  Inbound:                                        ingester
  Attention:                                      notifier
  Working:           synchronizer
  Sink:              dreamer
  Organization:      maintainer (fold-only)
```

五类不同方向的"翻译":

| 模块 | 翻译方向 |
|---|---|
| ingester | 外部异构格式 → workspace 统一文件 |
| notifier | L2 资源自治状态 → agent 注意力(`notify` 推送) |
| synchronizer | agent 事件流 → 工作过程叙事(写 hot) |
| dreamer | 工作过程 + 原始资料 → 长期知识(双源合流,写 cold) |
| maintainer | 散乱叶子 → 有层次的 topic tree(组织 cold) |

ingester 和 notifier 是机械(确定性阈值/流水线);其它三个是 LLM 决策模块,各自跨越一层语义鸿沟。

### 7.3 Maintainer:Topic Tree 密度折叠

`digest/` 整体视为一颗 **topic tree**:文件夹 = 中间节点,文件 = 叶子。Maintainer 唯一职责:随写入持续,某中间节点下叶子过密时,**折叠**为新的子中间节点 + 高密度摘要。

```
触发前:某中间节点叶子过多 / 太碎
  digest/infra/
  ├── logging.md
  ├── tracing.md
  ├── metrics.md
  ├── alerting.md
  ├── dashboards.md
  └── slo.md

折叠后:LLM 判断聚类,引入子中间节点 + 摘要
  digest/infra/
  ├── observability/              ← 新中间节点
  │   ├── _index.md               ← 新生成的高密度摘要
  │   ├── logging.md              ← 内容不变,只搬位置
  │   ├── tracing.md
  │   ├── metrics.md
  │   ├── alerting.md
  │   ├── dashboards.md
  │   └── slo.md
  └── ...(未被折叠的叶子原位)
```

**设计承诺**(在 `maintain` 通用不变量之上进一步收紧):

| # | 承诺 | 含义 |
|---|---|---|
| **M-1** | **Fold-only**,无 merge / move / promote / demote / introduce | 树只向下生长,从不反向 |
| **M-2** | 叶子内容 0 修改,只被搬位置 | 与 `maintain` 内容守恒一致 |
| **M-3** | 新中间节点带一个高密度摘要文件,读摘要就能决定要不要深入 | 折叠后可读性不降反升 |
| **M-4** | 每次只处理一个候选节点 | 最小化变更面 |
| **M-5** | 不能聚类时,**不动**(默认保守) | 宁可不折,不要错折 |

LLM 唯一的决策点:

1. 这些叶子能不能聚类(if not → 不动)
2. 新中间节点叫什么、摘要怎么写

其它都机械:阈值判断(L1 file_watcher 派生 L2 自治状态提供信号)、移动文件(crud)、wikilink 重定向(graph/retarget)。

### 7.4 为什么没有 retriever 模块

L4 模块的存在条件 = "有跨原子编排 / 需要 LLM 决策"。Retrieve 不满足:

- 三种问法(state / semantic / topological)各自被 **L3 原子工具**直接覆盖(list+filter / search / traverse)
- 没有跨原子状态、没有 LLM 决策点
- Agent 直接调用 L3 原子即可

---

## 8. 跨切面:Schema

Schema(资料的 frontmatter / wikilink / 章节约定)是横跨三层、各写动作的共同契约。reme 的核心立场:

| 立场 | 说明 |
|---|---|
| **reme 核心只保留 `name` / `description` 两个字段** | 其它都是 opinionated convention,服务消费层可以替换 |
| **Schema 是"协议"不是"代码"** | 用 markdown 文字描述,LLM agent 自我约束;不内嵌 schema validator |
| **三层共用同一套 wikilink 协议** | 全路径引用,无 short-link / no-ext 解析 |

### 8.1 协议文档(opinionated default)

| 内容 | 谁规定 |
|---|---|
| 目录结构(三层 + folder 单位) | 第 2 节本文档 |
| 动作语义契约(notify / sync / digest / maintain 的输入产出不变量) | 第 3 节本文档 |
| Frontmatter 推荐字段(4 轴等) | `reme/steps/jobs/protocol.md` (opinionated) |
| 章节约定(Objective/Plan/Progress/...)| sync / digest 各自的 prompt(opinionated) |

### 8.2 重载入口

服务消费层(plugin / 自定义 caller)无需 fork reme,可通过以下方式替换 schema:

| 入口 | 适用场景 |
|---|---|
| 替换 protocol 文档 | 改 frontmatter / wikilink / 章节约定 |
| 替换 prompt 模板 | 改 sync / digest 的决策流程 |
| 替换 toolkit | 改 ReAct agent 可见的工具集 |

---

## 9. 反例:不属于本架构的设计

明确画出**不允许**的设计,免得后续讨论或扩展时滑回去:

| # | 反例 | 违反的不变量 |
|---|---|---|
| ✗-1 | Agent 通过任意 verb 直接写 digest | I-1(digest 写权只属 dreamer / maintainer) |
| ✗-2 | 多 agent 并发改同一个 daily folder | I-2(daily 单作者) |
| ✗-3 | 任何动作改写 resource 的原文 | I-3(resource immutable) |
| ✗-4 | 跨层引用引入第二套机制(hash-id / external ref / SQL) | I-4(wikilink 是唯一跨层载体) |
| ✗-5 | dreamer 改写 daily 正文 | `digest` 不变量(§3.5) |
| ✗-6 | maintain 改写 daily / resource 的语义内容 | `maintain` 不变量(§3.6) |
| ✗-7 | `notify` 维护 resource 上的 `referenced_by` 反指 | `notify` 完全单向(§3.3) |
| ✗-8 | 把 state / semantic / topological 合并成单一 read verb | R-1 |
| ✗-9 | Retrieve 自动 eager-expand provenance | R-4 |
| ✗-10 | digest / maintain 同步阻塞 agent 请求 | 5.4 节奏分级 |
| ✗-11 | digest / maintain 强一致(agent 写完 daily 立即可查 digest) | 5.5 eventual consistency |
| ✗-12 | maintainer 做 merge / move / promote / demote 等"通用重组" | M-1(fold-only) |
| ✗-13 | maintainer 改写既有叶子的内容(不只是搬位置) | M-2(叶子内容 0 修改) |
| ✗-14 | L4 模块在 frontmatter 里写 `status` / `pending` 等可派生状态字段 | 状态由 L1 file_watcher 派生到 L2 自治状态,L4 不重复 |
| ✗-15 | 为 retrieve 单设 L4 模块或聚合 verb | §7.4(L3 原子已足够) |
| ✗-16 | notify 写入 workspace(在 resource 上加 `notified` frontmatter 或新建 daily 占位) | notify 只写 L2 推送队列,**不落任何文件**;ack 由 L1 watcher 检测 wikilink 派生 |
| ✗-17 | Service 自决推什么 notify 候选 | notify 决策在 Runtime(notifier);Service 只是 MCP transport,从 L2 推送队列读取(F-5) |
| ✗-18 | agent 主动调用 `notify` 想"标记这个 resource 我要看" | notify 是 reme→agent 单向,反向是 agent 用 sync 写 wikilink(自然 ack) |

---

## 附录:术语索引

| 术语 | 定义 |
|---|---|
| **resource/** | 不可变原始资料层 |
| **daily/** | agent 任务工作区层 |
| **digest/** | 沉淀知识层 |
| **State 问** | 在某层做 list + 过滤的状态查询 |
| **Semantic 问** | 跨层全文/向量检索 |
| **Topological 问** | 沿 wikilink 走的拓扑查询 |
| **Provenance** | 下游节点反查到上游来源的能力 |
| **Retarget** | 节点移动时对所有入向 wikilink 的原子重写 |
| **L5 Service** | 服务 agent 请求的进程(HTTP / MCP);执行栈最上层;也是 notify 的 MCP transport |
| **L5 Runtime** | 自治维护 workspace 的进程;scheduler 在其中按 L2 自治状态阈值触发 background Action |
| **L4 Action** | 6 类动作语义:ingest / notify / sync / retrieve / digest / maintain |
| **L4 模块** | 实现 Action 的架构角色;五个:ingester / notifier / synchronizer / dreamer / maintainer(retrieve 不构成独立模块) |
| **ingester** | L4 模块,机械:外部源原样落 resource + 抽 frontmatter + 入索引 |
| **notifier** | L4 模块,机械:从 L2 资源自治状态选 notify 候选 → 写 L2 推送队列;Service MCP 拿走推给 agent |
| **synchronizer** | L4 模块,LLM-driven:agent 事件织入 daily 工作叙事;响应 notify 的也走这里 |
| **dreamer** | L4 模块,LLM-driven:resource + daily 双源合流为 digest 长期条目 |
| **maintainer** | L4 模块,LLM-driven,**fold-only**:digest topic tree 的密度折叠 |
| **scheduler** | L5 Runtime 内部触发器:按 cron + L2 自治状态阈值拉起 background L4 模块(notifier / dreamer / maintainer) |
| **Topic tree** | digest/ 的心智模型:文件夹 = 中间节点,文件 = 叶子 |
| **Fold(密度折叠)** | maintainer 唯一操作:把过密叶子归簇到新子中间节点 + 写高密度摘要 |
| **L3 原子工具** | 基础(create/append/edit/read/write/move/delete/list/stat,直 fs)+ 高级(search/traverse/frontmatter,走 L2)两组 |
| **L2 文件状态** | `file_store` + `file_graph` + 自治状态(resource 入流批次/orphan、daily 任务索引、digest 密度水位/断链)+ 推送队列;由 L1 派生(推送队列由 notifier 写) |
| **L2 推送队列** | `notify` 的 L2 状态条目;notifier 写 pending,Service MCP 推送后置 notified,L1 watcher 检测到 daily→resource wikilink 后置 acknowledged |
| **L1 file_watcher** | fs event → L2 state delta 的唯一派生桥;承担索引同步 + 自治状态派生 + `notify` ack 派生 |
| **L0 workspace filesystem** | 物理目录:resource/ + daily/ + digest/;唯一真相源 |
| **Eventual consistency** | digest / maintain 异步处理,有可见延迟窗口;L0↔L2 之间 watcher 滞后窗口同理 |
| **Opinionated default** | reme 提供的参考实现,服务层可替换 |
