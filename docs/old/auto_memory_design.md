# auto-memory 设计(实时事件拆分 / 写入 daily)

> 本文档记录 reme 中 **auto-memory** 的设计讨论 —— 把 agent 连续的对话 / 任务流切成离散的 daily 事件原子,inline 落到 `daily/` 层。
>
> 配套阅读:
> - `structure.md` §2.1-2.2(daily 层定位)/ §3.4(sync 动作语义)/ §7.1(synchronizer 模块)
> - `auto_dream_design.md`:auto-memory 产物如何被 dream 消化(dream 读 daily 作为入流之一)
> - `auto_consolidate_design.md`:digest 的组织端 / CAS 写入协议;auto-memory 不直接复用,但事件级"拆"与节点级 split 在概念上同构(都把过载粒度切小)
> - `auto_cognition_design.md`:auto-cognition 三阶段顶层思想(写入 / 巩固 / 检索);daily 节点是 cognition 图视图的一部分(承载 `derived_from::` 反指),但不参与 Stage 2 巩固改造
>
> **服务全景**:reme 服务两条主线 —— **auto-memory**(本文档,入流端 / daily 写入)与 **auto-cognition**(顶层思想:写入 = auto-dream,巩固 = auto-consolidate,检索 = auto-recall)。auto-memory 把 agent 实时事件流切成 daily 事件原子;它的产物是 dream(cognition Stage 1)消化的两路输入之一(另一路是 resource)。
>
> **核心立场**:auto-memory 是 `structure.md` §3.4 `sync` 动作的实现侧 —— 强调 **inline 实时**与**事件边界检测**。是不是改名 sync → auto-memory 留给上层文档对齐,本文档聚焦机制。

---

## 0. 问题陈述

agent 的对话与任务过程是连续事件流(用户回合、工具调用、上下文切换、中断恢复),但记忆系统需要离散的、可独立检索的事件单元。auto-memory 解决这个切分问题。

| 输入 | 输出 |
|---|---|
| agent 当前事件流(对话回合 / 工具调用 / 任务切换信号);可选 `notify` 候选作为 cue | `daily/<date>/<event-slug>/<note>.md` 事件原子;`daily/<date>.md` 主索引 |

**设计目标**:
1. **事件边界尽量与 agent 语义意图一致** —— 同一个意图(同一个任务 / 同一段思路)→ 同一个事件;意图切换 → 新事件
2. **inline 实时写入** —— 不滞后,不批处理;agent 一边工作,记忆一边落地
3. **保持 daily 写权契约** —— I-2 单作者(同 folder 不并发改);folder 名 = summary note 名(I-3 可移动单元)

**显式排除**(不属于 auto-memory 职责):
- ❌ 蒸馏 / 沉淀:那是 auto-dream(`auto_dream_design.md`)的事
- ❌ 实体识别 / wikilink 自动补全:cognition 三阶段不在写入后做"事后补 wikilink"(详 `auto_cognition_design.md` §1.1);所有 wikilink 由 dream 在写入瞬间产出
- ❌ 改写 resource / digest:auto-memory 只写 daily(I-1 / I-3)

---

## 1. 已对齐决策

### 1.1 物理布局:与 `structure.md` §2.2 对齐

| 项 | 决策 |
|---|---|
| **主轴** | 时间 + 任务:`daily/<date>/<event-slug>/` |
| **event-slug** | LLM 抽取的事件短名(snake_case / dash-case;不强制 schema),同 `<date>` 下唯一 |
| **folder 内** | 一个事件可有 N 个 note(`progress.md` / `decision.md` / `references.md` 等),由消费层 schema 决定;最少含一个 summary note,与 folder 同名 |
| **主索引** | `daily/<date>.md`:当天事件列表(机械写入,wikilink 指向各 event folder)|
| **跨日索引** | 不强制;dream 消费时按 `<date>` 范围拉取即可 |

**为什么不是单文件 event**(report §5.1 一种简化方向):
- 单文件 event = `daily/<date>/<event-slug>.md` 比 folder 模型简单,但失去"一个事件可包含多个视角 note"的灵活度
- 现行 `structure.md` 已定 folder 单位模型;auto-memory 沿用,不破坏既有 I-2 / I-3
- 若后续 dogfooding 验证单事件普遍只有一份 note,可演进为 folder 内只放一份 summary,机械上等价于单文件方案 —— 演进路径平滑,不需要现在选

### 1.2 事件边界:语义意图切换驱动

事件边界由 LLM 在 inline 写入时判:**当前回合的意图是否仍属上一个 event**。

| 维度 | 决策 |
|---|---|
| **决策时机** | 每个 agent 回合写入前 inline 判 |
| **决策依据** | 上一个 active event 的 summary + 当前回合内容;LLM 输出 `{continue: bool, new_event_slug?: str, summary_patch?: str}` |
| **continue=true** | append 当前回合到 active event(append-only 或 LLM 重写 summary,详 §1.3) |
| **continue=false** | 关闭 active event(写最终 summary)+ 开新 event folder(slug 由 LLM 给)|
| **同时 active 多事件** | 不允许(I-2 单作者)—— 一时刻只一个 active event;真要并行任务,agent 自己 sync 切换 |

**已排除**:
- 时间窗口切分(N 分钟无活动则切)—— 对话节奏因任务而异,时间窗口噪声大
- 关键词切分(出现"切换 / 现在做 X"等触发词)—— 假阳性高,且不所有切换都明显说出
- 后置 batch 切分 —— inline 写入要求 event 必须当下可决定归属,不能等

### 1.3 事件内写入模型

active event 内,每个回合的内容写到 event folder 下,有两种模式可选(消费层 schema 决定):

| 模式 | 形态 | 适用 |
|---|---|---|
| **append-only** | 一份 `<event-slug>.md`,新回合 append 到末尾(章节 / 时间戳 / 等)| 实现最简;事件短(< 几十回合)时可读性 OK |
| **多 note 重写** | summary note(folder 同名)+ 各视角 note(`progress.md` / `decision.md`);LLM 把新内容融到对应 note,summary note 重写为当下概览 | 事件长 / 多视角时可读性高;LLM 成本高 |

**默认 opinionated default**:append-only(最简启动)。消费层可改 prompt + schema 走多 note。

**与 E-1 守恒的关系**:daily 不强制 E-1 守恒(它是工作记录,允许 LLM 删旧加新);只在 multi-note 重写模式下,可选启用类似守恒(保留所有 wikilink),具体由消费层决定。

### 1.4 主索引 `daily/<date>.md`

当天事件 list 视图,机械维护(无需 LLM):

| 触发 | 操作 |
|---|---|
| 新建 event folder | 主索引 append 一行 `[[daily/<date>/<event-slug>/<event-slug>.md|<event-slug>]]` |
| 事件关闭(被切下一个 event) | 主索引该行 append 最终 summary 摘要(可选,LLM 写最终 summary 时附带写入) |
| 索引文件不存在 | 写入第一个 event 时创建 |

主索引**仅承担当天浏览锚点**:文件系统 `ls daily/<date>/` 也能看见,但有主索引人/agent 可直接 `read daily/2026/05/28.md` 拿到 list 视图 + summary 一览。

不维护跨日索引(`daily/2026/05.md` 或 `daily.md`):dream 消费时按时间范围拉取即可;`list daily/<date>/` 已经覆盖浏览需求。

### 1.5 与 notify 的协作

`notify` 是 reme → agent 的虚边推送(`structure.md` §3.3),把"有新 resource 值得看"传递给 agent。auto-memory 在以下两点与 notify 协作:

| 维度 | 协作方式 |
|---|---|
| **新事件 cue** | agent 收到 notify 后,如果决定响应(开始处理这个候选),通常会触发**新 event** —— auto-memory 把 notify payload 作为 hint(候选 resource 路径)写入新 event 的 summary,顺手用 wikilink 引上 |
| **acknowledge 派生** | event note 里出现指向 `[[resource/...]]` 的 wikilink → L1 watcher 将该 resource 推送状态置 `acknowledged`(`structure.md` §3.3 / §6.2);auto-memory 自身不调任何 ack API |

**关键约束**:auto-memory **不强制** agent 用 wikilink 引 notify 候选 —— agent 可能略过、也可能不通过 wikilink 而是直接读 resource。ack 是 daily → resource wikilink 的副产品,不是 auto-memory 显式负责的事。

---

## 2. 待对齐边界点

### 2.1 LLM 决策频率与成本

inline 边界检测的最朴素形态是每回合调一次 LLM。在长对话 + 高频回合下成本可观。可选优化:
- **continue 假设默认**:大多数回合是 continue(同一意图内),LLM 可能只在"看似切换"启发(token 跨度大 / 工具种类突变 / 用户显式说"接下来")时跑;否则默认 continue 不调 LLM
- **批回合**:每 N 回合批一次,延迟切分(代价:active event 边界滞后,首版可接受)

首版默认每回合调一次(最简,正确率高),M1+ 视成本优化。

### 2.2 中断恢复 / 跨进程 active event

agent 进程重启 / Service 重启后,如何识别"还有 active event"?

候选方案:
- **L2 自治状态**:L1 watcher 派生 `daily/<date>/<event-slug>/` 中最新 mtime 的 event 为 active(默认 N 分钟内有写入)
- **状态文件**:`.daily-active` 维护 active event slug,Service 启动时读
- **每次重建**:agent 进程重启视为新 event,旧的关闭(切到 §1.2 continue=false 路径)—— 最简但会增加事件数

倾向 §1.2 自然路径(进程重启 = LLM 下次判 continue=false 概率高)+ 不维护状态文件,详细 worker recovery 留给 Service 实现。

### 2.3 多 agent 同 workspace 的 active event 隔离

I-2 daily 单作者契约在多 agent 场景下需细化。候选:
- per-agent date subfolder:`daily/<date>/<agent-id>/<event-slug>/`
- 单 agent 模式 + agent ID 进 event-slug:`<date>/<agent-id>_<event-slug>/`

第二种破坏 slug 短名习惯;第一种引入额外层级。倾向后者作为消费层契约,reme 核心不固化。

### 2.4 事件粒度的 prompt 引导

边界检测的 prompt 决定切分粒度。粗 = event 大 / dream 看每个 event 时容易 overflow;细 = event 数爆炸 / 主索引拥挤。

**opinionated default prompt 倾向**:
- 一个意图 = 一个事件(用户提了 X 问题 / agent 开了 Y 任务 → 直到这个意图收尾)
- 跨意图的"附带工作"(查资料 / 算个数)归入当前意图,不开新 event
- 真新意图("好,现在我们做下一件事")才切

详细 prompt 落 `reme/steps/jobs/protocol.md` 或 synchronizer 的 prompt 模板。

### 2.5 与 resource ingest 的时序

如果 ingest 与 auto-memory 同时活跃(External push 推 resource 进来 + agent 在 sync),且 agent 想响应这个新 resource:
- ingest 写完 resource → L1 watcher 派生 L2 → notifier 决策推送(`structure.md` §5.3)→ Service MCP 推给 agent
- agent 在当前回合或下一回合响应 → auto-memory 判 continue=false 开新 event,wikilink 引上 resource

整条链 sub-second 到 seconds(notify 节奏);auto-memory 不直接知道 ingest,只在 agent 决定响应时被动接收 notify payload。

### 2.6 跨日任务延续

event 物理路径含日期(`daily/<date>/<event-slug>/`),同一意图跨日的任务无法用同一 event folder 承载。候选模型:

| 模式 | 形态 | 适用 |
|---|---|---|
| **每日新 event,wikilink 反指前日** | new day 起新 folder;summary note frontmatter 加 `inherits: [[daily/<prev-date>/<prev-slug>/<prev-slug>.md]]`;新 event body 不复制旧内容,仅引用 | event-slug 短,日切口干净;查 backlinks 拼出整条任务链 |
| **同 event 重复写不同日** | 不允许(I-2 single author + event folder date 在路径上,跨日写违反路径不可变) | × |
| **任务 ID 跨 daily 抽象** | 引入 `task-id` 维度,daily event 只是某 task 的某一日切片;额外维护 task index | 复杂度高,M0 不引入 |

**倾向**:第一种(`inherits` frontmatter wikilink)—— 与 §1.4 主索引一致(机械维护),实现侧 LLM 在 §1.2 boundary 判定时若发现意图与最近 N 天某个 active 任务一致,直接写入 inherits 即可。详细 boundary prompt 落 §2.4。

INHERIT 行为细节(扫描窗口、predecessor 是否关闭、Plan/Objective 是否拷贝)归消费层 schema 决定;reme 核心只承认 `inherits:` frontmatter wikilink 作为跨日链路载体。

---

## 3. 与其它层的协作

| 上下游 | 关系 |
|---|---|
| ← **notify** | 接收 notify payload 作为新 event cue;不强制响应,不强制 wikilink 引 |
| ← **resource** | 只读(通过 wikilink 引);不写 |
| → **daily** | **唯一写者**(I-2);写 event folder + 主索引 |
| → **auto-dream** | dream 读 daily 作为入流(`auto_dream_design.md` §4.2 dream scope);auto-memory 写完即对 dream 可见(走 L2 索引,有 eventual 窗口) |
| → **auto-cognition (三阶段)** | daily 节点是 cognition 图视图的一部分;dream(Stage 1)读 daily 作为入流;consolidate(Stage 2)只对 digest 节点跑 dups / community / decay,**不改 daily**;recall(Stage 3)三层并行召回时 daily 也参与命中 |

**关键边界**:auto-memory 是 daily 写入端的**唯一**入口;cognition 三阶段没有任何子阶段会**事后改写 daily**(无写回路径)。daily 一旦由 auto-memory 写完,就只被读不被改(I-2 / I-3 仍守);后续 dream / consolidate / recall 都是只读消费。

---

## 4. 下一步

1. **synchronizer step 实现**:event 边界检测 prompt + active event 状态管理 + inline 写入(append-only 默认)
2. **主索引维护**:`daily/<date>.md` 机械维护(新 event 时 append、关闭时附 summary)—— 走 crud/daily 基础工具
3. **notify ack 派生验证**:L1 watcher 派生 acknowledged 状态(`structure.md` §6.2),与 auto-memory 的 wikilink 写入端到端跑通
4. **多 agent 隔离 schema**(M1+):若实际有并发 agent,确定 daily 子目录 / slug 命名约定
5. **粗 / 细粒度 prompt 调参**:dogfooding 后看实际 event 数 / dream 消化效率,调 boundary prompt

实现进入 `reme/steps/jobs/` 与 `reme/file_graph/` 时,本文档与 `auto_dream_design.md` / `auto_cognition_design.md` 共同作为契约依据。
