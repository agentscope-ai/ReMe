# ReMe 应用场景

## 金融场景：产业链知识库

**主角**：王分析师，新能源行业研究员，每天处理 10+ 篇研报、数十条产业新闻、若干场公司调研。

**痛点**：信息散落在飞书文档、PDF 研报、微信群消息、调研纪要里，"上次调研宁德时代时聊到的钴价话题"再也找不回来。

### 一周内 ReMe 自动织出的产业链图谱

---

#### Day 1（周一）盘后：素材摄入 + 对话

王分析师把今天看到的 3 篇研报扔进 `resource/`，又和 Agent 口述了对刚果(金)矿权变更的看法：

```
对话片段：
> 今天嘉能可发了三季报，钴产量同比下滑 18%……
> 刚果(金)那边的政策变化，对洛阳钼业 KFM 矿的影响要重点跟……
> 下游三元正极厂商已经开始转向高镍低钴方案……
```

**auto-memory** 实时把对话写入当天日记；**auto-resource** 自动解析研报写入加工笔记：

```
daily/
├── 2026-05-18.md              ← 当天索引页，汇总所有事件
└── 2026-05-18/
    ├── session_001.md         ← auto-memory 写入的对话日志
    │                            （含嘉能可三季报、刚果金矿权、高镍化趋势等事件）
    ├── resource_001.md        ← auto-resource 对研报 1 的加工笔记
    ├── resource_002.md        ← 研报 2 加工笔记
    └── resource_003.md        ← 研报 3 加工笔记
```

**Day 1 夜间 auto-dream**——CronDreamer 扫描当天 4 个文件，逐个执行 Extract → Integrate 管线：

处理 `session_001.md`：
- **Phase 1 Extract**：从对话日志中提取 3 个抽象单元——「嘉能可钴产量下滑」(wiki)、「刚果金矿权政策风险」(wiki)、「三元正极高镍化趋势」(wiki)
- **Phase 2 Integrate**（每个 unit 独立一个 Agent）：
  - 搜索已有 digest，均无匹配 → 决策 **CREATE**
  - 新建 `digest/wiki/嘉能可.md`、`digest/wiki/钴.md`、`digest/wiki/三元正极.md`
  - 写入时自动织链接：`derived_from:: [[daily/2026-05-18/session_001]]`，以及概念互联 `relates_to:: [[三元正极]]`

处理 `resource_001.md`（嘉能可三季报）：
- **Phase 1**：提取「嘉能可钴业务财务数据」(wiki)
- **Phase 2**：搜索到刚刚新建的 `digest/wiki/嘉能可.md` → 决策 **CORROBORATE**，追加财务佐证段落

Day 1 结束时 `digest/wiki/` 下新增：

```
digest/wiki/
├── 嘉能可.md          ← CREATE + CORROBORATE（研报佐证）
├── 钴.md             ← CREATE
└── 三元正极.md        ← CREATE
```

`钴.md` 长这样：

```markdown
---
name: 钴
description: 锂电正极材料关键原料，主产区刚果(金)
tags: [新能源, 原料, 钴]
---

所属领域:: [[新能源]]
下游产品:: [[三元正极]]

# 钴

## 供给端
主要生产商 [[嘉能可]]，产能集中于刚果(金)。
嘉能可三季度钴产量同比下滑 18%。
derived_from:: [[daily/2026-05-18/session_001]]

## 政策风险
刚果(金)矿权政策变化，可能影响 KFM 矿运营。
derived_from:: [[daily/2026-05-18/session_001]]
```

---

#### Day 2（周二）：宁德时代调研

王分析师参加宁德时代调研会后，和 Agent 聊调研要点：

```
> 宁德今年全面切换 9 系高镍三元，钴用量还会继续降……
> 产能利用率 85%，比上季度高 5 个点……
```

auto-memory 写入 `daily/2026-05-19/session_001.md`。

**Day 2 夜间 auto-dream** 处理这份 session：
- **Phase 1**：提取「宁德时代高镍切换」(wiki)、「宁德时代产能利用率」(wiki)
- **Phase 2**：
  - 「宁德时代高镍切换」→ 搜索到 `digest/wiki/三元正极.md` 已存在高镍化内容 → 决策 **REFINE**，补充"宁德 9 系切换"作为具体案例，并新建 `digest/wiki/宁德时代.md`
  - 「产能利用率」→ 无匹配 → 写入 `digest/wiki/宁德时代.md`（已存在，追加章节）

Day 2 结束时图谱新增节点和边：

```
digest/wiki/
├── 嘉能可.md
├── 钴.md
├── 三元正极.md        ← REFINE：新增宁德 9 系切换案例
└── 宁德时代.md        ← CREATE：含高镍切换 + 产能数据
                          relates_to:: [[三元正极]]
                          relates_to:: [[钴]]
```

---

#### Day 3（周三）：亿纬电话会 + 洛阳钼业跟踪

两场对话产生两份 session。夜间 auto-dream 逐个处理：

- `session_001.md`（亿纬电话会）→ Phase 2 搜到 `宁德时代.md`、`三元正极.md` → CREATE `digest/wiki/亿纬锂能.md`，并在 `三元正极.md` 上 CORROBORATE 高镍趋势
- `session_002.md`（洛阳钼业跟踪）→ Phase 2 搜到 `钴.md` → REFINE `钴.md`，补充洛阳钼业 KFM 矿最新动态；CREATE `digest/wiki/洛阳钼业.md`

Day 3 结束时图谱：

```
digest/wiki/
├── 嘉能可.md
├── 洛阳钼业.md        ← CREATE
├── 钴.md             ← REFINE：补充洛阳钼业信息
│                        relates_to:: [[嘉能可]], [[洛阳钼业]], [[三元正极]]
├── 三元正极.md        ← CORROBORATE：亿纬佐证
│                        relates_to:: [[钴]], [[宁德时代]], [[亿纬锂能]]
├── 宁德时代.md
└── 亿纬锂能.md        ← CREATE
```

每个节点都是**当天 dream 从一份 daily 文件中提取并整合的结果**，不存在跨天"聚合"——跨文件的关联通过 Phase 2 的 search 自然发现已有 digest，从而把新信息写入正确的位置。

---

#### Day 5（周五）：用户主动检索

王分析师准备组会要讲新能源板块，主动问 Agent：

> **"帮我分析一下锂电相关上下游"**

Agent 调用 ReMe 的 `search` Job，走向量 + BM25 + RRF 融合检索，命中已有的 digest 节点，并通过渐进式展开获取上下游全貌：

**第一跳——直接命中 chunk 全文 + 评分**：

```
digest/wiki/钴.md:5-22 [score=0.0234 vector=0.0156 keyword=0.0078]
# 钴 / ## 供给端
主要生产商嘉能可、洛阳钼业，产能集中于刚果(金)……

digest/wiki/三元正极.md:10-30 [score=0.0211 vector=0.0148 keyword=0.0063]
# 三元正极 / ## 高镍低钴路线
2026 年起主流厂商加速 9 系产品，宁德已全面切换……

digest/wiki/宁德时代.md:1-20 [score=0.0193 vector=0.0135 keyword=0.0058]
digest/wiki/嘉能可.md:5-30   [score=0.0167 vector=0.0117 keyword=0.0050]
digest/wiki/洛阳钼业.md:1-22 [score=0.0152 vector=0.0106 keyword=0.0046]
```

**第二跳——展开 wikilink 邻居目录（只有标题，不展开正文）**：

```
digest/wiki/钴.md 的邻居：
  outlinks (3):
    → digest/wiki/三元正极.md  name="三元正极" description="高镍低钴技术路线"  via predicate=下游产品
    → digest/wiki/嘉能可.md    name="嘉能可"   description="全球钴业巨头"      via predicate=relates_to
    → digest/wiki/洛阳钼业.md  name="洛阳钼业" description="KFM 矿运营商"      via predicate=relates_to
  inlinks (2):
    ← daily/2026-05-18/session_001.md  name="盘后对话"  via plain
    ← daily/2026-05-19/session_001.md  name="宁德调研"  via plain

digest/wiki/三元正极.md 的邻居：
  outlinks (2):
    → digest/wiki/宁德时代.md  name="宁德时代" description="全球动力电池龙头"  via predicate=relates_to
    → digest/wiki/钴.md        name="钴"       description="锂电正极关键原料"  via predicate=relates_to
  inlinks (1):
    ← digest/wiki/亿纬锂能.md  name="亿纬锂能" description="动力电池厂商"      via predicate=relates_to
```

**第 N 跳——Agent 按需深入**：Agent 看过目录后，决定展开 `亿纬锂能.md` 的正文获取补充信息，调用 `read` Job 拉取。

Agent 基于检索结果**直接回复**王分析师：

> "锂电产业链分三段：上游钴矿（嘉能可、洛阳钼业，刚果金集中）、中游三元正极（高镍化加速）、下游电池厂（宁德/亿纬）。这周你提到的事件分别落在：嘉能可三季报 → 上游产能收缩；高镍化趋势 → 中游路线切换；宁德 9 系切换 → 下游需求验证。"

---

#### Day 7：图谱已经长出层次

经过一周每天的 auto-dream 逐文件处理，图谱自然生长出来：

```
                        ┌─────────────┐
              ┌────────►│   锂电      │◄────────┐
              │         └──────┬──────┘          │
              │  upstream      │                 │ upstream
              │                │ 下游产品         │
      ┌───────┴──────┐        ▼          ┌──────┴──────┐
      │   钴         │   ┌─────────┐     │   锂        │
      │ (刚果金产区) │◄──┤  原料   ├────►│ (盐湖产区)  │
      └───────┬──────┘   └────┬────┘     └─────────────┘
              │ relates_to    │ relates_to
              ▼               ▼
      ┌──────────────┐   ┌──────────────┐
      │   嘉能可     │   │  三元正极    │◄── 高镍化趋势
      │   洛阳钼业   │   └──────┬───────┘
      └──────────────┘          │ relates_to
                                ▼
                        ┌──────────────┐
                        │   宁德时代   │  ← Day 2 调研
                        │   亿纬锂能   │  ← Day 3 电话会
                        └──────────────┘
```

**没有一个节点是凭空编造的**——每条边对应笔记里的一句 `relates_to:: [[X]]` 或 `derived_from:: [[daily/...]]`，每个节点点开就是 Markdown，每段内容都能追溯到原始 daily 事件。图谱不是一次性生成的，而是每天 dream 一点、链接一点，渐进生长出来的。

---

### ReMe 在这个场景下的核心价值

分析师只负责"看 + 说"，知识图谱自己长出来：

- **auto-memory** 把每次对话实时写入当天日记
- **auto-dream** 每天逐文件执行 Extract → Integrate 管线，先搜已有 digest 再决策（CREATE / CORROBORATE / REFINE / CORRECT），知识卡片渐进生长
- **auto-link** 是 dream 写入的副产品——`derived_from::` 溯源 + `relates_to::` 概念互联，图谱随每次 dream 自动变密
- **混合检索 + 渐进展开** 让 Agent 先看骨架再决定深入哪个节点，不把 Top-K 全文塞进上下文
