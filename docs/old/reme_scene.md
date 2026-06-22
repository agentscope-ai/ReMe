# ReMe 应用场景

本文描述 ReMe 在真实 Agent 工作流里的使用方式。目录、Job 名称和能力边界按 `reme/` 最新代码整理。

ReMe 的共同模式是：

```text
对话 / 外部材料
      |
      +--> auto_memory / auto_resource
      |        写入 daily/
      |
      +--> auto_dream
      |        从 daily/ 提炼 digest/{personal,procedure,wiki}/
      |        同时写 daily/<date>/interests.yaml
      |
      +--> search / node_search / read / traverse / proactive
               供 Agent 检索、联想、读取兴趣主题
```

## 场景一：金融分析师的产业链知识库

**主角**：王分析师，新能源行业研究员。每天处理研报、产业新闻、公司调研和盘后口述。

**痛点**：信息散在文本研报、网页摘录、群消息、调研纪要和对话里。几天后再问“上次宁德调研里提到的钴价影响”，很难把原始事件、公司、材料路线和上游矿企串起来。

### Day 1：盘后对话和研报进入 Daily

王分析师把 3 篇研报同步到 `resource/2026-05-18/`，又和 Agent 口述：

```text
今天嘉能可发了三季报，钴产量同比下滑 18%。
刚果(金)矿权政策变化，对洛阳钼业 KFM 矿的影响要重点跟。
下游三元正极厂商继续转向高镍低钴方案。
```

ReMe 产生两类浅加工文件：

```text
resource/
└── 2026-05-18/
    ├── glencore-q3.md
    ├── cobalt-policy.md
    └── cathode-trend.md

reme_session/
└── dialog/
    └── 2026-05-18-close.jsonl

daily/
├── 2026-05-18.md
└── 2026-05-18/
    ├── 2026-05-18-close.md
    ├── glencore-q3.md
    ├── cobalt-policy.md
    ├── cathode-trend.md
    └── interests.yaml          # auto_dream 后生成
```

对应链路：

- `auto_memory` 保存原始对话到 `reme_session/dialog/<session_id>.jsonl`，再让 Agent 把重要事实写入 `daily/<date>/<session_id>.md`。
- `resource_watch_loop` 监听 `resource/` 文本文件变化，并触发 `auto_resource_step` 写同名 daily note。
- `daily_create` 会维护 `daily/<date>.md` 当天索引页。

### Day 1 晚上：Auto Dream 进入 Digest

运行：

```bash
reme auto_dream date=2026-05-18
```

`auto_dream` 是四步管线：

```text
dream_extract_step
  扫描 daily/2026-05-18.md 和 daily/2026-05-18/ 下 changed 文件
  输出 units 和 topics

dream_integrate_step
  每个 unit 用 node_search 召回已有 digest 节点
  决定 CREATE / CORROBORATE / REFINE / CORRECT

dream_topics_step
  写 daily/2026-05-18/interests.yaml

dream_finish_step
  checkpoint 成功处理的 daily 输入
```

本场景中的产物：

```text
digest/
└── wiki/
    ├── 嘉能可.md
    ├── 钴.md
    └── 三元正极.md
```

示例 `digest/wiki/钴.md`：

```markdown
---
name: 钴
description: 锂电正极材料关键原料，主产区集中于刚果(金)
---

downstream_product:: [[digest/wiki/三元正极.md]]
producer:: [[digest/wiki/嘉能可.md]]
source_event:: [[daily/2026-05-18/2026-05-18-close.md]]

# 钴

## 供给端
嘉能可三季度钴产量同比下滑 18%，需要继续跟踪供给收缩对价格的影响。

## 政策风险
刚果(金)矿权政策变化可能影响 KFM 矿运营，需联动跟踪洛阳钼业。
```

注意：wikilink 是字面路径语义，推荐写完整 workspace-relative 路径和 `.md` 扩展名。ReMe 不会自动把 `[[钴]]` 解析成某个文件。

### Day 2：调研信息补充已有节点

王分析师参加宁德时代调研：

```text
宁德今年全面切换 9 系高镍三元，钴用量还会继续降。
产能利用率 85%，比上季度高 5 个点。
```

`auto_memory` 写入：

```text
daily/2026-05-19/ningde-research.md
```

`auto_dream date=2026-05-19` 时：

- `dream_extract_step` 提取“宁德高镍三元切换”“宁德产能利用率”。
- `dream_integrate_step` 用 `node_search` 在 `digest/` 内召回 `digest/wiki/三元正极.md` 和 `digest/wiki/钴.md`。
- Agent 对 `三元正极.md` 做 `REFINE`，把宁德 9 系切换作为案例写入。
- Agent `CREATE` 或更新 `digest/wiki/宁德时代.md`。

此时图谱逐步长成：

```text
digest/wiki/
├── 嘉能可.md
├── 钴.md
├── 三元正极.md        # REFINE: 高镍低钴趋势 + 宁德案例
└── 宁德时代.md        # CREATE: 产能利用率 + 9 系切换
```

### Day 5：用户检索“锂电上下游”

王分析师问：

```text
帮我分析一下锂电相关上下游。
```

Agent 调用：

```bash
reme search query="锂电 上下游 三元 正极 钴 宁德" limit=5
```

`search` 返回 chunk 正文、行号、分数，以及命中文件的 outlinks/inlinks 目录。默认配置下结果来自 BM25 + 图展开。

检索结果形态：

```text
========== digest/wiki/钴.md:8-20 [score=0.0148 keyword=3.7112] ==========
# 钴
## 供给端
嘉能可三季度钴产量同比下滑 18%...

  outlinks:
    -> digest/wiki/三元正极.md name="三元正极" via predicate=downstream_product
    -> digest/wiki/嘉能可.md name="嘉能可" via predicate=producer
  inlinks:
    <- digest/wiki/三元正极.md name="三元正极" via predicate=upstream_material

========== digest/wiki/三元正极.md:5-18 [score=0.0139 keyword=3.2017] ==========
...
```

Agent 可以只看邻居目录就拼出产业链骨架；需要细节时，再调用：

```bash
reme read path=digest/wiki/宁德时代.md
reme traverse path=digest/wiki/钴.md depth=2 direction=both
```

最终答复：

```text
锂电链条可以分三段：
1. 上游原料：钴，供给集中于刚果(金)，嘉能可是核心生产商，洛阳钼业 KFM 矿需要跟踪政策影响。
2. 中游材料：三元正极，高镍低钴路线持续推进。
3. 下游电池：宁德时代已切换 9 系高镍三元，验证下游需求方向。

这条结论分别来自 2026-05-18 的盘后对话、嘉能可三季报资源笔记和 2026-05-19 宁德调研记录。
```

### Proactive：读取当天兴趣主题

`auto_dream` 会写：

```text
daily/2026-05-18/interests.yaml
```

示例：

```yaml
date: 2026-05-18
topic_count: 3
diversity_days: 7
topics:
  - title: 刚果(金)矿权政策对钴供给的影响
    reason: 用户当天多次提到 KFM 矿和钴价风险
    keywords: [钴, 刚果金, 洛阳钼业, KFM]
    paths:
      - daily/2026-05-18/2026-05-18-close.md
```

调用：

```bash
reme proactive date=2026-05-18
```

`proactive` Job 返回 `interests.yaml` 中的 topics 和可选 YAML 原文。

### 场景价值

- 分析师只负责看资料和表达判断，ReMe 把事实落到 daily，把长期概念沉淀到 digest。
- `node_search` 让 dream 先找已有 digest 再写，避免同一概念每天新建一个文件。
- `search` 的图展开让 Agent 先看结构再读正文，减少上下文浪费。
- 所有结论都落在 Markdown 中，可用普通编辑器审计。

## 场景二：研发 Agent 的跨会话程序化记忆

**主角**：张研发，长期在 Claude Code、AgentScope 或其他 Agent 中处理项目问题。

**痛点**：同类 bug 多次出现，Agent 每次都从零开始排查；用户的代码风格、测试习惯、项目偏好只存在于当次对话里。

### 第一次会话：构建卡死

用户说：

```text
pnpm build 卡在 92%，CPU 不高，内存涨得很快。
```

Agent 排查过程：

```text
1. 清缓存，无效。
2. 升级 terser 插件，无效。
3. 发现 fork-ts-checker 内存不足。
4. 设置 NODE_OPTIONS=--max-old-space-size=8192 后通过。
```

`auto_memory` 写入：

```text
reme_session/dialog/build-oom-2026-03-10.jsonl
daily/2026-03-10/build-oom-2026-03-10.md
```

`auto_dream` 后生成：

```text
digest/
├── procedure/
│   └── typescript-build-oom.md
└── personal/
    └── code-style.md
```

示例 `digest/procedure/typescript-build-oom.md`：

```markdown
---
name: TypeScript 项目构建 OOM 排查路径
description: build 卡住且内存上涨时，优先检查类型检查进程内存
---

source_event:: [[daily/2026-03-10/build-oom-2026-03-10.md]]
related_preference:: [[digest/personal/code-style.md]]

# TypeScript 项目构建 OOM 排查路径

## 症状
构建卡在后段，CPU 不高但内存持续上涨。

## 优先路径
1. 检查 fork-ts-checker 或类型检查子进程是否 OOM。
2. 先尝试 `NODE_OPTIONS=--max-old-space-size=8192`。
3. 清缓存和升级压缩插件只有在有明确证据时再做。

## 已知无效路径
- 单纯删除 `.cache` 未解决 2026-03-10 的问题。
- 升级 terser 插件未解决 2026-03-10 的问题。
```

示例 `digest/personal/code-style.md`：

```markdown
---
name: 用户代码风格偏好
description: 用户在开发任务中反复表达的工程偏好
---

# 用户代码风格偏好

## 注释
用户不喜欢解释代码字面含义的注释，只接受解释 WHY 或复杂约束的注释。

## 测试
用户偏好针对风险点写聚焦测试，不喜欢大范围无关重构。
```

### 第二次会话：相似问题快速召回

六周后用户问：

```text
vite build 也卡在打包阶段，是同一类问题吗？
```

Agent 先调用：

```bash
reme search query="vite build 卡住 内存 上涨 TypeScript OOM" limit=5
```

命中：

```text
digest/procedure/typescript-build-oom.md
daily/2026-03-10/build-oom-2026-03-10.md
```

Agent 回复可以直接跳过低价值路径：

```text
上次类似问题是 TypeScript 类型检查进程 OOM。建议优先检查构建阶段的内存和类型检查子进程，
先试 `NODE_OPTIONS=--max-old-space-size=8192`。上次清缓存和升级压缩插件都没有解决。
```

### 场景价值

- `digest/procedure/` 保存“怎么做”和“哪些路径无效”，让 Agent 复用排查经验。
- `digest/personal/` 保存用户偏好，让 Agent 跨会话遵守同一工程风格。
- 原始对话仍在 `reme_session/dialog/`，daily 记录可追溯，digest 只是长期提炼结果。

## 场景三：个人第二大脑

**主角**：李工。日常和 Agent 聊工作、读书、家庭安排、跑步训练和旅行计划。

**痛点**：普通聊天记录按时间堆叠，三个月后只能全文搜索，很难回答“上次 Alice 推荐的那本书是什么”“我为什么改了训练计划”这类联想式问题。

### 日常输入

李工的一天产生：

```text
daily/2026-04-20/
├── lunch-with-alice.md
├── running-plan.md
└── frontend-design-review.md
```

`auto_dream` 抽取到：

```text
digest/
├── personal/
│   ├── alice.md
│   └── exercise-preferences.md
├── procedure/
│   └── frontend-review-checklist.md
└── wiki/
    └── deep-work.md
```

示例：

```markdown
---
name: Alice
description: 用户朋友，常推荐阅读材料
---

recommended_book:: [[digest/wiki/deep-work.md]]
source_event:: [[daily/2026-04-20/lunch-with-alice.md]]

# Alice

## 阅读推荐
2026-04-20 午餐时推荐过一本关于注意力和深度工作的书。
```

### 一次联想式回忆

用户问：

```text
上次 Alice 推荐的那本讲注意力的书叫什么？
```

Agent 可以先搜：

```bash
reme search query="Alice 推荐 注意力 书 深度" limit=5
```

命中：

```text
digest/personal/alice.md
  outlinks:
    -> digest/wiki/deep-work.md via predicate=recommended_book
daily/2026-04-20/lunch-with-alice.md
```

再读：

```bash
reme read path=digest/wiki/deep-work.md
```

最终答复：

```text
是《深度工作》。记录显示 Alice 在 2026-04-20 午餐时推荐过，
你后来把它归到注意力和工作方法主题下。
```

### 场景价值

- daily 保留“当时发生了什么”。
- digest/personal 记录人、偏好、长期关系。
- digest/wiki 记录书、概念、主题。
- wikilink 把“人 -> 书 -> 主题 -> 原始事件”串起来，比单纯按时间翻聊天记录更接近人的回忆方式。
