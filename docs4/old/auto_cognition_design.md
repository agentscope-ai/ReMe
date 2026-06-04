# auto-cognition 设计(顶层:心智循环)

> 本文档:reme4 中**长期记忆系统**的顶层认知模型 —— 把 agent 的记忆生命周期类比人类睡眠/觉醒回路,推导出**三阶段分工**与**15 维能力清单**。
>
> **三阶段实现各有专属文档**:
> - Stage 1 写入(REM 重放抽象) → `auto_dream_design.md`
> - Stage 2 巩固(NREM 深度整合) → `auto_consolidate_design.md`
> - Stage 3 检索(觉醒态提取) → `auto_recall_design.md`
>
> 配套阅读:
> - `auto_memory_design.md`:入流端(daily 写入),与 cognition 平行 —— cognition 负责"已落地后的认知循环",memory 负责"经历落地"
> - `structure.md` §4(retrieve 三种问法)
>
> **核心立场**:
> - 长期记忆不是"存 + 取"两个动作,是**写入 → 巩固 → 提取**的循环 —— 三段时间尺度不同(同步 / 周期 / 同步),设计形态不同
> - vault 是**事实层**,只承载经过 LLM 写入认证的关系;`meta/` 是**派生层**,承载概率推断的统计信号
> - 任一阶段独立演化,任一信号缺失系统降级而不崩

---

## 0. 心智循环:reme 的认知模型

agent 的长期记忆系统在概念上对应人脑的**海马—皮层回路 + 睡眠—觉醒周期**:

```
            ┌─────────────────────────────────┐
            │   外部经验(daily / resource)   │
            └─────────────┬───────────────────┘
                          │ (auto-memory 写 daily)
                          ▼
   ┌─────────────────────────────────────────────────┐
   │                                                 │
   │   ┌────────────────┐    抽象 / 关系编织         │
   │   │  Stage 1       │ ◄─ 类比 REM 睡眠           │
   │   │  auto-dream    │    "重放 + 写进 schema"   │
   │   └───────┬────────┘                            │
   │           │ 写 vault(digest body + wikilink)  │
   │           ▼                                     │
   │   ┌────────────────┐                            │
   │   │  vault(事实)  │                            │
   │   └───────┬────────┘                            │
   │           │ 只读                                │
   │           ▼                                     │
   │   ┌────────────────┐    长期组织 / 派生指标     │
   │   │  Stage 2       │ ◄─ 类比 NREM 慢波睡眠      │
   │   │ auto-consol-   │    "巩固 + 修剪 + 集群"   │
   │   │ idate          │                            │
   │   └───────┬────────┘                            │
   │           │ 写 meta/ + audit/(派生层)          │
   │           ▼                                     │
   │   ┌────────────────┐                            │
   │   │  meta(派生)   │                            │
   │   └───────┬────────┘                            │
   │           │ 只读                                │
   │           ▼                                     │
   │   ┌────────────────┐    query → 答案合成       │
   │   │  Stage 3       │ ◄─ 类比觉醒态 cue retrieval│
   │   │  auto-recall   │    "融合 + pattern complete"│
   │   └───────┬────────┘                            │
   │           │                                     │
   └───────────┼─────────────────────────────────────┘
               │ 召回结果给 agent
               ▼
            ┌─────────────────────────────────┐
            │           agent query           │
            └─────────────────────────────────┘
```

**心智循环回答四个根本问题**:

| 问题 | 谁回答 |
|---|---|
| 我经历过什么? | auto-memory(daily 入流) |
| 我从中学到什么? | Stage 1 — auto-dream |
| 这些知识如何长期组织? | Stage 2 — auto-consolidate |
| 我需要时如何调用? | Stage 3 — auto-recall |

memory 负责"经历落地",cognition 三阶段负责"已落地经历的认知循环"。

---

## 1. 三阶段全景

| 阶段 | 神经科学类比 | 时间尺度 | 改 vault | 实现归属 |
|---|---|---|---|---|
| **Stage 1 dream** | REM 重放抽象 | 同步(随入流即跑) | 是(写 digest body) | `auto_dream_design.md` |
| **Stage 2 consolidate** | NREM 深度巩固 | 周期 / idle(daily / weekly)| **否**(写 `meta/` + `audit/`)| `auto_consolidate_design.md` |
| **Stage 3 recall** | 觉醒态 cue retrieval | 同步(query 触发) | 否(只读;唯一对外写是 `meta/access_log.json`)| `auto_recall_design.md` |

**关键的不对称**:
- 写入与检索是**同步**的(用户 / agent 等待),巩固是**离线**的(idle / 周期)
- 改 vault 的资格被严格限制在 **dream + consolidate 中的 split** —— 其它阶段全只读
- 三阶段时间尺度差三个数量级,这是设计形态(同步 vs 异步 vs idle)的根本来源

---

## 2. 系统级能力(贯穿三阶段)

不属任何单阶段,但任一阶段不能违反:

| 能力 | 含义 |
|---|---|
| **事实层 vs 派生层分离** | vault 只承载经 LLM 写入认证的关系(显式 wikilink);`meta/` 承载概率推断的派生指标(community / recency / archived);两者绝不混同 |
| **不变量守恒** | F-invariants(0 文件移动 / 改正文限定 subject / wikilink 是 body 一部分)+ E-invariants(边守恒 E-1/E-2/E-3)横跨三阶段;详 `auto_dream_design.md` §4.3-§4.4 |
| **阶段独立演化** | 任一阶段算法升级不破坏其它阶段(community 算法换 → dream 不变;打分公式调 → consolidate 不变) |
| **缺失即降级** | 任一派生信号缺失,系统降级而不崩;冷启动可用 |
| **全程可审计** | 每阶段产 audit / report / log,人 / agent 可检视追溯 |

---

## 3. Stage 1 — auto-dream:经验 → 抽象

**类比**:REM 睡眠的记忆重放与抽象提炼。脑在做梦时把白天事件拆解、重组,提取出可泛化的模式,登记进皮层 schema。

**根本目的**:把"原始经历"转化为"长期值得调取的教训",同时把它编织进已有知识图谱。

### 3.1 五个能力维度

逻辑递进 —— 输入 → 抽象 → 整合 → 编织 → 写入:

| # | 能力 | 它在问什么 | 失效后果 |
|---|---|---|---|
| 1 | **抽象判断**(gate) | 这段材料里有"值得长期记住"的东西吗? | 噪声进 vault / 只蒸馏不抽象 |
| 2 | **经验重放**(召回) | 这个抽象在已有记忆里**已经存在**吗?以什么形式? | 重复节点 / 错过整合机会 |
| 3 | **整合决策** | 创建新节点,还是丰富已有节点?若已有 —— 是再次印证 / 精化范围 / 修正错误? | 已有信息丢失 / 错误没纠正 |
| 4 | **关系编织** | 这个抽象与谁有关系?谁是它的来源? | wikilink 缺失,后续 retrieve 漏召 |
| 5 | **写入安全** | 写入会不会破坏 vault 既有事实?并发冲突如何处理? | 边丢失 / race condition |

### 3.2 关键定性

- dream 是 vault 的**唯一写者**(在 cognition 三阶段里;memory 写 daily 不算)
- **写入瞬间是关系建立的唯一可信时机** —— 错过的关系不靠后台扫回(那不是 consolidate 的工作)
- 一次写入,所有未来检索受益(持久化优于实时计算)

详细机制见 `auto_dream_design.md`。

---

## 4. Stage 2 — auto-consolidate:抽象 → 网络

**类比**:NREM 慢波睡眠的系统巩固 + 突触代谢稳态。脑在深睡时把分散事件融入 schema、修剪弱连接、把长期不用的记忆淡出意识可达范围。

**根本目的**:跨时间累积地把 vault 从"一堆节点"组织成"有结构、有权重、有时效的网络",但**只产派生信号,不污染事实层**。

### 4.1 五个能力维度

按作用尺度从微观到宏观:

| # | 能力 | 作用尺度 | 类比 | 输出形态 |
|---|---|---|---|---|
| 1 | **结构维护** | 节点级 | 海马表征过密 → 分化新单元 | 改 vault(split,唯一例外)|
| 2 | **跨节点关系发现** | 节点对级 | 多次睡眠中识别"同一件事" → schema | `audit/` 报告 |
| 3 | **主题集群形成** | 子图级 | 皮层网络的功能性分区 | `meta/communities.json` |
| 4 | **时效性管理** | 节点级 / 时间维度 | 突触代谢稳态 + 遗忘 | `meta/access_log.json` + `meta/archived.json` |
| 5 | **健康监控** | 系统级 | 神经环路诊断 | 告警 / 严重告警 |

### 4.2 关键定性

- consolidate 是**纯只读 + 派生写**(读 vault,写 `meta/` + `audit/`)
- **唯一例外是 split** —— 改 vault 的维护任务,但触发严格(D3 inline 写后)且只改自身负责的 parent + children
- **关系判断有错率 → 报告优先,人/agent 介入,不主动合并**(夸大置信度的代价是污染事实层)
- 离线 / 周期 / idle —— 与前台不抢资源;失败不影响主流程,下次重跑

详细机制见 `auto_consolidate_design.md`。

---

## 5. Stage 3 — auto-recall:网络 → 答案

**类比**:觉醒态的 cue-driven retrieval + pattern completion。脑接到 query,激活相关皮层模式,补全成完整答案;同时召回过程本身强化被用到的记忆痕迹。

**根本目的**:接到当前 query 时,从 vault + 派生信号合成最相关的过去经验 —— 既要**覆盖率**(不漏)也要**信噪比**(不冗余)。

### 5.1 五个能力维度

按召回流程从输入到输出:

| # | 能力 | 它在解决什么 |
|---|---|---|
| 1 | **多路召回** | 不同问法走不同算子(state / semantic / topological 三分立);agent 自选,不强加聚合 verb |
| 2 | **多信号融合** | 单一文本相似度不够 —— 还要节点权威性 / 主题集群 / 时效性;乘法融合 |
| 3 | **信噪比管理** | 节点级去重 + 节点级 surface(frontmatter 一同呈现)+ multi-hop 可控展开 + 冷藏过滤 |
| 4 | **召回反馈** | 被命中的节点 → 写访问日志 → 影响下次 recency / archived 判定 |
| 5 | **鲁棒降级** | 派生信号缺失 → 退到基础召回;version 不兼容 → warning + 跳过该因子 |

### 5.2 关键定性

- recall 是**只读** —— 唯一对外写入是 `meta/access_log.json`(经 ring buffer + consolidate 聚合)
- recall **不引入新 L4 模块**(`structure.md` ✗-15)—— 三种问法分别由 L3 原子工具(`list_step` / `search_step` / `traverse_step`)直接覆盖
- 默认路径 **0 LLM 调用**(信号都是离线维护好的);LLM rerank / query rewrite 是 SDK 上层选项

详细机制见 `auto_recall_design.md`。

---

## 6. 能力地图(横切视角)

15 维按"作用对象"重排,可以看到三阶段如何分工:

| 作用对象 | dream(写入) | consolidate(巩固)| recall(检索)|
|---|---|---|---|
| **节点(单个)** | 1 抽象判断 / 3 整合决策 / 5 写入安全 | 1 结构维护(split) | 3 信噪比(节点级合并/surface) |
| **节点对 / 关系** | 4 关系编织(wikilink) | 2 跨节点关系发现(dups 报告) | (消费已有边,不产新关系) |
| **子图 / 集群** | 2 经验重放(召回邻居) | 3 主题集群形成(community)| 2 多信号融合(community boost) |
| **时间维度** | (写入瞬间) | 4 时效性管理(decay / archived)| 4 召回反馈(access log)|
| **系统健康** | 5 守恒校验 | 5 健康监控(D1 / D10) | 5 鲁棒降级 |
| **入口形态** | 异步 fan-out per sub-unit | 周期 batch / idle | 同步 query response |

**几个观察**:
- "节点对 / 关系"列在 recall 是空 —— recall 不产新关系,只用已有边(避免 query-time 高成本推断)
- "时间维度"行 dream 缺位 —— 写入瞬间无"时间维度"概念(那是 consolidate 后续才能提取的统计)
- 每行至少有一个阶段负责 —— 没有能力被全阶段忽略

---

## 7. 跨阶段不变量

所有阶段共同遵守的硬约束。任何阶段越界 = 设计错误。

### 7.1 F-invariants(继承 `auto_dream_design.md` §4.3)

| # | 约束 | 跨阶段含义 |
|---|---|---|
| F-1 | 0 文件移动 | 没有任何阶段可以 move 文件;rename 走 `wikilink_handler.retarget_links` 显式路径 |
| F-2 | 改正文限定 subject | dream 改 subject body / consolidate split 改 parent + children body;**recall 绝不改任何 body** |
| F-3 | maintainer 只做 split | consolidate 内的结构维护只做 split;无 merge / dissolve / re-edge |
| F-10 | inbound 不动 | split 后外部 wikilink 仍指 parent,不强制重定向 |
| F-11 | wikilink 是 body 一部分 | 没有"独立的边";所有关系变化是 body 编辑副作用 |

### 7.2 E-invariants(边守恒)

- E-1:dream update 出边 ⊇ 原出边
- E-2:split 后 `(parent_new ∪ ∪children_outbound) ⊇ parent_old`
- E-3:inbound wikilink split 时不动

**recall 不写 body** → E-* 与之无关;但 recall 看到的 wikilink 图永远是 dream / split 守恒后的状态。

### 7.3 派生信号边界

- **consolidate / recall 不写 vault** —— 关系判断、活跃度统计、社区划分都是概率推断,不污染事实层
- **`meta/*.json` 不被 retrieve 召回** —— 只作权重信号,不进入"召回结果"集合
- **audit/ 不被自动消费** —— 报告永远等待人 / agent 介入,不闭环回写

---

## 8. 跨阶段数据流(契约总览)

```
┌──────────────┐ wikilink ┌──────────────┐
│  auto-dream  │─落 body──►│   vault/    │
│  (Stage 1)   │           │  (事实层)   │
└──────────────┘           └──────┬──────┘
                                  │ 只读
                                  ▼
                           ┌──────────────────┐
                           │ auto-consolidate │
                           │   (Stage 2)      │
                           └─┬────────┬───────┘
                             │        │
              meta/ 元数据───┘        └─── audit/ 报告
              (派生层)                    (人工介入)
                    │
                    │ 只读
                    ▼
            ┌──────────────┐
            │ auto-recall  │ ◄─ user query
            │  (Stage 3)   │
            └──────┬───────┘
                   │ 命中钩子(异步)
                   ▼
              meta/access_log.json
              (recall 唯一对外写入,经 consolidate 聚合)
```

| 产物 | 路径 | 写入者 | 读取者 | 缺失行为 |
|---|---|---|---|---|
| **vault wikilink** | `digest/**.md` body | dream / split | recall(图遍历) | — |
| **dups 报告** | `audit/<date>/auto_link_dups.md` | consolidate | 人 / agent | — |
| **communities** | `meta/communities.json` | consolidate | recall | 不做同社区 boost |
| **access log** | `meta/access_log.json` | recall(写命中) + consolidate(聚合) | recall(读 recency)| recency_factor = 1.0 |
| **archived list** | `meta/archived.json` | consolidate | recall(默认过滤)| 不过滤 |
| **centrality** | `file_graph` 反向索引(实时,不存)| 自动 | recall(O(1) 查) | — |

**契约稳定性**:`meta/*.json` 都带 `version` + `computed_at`;recall 启动时校验 version,不兼容则降级。

**冷启动**:`meta/` 为空 → recall 仍能跑(base + centrality + 图)→ 排序略弱不崩。

---

## 9. 系统级断言(把"要什么"提炼到 5 条)

1. **抽象与事实分层** —— vault 是经 LLM 写过的事实;`meta/` 是统计 / 算法的派生;两者绝不混同

2. **关系建立的时机集中在写入瞬间** —— dream 写入是关系唯一可信来源;consolidate 不补 vault 关系,recall 不预存关系矩阵

3. **维护是离线的派生劳动,不是补救** —— consolidate 不修 dream 的疏漏(那叫返工),它做的是 dream 不擅长的事(全局视角 / 统计视角 / 时间视角)

4. **检索是融合,不是检索** —— recall 的价值不在"找文本相似",而在"把文本 / 图 / 时效 / 权威多个独立信号合成一个答案"

5. **整个心智循环可降级** —— 任一阶段失效或失准,整个系统降级而不崩;冷启动有意义;dogfooding 可演进

---

## 10. 与 auto-memory 的边界

auto-memory 写入的 daily event 节点也是图的一部分(承载 daily → digest 的 `derived_from::` 边)。但 daily 节点**不参与 cognition 三阶段的全部改造**:

| cognition 阶段 | 是否触及 daily |
|---|---|
| **dream** | 只读(作为入流之一) |
| **consolidate** | 不参与 dups / community / decay(daily 是时间索引,本质不去重 / 不冷藏) |
| **recall** | 三层并行召回时 daily 也参与命中(`structure.md` R-2 默认 `digest > daily > resource`) |

**关键约束**:cognition 三阶段任何子阶段都**不改写 daily**(无写回路径);daily 由 auto-memory 写完即只读。

---

## 11. 演进 / 待补

**当前实现状态**:
- ✅ Stage 1 dream 已实现并跑通(`reme4/steps/evolve/dream.py` + `dream.yaml`)
- ⏳ Stage 2 consolidate split 部分将实现;dups / community / decay / archived 待实现
- ⏳ Stage 3 recall 增强未实现(当前 search.py 已有 vector + keyword + RRF + 一跳 expand)

**顶层级演进议题**(不属任何单阶段):
- ⏳ **能力成熟度路标** —— 把 15 个能力维度按 M0(必须)/ M1(期望)/ M2(演进)分级
- ⏳ **跨阶段集成测试** —— vault 从空到充实的端到端 dogfooding,验证三阶段配合是否符合"心智循环"预期
- ⏳ **可观测性聚合** —— 三阶段各自的 audit / log 现在分散;是否需要统一的 cognition 健康面板

各阶段实现进度详见各自文档的"下一步"章节。
