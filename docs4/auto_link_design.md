# auto-link 设计(背景实体识别 + wikilink 写回)

> 本文档记录 reme4 中 **auto-link** 的设计讨论 —— 在已写入节点之间发现隐含关系,把这些关系作为 `[[...]]` wikilink **写回 body**,形成可见、可编辑的图结构增强。
>
> 配套阅读:
> - `structure.md` §1.2(三层数据视角)/ §4(retrieve 三种问法)
> - `auto_memory_design.md`:auto-link 可反向扫 daily event,补实体 wikilink(daily → digest)
> - `auto_dream_design.md`:wikilink 模型(§1.4 边语法 / §1.5 演化 / §1.5.5 边守恒 E-1 / E-2 / E-3);auto-link 借这套基础设施
> - `auto_maintain_design.md`:CAS 写入协议(§5);auto-link L1 写回与 dream G\* / maintain split 三方共用同一套 CAS
>
> **三层对应**:reme 服务整体三层 —— auto-memory / auto-dream / **auto-link(本文档)**。auto-link 是图关系的**后置增强** —— 在已落地的 vault 上做实体识别 + wikilink 写回,补足 content link(写记忆时由 LLM 直接产生的 `[[...]]`)在长 tail 隐含关系上的盲区。
>
> **核心立场**:auto-link **写回 body**,不只是产报告。生成的 wikilink 是**可见、可编辑**的(写在 Markdown 文件里),agent / 人可后续 curate。auto-link 不引入新材料,纯 additive 插入 wikilink,天然满足 E-1 守恒;复用 dream 的 CAS 写入协议,不引入新基础设施。

---

## 0. 问题陈述

content link(`auto_dream_design.md` G\* / split 写入时由 LLM inline 产生的 `[[...]]`)解决了"写记忆时显式的关系"。但有一类关系不会在 inline 写入时自然涌现,需要后台扫描已写入的 vault 才能识别:

1. **历史 body 的实体未链接** —— G\* update 时 LLM 关注新材料融入,可能忽略已有 body 中某个未链接的实体(例如 body 提到 "JWT" 但没写 `[[digest/auth/jwt-overview.md]]`)
2. **跨节点 / 跨桶的隐含关联** —— 节点 A 提到 "rate limit",但 `digest/api/rate-limit.md` 是后来才被 split 创建 → A 写入时没机会建立这条边
3. **同主题未连 / 同概念重复** —— G\* 漏判去重把同概念建成两个节点;或两个主题相关但 0 链接的节点彼此不知晓

auto-link 承担这部分:**后台扫描已写入节点 → 实体识别 / 候选挖掘 → wikilink 写回 body**。

---

## 1. 已对齐决策

### 1.1 与 content link 的边界

| 维度 | content link(在 dream) | auto-link(本文档) |
|---|---|---|
| 何时产生 | 写记忆 inline:G\* update / M split prompt | 后台扫描:离线 / 周期 / 触发后异步 |
| 由谁产生 | LLM 在 dream 写入流中顺手写出 | LLM 在 auto-link 扫描流中识别后写出 |
| 输入 | 新材料 + 召回候选节点 | 已写入 body + 全 vault 索引 |
| 改 body | 是(重写整段 body) | 是(纯 additive 插入 wikilink,不改文字) |
| 守恒 | E-1 强守恒(out ⊇ old) | E-1 天然满足(纯增) |
| 用途 | 写入即关系明示 | 弥补 inline 漏判,挖掘长 tail 关系 |

### 1.2 写回模型:纯 additive,复用 dream CAS

auto-link 写回是**纯 additive** 操作 —— 在已有 body 文字中找到实体 mention,替换为 wikilink 形态:

```
Before:  "JWT 轮换的核心是密钥派生 ..."
After:   "[[digest/auth/jwt-rotation.md|JWT 轮换]]的核心是[[digest/auth/jwt-key-derivation.md|密钥派生]] ..."
```

| 维度 | 决策 |
|---|---|
| **alias 必须保留原文** | `[[path.md\|<原文>]]` 形态;原文一字不改 —— 守住"不改写其它节点正文" (`auto_dream_design.md` §1.5.4 F-2) 的精神 |
| **predicate 默认为空** | auto-link 默认产生无谓词 wikilink;升 typed link 走 L3(详 §1.3) |
| **不引入 anchor** | 与 dream 一致(`auto_dream_design.md` §1.4 / §3.13);target 永远是节点路径 |
| **CAS 写入** | 完全复用 `auto_maintain_design.md` §5 的 read-stamp + CAS-write 协议(冲突重做 ≤ 3 次) |
| **E-1 守恒** | 纯 additive:`new outbound = old outbound ∪ {new wikilinks}`;`new ⊇ old` 天然满足,守恒校验默认通过 |
| **rollback** | 若 auto-link 误插入(例如 entity mention 是同名歧义),走标准 edit 或 retarget 撤销;auto-link 不维护"我插过哪些"audit log(留给 SDK 决定) |

**为什么是 additive 而不是重写**:
- additive = 0 文字风险(原文不变,只在原 mention 周围加 `[[ | ]]` 包装)
- 重写 = 触发完整 E-1 守恒校验 + LLM 重写整段语义守恒 prompt + 多次 LLM 调用 = 跟 G\* update 重复
- additive 失败可见:产生坏 wikilink 时,人/agent 直接编辑 body 修就行

### 1.3 候选挖掘类型(L1-L4)

| # | 类型 | 描述 | 写回形态 |
|---|---|---|---|
| **L1** | **实体识别**(主路径) | 扫 body,识别已是 digest 节点的实体名(模糊匹配 + 语义召回);未被 wikilink 化的 mention → 加 `[[path.md\|<mention>]]` | additive wikilink 插入 |
| **L2** | **同主题未连**(旧 D7) | 两个 digest 节点谈相关主题但 0 wikilink → 候选 add link;LLM 判后在 body 末尾追加一句引用 | additive(在合适位置 / 节末追加 `参见 [[other.md\|other]]`)|
| **L3** | **隐含 predicate 推导** | 已有 `[[A]]` 但 LLM 可推断关系类型(`is_a` / `causes` / `extends` / ...)→ 升级为 typed link | 改 `[[A]]` → `is_a:: [[A]]`(predicate 升降级走显式 audit,详 §2.1)|
| **L4** | **重复语义检测**(旧 D8) | 两个节点描述同一概念但被独立 create(G\* 漏判去重)→ 候选 merge | **不写回**;产报告 + 提示人/agent 触发 G\* update 路径手工合并 |

**L1 是主路径** —— 它是 auto-link 最核心、最频繁、最高 ROI 的操作:每个 digest 节点写完后,后台扫一遍 body,找未链接的已知实体,additive 加 wikilink。

**L2-L3 是辅助** —— 周期扫,产候选,LLM 终判,写回部分(L2 节末追加 / L3 升 predicate)。

**L4 不写回** —— 节点合并是结构改动,影响 E-1 守恒边界 + inbound 链路 + provenance 链路,不适合自动写;auto-link 只产报告,人/agent 决定走 G\* update 路径解决。

### 1.4 触发节奏

| 模式 | 何时 | 适用 |
|---|---|---|
| **inline post-write**(默认) | 每次 G\* update / M split 写完 body → enqueue auto-link L1 job(异步,FIFO,CAS 保护)| L1 实体识别;反应即时,与 D3 写后检测同节奏 |
| **周期 batch**(可选)| cron(daily / weekly)扫全 vault | L2 / L3 候选挖掘;成本可控 |
| **手动触发** | SDK / 人显式调用 | 全量重扫 / 修复 |

**L1 inline 的必要性**:新 split 出的 child 节点立即被既有 body 引用(用 wikilink 而非纯 mention)的关键 = 写入即扫描;不 inline 会让"刚创建的 child 节点"在很长时间内只有 split parent 一个 inbound,中心性失真。

**已排除**:
- inline 时同步 auto-link(阻塞 G\* return)—— 时延不可接受;auto-link 始终异步
- 所有 L\* 都 inline —— L2-L3 候选挖掘 RTL 跨节点,成本高,只适合 batch
- 全 cron 唯一触发 —— L1 滞后过久,新节点孤岛

### 1.5 中心性算法(retrieve 加权依赖)

retrieve 时节点权重 = base × intent 调节 × **中心性增益**(详 `auto_dream_design.md` §1.5.8)。中心性需要 auto-link 这一层提供 —— content link 给底子,auto-link 补 long tail,二者合起来才是完整的图。

| 选项 | 优点 | 缺点 |
|---|---|---|
| **简单入度** | 实现最简;split parent 入度天然高;auto-link L1 加边后入度即时反映 | 不区分"权威节点"vs"被随手提的节点";高入度 ≠ 高权威 |
| **PageRank** | 经典;权威性传递 | 实现复杂 + 增量更新成本(每次写边重算成本高,需 incremental algorithm)|
| **eigenvector centrality** | 与 PageRank 相近 | 同上 |

**首版决策**:**简单入度**(file_graph 已有 inbound 链表,O(1) 查);auto-link L1 加边后入度立刻更新,split parent 自然涌现高入度。dogfooding 后视 retrieve 质量演进。

中心性是 retrieve 时**查询时计算**,不预存:
- file_graph 已建反向索引(inbound),计算 `len(inbound(node))` 是 O(1)
- 不预存避免"加边后中心性陈旧"问题
- PageRank 演进时可加增量计算 + 周期 refresh

---

## 2. 待对齐边界点

### 2.1 L3 predicate 升降级的 audit

L3 把 `[[A]]` 升级为 `is_a:: [[A]]` 时,**改了 edge identity** —— `(target, None)` 变成 `(target, "is_a")`,在 E-1 守恒视角下 = 删一条边 + 加一条边:

```
old outbound: {(A, None)}
new outbound: {(A, "is_a")}
diff:         missing = {(A, None)}; added = {(A, "is_a")}
```

不打 audit 走默认会被守恒校验拦下(`missing != ∅` → 重试 / 拒写)。

**决策方向**:
- L3 写入必须打 audit flag(消费层意图:升级 predicate,允许 drop + add 同时发生)
- audit flag 由 reme4 step 暴露(`maintainer_step(action="predicate_upgrade", from=..., to=...)`),不放在普通 write 路径
- 普通 G\* / auto-link L1 写入永远不带 audit flag,守恒校验照常严格

详细 audit flag 接口形态留到 SDK 阶段。

### 2.2 多歧义实体识别

L1 扫 body 找 "JWT" 这个 mention,vault 中有 `digest/auth/jwt-overview.md` 和 `digest/payment/jwt-payment-flow.md` 两个 candidate:

候选方案:
- LLM 上下文判 —— 把 body 周围段落给 LLM,选最相关 target
- 跳过模糊 case —— L1 只处理 unambiguous mention,歧义 case 留人/agent
- 全部链 —— `[[overview]][[payment-flow]]`,后续人 curate

**首版**:LLM 上下文判(每个候选 candidate 提供 description / 周围若干节点 summary,LLM 选择 top-1 或 drop);成本可接受(扫描已是离线 batch)。

### 2.3 auto-link 写回与 G\* / split 的并发

auto-link 写 body 走 §1.2 CAS,但有特殊情况:
- 同节点同时被 G\* update 与 auto-link L1 写入 → CAS 协议自动序列化 (`auto_maintain_design.md` §5):后到者重做
- auto-link L1 写完后立刻被 G\* update 覆盖(G\* 重写 body) → 看 G\* prompt 是否守住 auto-link 加的 wikilink(E-1 强守恒 → 守住)
- auto-link L1 与 D3 派发的 split job 同节点并发 → split 先到 / 后到都不影响最终拓扑(split 把 body 拆成 parent + children,auto-link 加的 wikilink 跟着对应内容段自然分配到 parent / child)

**结论**:CAS + E-1 + E-2 守恒已覆盖所有并发场景,auto-link 不需要新协调机制。

### 2.4 跨 vault / 跨进程

M0 单 reme 实例 + 单 vault,auto-link 走内进程 enqueue;多实例 / 跨进程留 M1+(同 `auto_maintain_design.md` §5)。

### 2.5 实体识别 vs 现成 NER 库

L1 实体识别可选:
- LLM 直接扫(贵但灵活,与 digest 节点同构)
- 现成 NER 库(spaCy 等)预筛 + LLM 终判(快但 entity 类型与 digest 节点形态可能不匹配)
- 纯字符串匹配(已知节点名字 + 简单变体)+ LLM 终判 ambiguity

**倾向**:从纯字符串匹配 + LLM 终判 ambiguity 起步(实现最简,效果可能已经够好);视 dogfooding 决定是否引入 NER 库。

---

## 3. 与其它层的协作

| 上下游 | 关系 |
|---|---|
| ← **auto-dream** | dream 写完一个节点 → 通过 inline post-write enqueue auto-link L1(§1.4);auto-link 用 dream 的 CAS 协议 |
| ← **auto-memory** | auto-link 可反向扫 daily event,把实体识别成 `[[digest/...]]`(daily → digest);auto-memory 写入端不主动调 auto-link,触发同 dream 路径 |
| → **digest body** | 主要写入对象 —— L1 additive 加 wikilink / L2 节末追加引用 / L3 升 predicate(走 audit) |
| → **daily body** | auto-link 扫 daily event 时同样可加 `[[digest/...]]`(I-2 daily 单作者需协调:auto-link 应在 event 关闭后才动该 event,不与 active event 并发改;实现细节留 step 层处理) |
| → **resource body** | I-3 immutable;auto-link **不写 resource**(reading-only) |
| → **L4 候选 report** | L4 重复语义检测产报告,落 `audit/<date>/auto_link_l4.md`(具体路径 / 形态留 step 层) |

---

## 4. 与 auto-dream 模型的引用关系

本文档复用 dream 定义的底层模型,所有具体规则在 `auto_dream_design.md` 中:

| 引用 | 来源 |
|---|---|
| wikilink 基础语法(`[[path.md\|alias]]` / predicate) | `auto_dream_design.md` §1.4 |
| 节点 / 边模型 | `auto_dream_design.md` §1.5 / §1.5.1 |
| F-invariants(F-1..F-11)| `auto_dream_design.md` §1.5.4 |
| 边守恒 E-1 / E-2 / E-3 | `auto_dream_design.md` §1.5.5 |
| 路径即 ID / rename | `auto_dream_design.md` §1.3 |
| CAS 写入协议 | `auto_maintain_design.md` §5 |
| anchor 不引入 | `auto_dream_design.md` §1.4 / §3.13 |
| SearchStep 召回 | `auto_dream_design.md` §3.10 |

---

## 5. 下一步

1. **L1 实体识别 step 实现** —— 字符串匹配 + 语义召回 + LLM ambiguity 终判 + additive wikilink 写回(§1.2 / §1.3)
2. **inline post-write trigger 接入** —— G\* update / M split CAS 写入成功后 enqueue auto-link L1 job(§1.4)
3. **L2 / L3 周期 batch 框架** —— cron(daily / weekly)+ 候选挖掘 prompt + 写回路径(§1.3)
4. **L3 audit flag 接口** —— `maintainer_step` 提供 `predicate_upgrade` 操作,带 audit context 走特殊守恒规则(§2.1)
5. **中心性 retrieve 增益** —— file_graph inbound count → retrieve 加权乘子(§1.5)
6. **L4 报告框架** —— 重复语义检测产报告,提供 SDK / 人介入入口(§1.3 / §3)

实现进入 `reme4/steps/jobs/` 与 `reme4/file_graph/` 时,本文档与 `auto_dream_design.md` 共同作为契约依据。
