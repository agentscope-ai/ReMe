# auto-consolidate 设计(Stage 2 巩固:主动解决 vault 长期演化的实际问题)

> 本文档:reme4 中 **auto-cognition 三阶段** 的 **Stage 2 — 巩固阶段** 实现。覆盖 vault 长期演化中累积的实际问题(冗余 / 过载 / 稀疏 / 腐败 / 抽象缺位),通过周期 batch + 写后 inline 的方式**主动改 vault**,让记忆系统保持健康。
>
> 配套阅读:
> - `auto_cognition_design.md`:三阶段顶层心智循环
> - `auto_dream_design.md`:Stage 1 写入 / 节点 + 边模型 / F-invariants 原始定义 / 边守恒
> - `auto_recall_design.md`:Stage 3 检索 —— 消费本文档产出的信号
> - `auto_memory_design.md`:auto-memory 写 daily,daily 节点不参与本文档的巩固改造
> - `structure.md` §3.6(maintain 动作语义)
>
> **核心立场**:
> - consolidate **不是产报告等人介入**,是**主动解决问题** —— 类比 NREM 慢波睡眠的 systems consolidation:跨多事件抽 schema、修剪弱连接、稳态突触强度。这些都是真实发生的改造
> - vault **会被 consolidate 改**,但每个动作有严格的**置信度门槛 + 守恒规则 + 审计 trail + 渐进 rollout**
> - 灰色地带(置信度不够)才产报告等人介入;高置信度自己解决
> - **community detection 是巩固的中枢** —— P0 基础设施,P1-P3 三个动作(abstract / merge / reinforce)都依赖它

---

## 0. 问题陈述与五大动作全景

dream 写入是单点视角,有三类视野局限:**写入瞬间没有跨节点视角 / 跨时间视角 / 全局拓扑视角**。这些局限会让 vault 长期演化中累积五类实际问题:

| # | 问题 | 类比 | 表现 | 解决 |
|---|---|---|---|---|
| 1 | **冗余** | 同事件留下重复记忆痕迹 | dream 漏判去重 / 术语演化 / 跨桶建成两份 | merge |
| 2 | **过载** | 单一突触表征过密 | 节点 body 累积过长 / 单节点杂糅多主题 | split |
| 3 | **稀疏** | 应有连接未建立 | dream 写入瞬间漏召回的相关节点 / 反复共现但无 wikilink | reinforce |
| 4 | **腐败** | 长期不激活的痕迹 | 旧节点过时 / 半年没人读 / 内容已被矛盾 | archive |
| 5 | **抽象缺位** | 跨多 instance 缺 schema | vault 只有原子节点,没有"主题层"视角承接全局问 | abstract |

### 0.1 四大动作 + 优先级

| 优先级 | 动作 | 解决问题 | 触发节奏 | 改 vault | 风险 | 收益 |
|---|---|---|---|---|---|---|
| **P0** | **community detection** | (基础设施) | weekly batch | 否 | 0(只产 meta) | 基础(下游动作的依据)|
| **P1** | **abstract** | 抽象缺位 | weekly batch(基于 P0) | 是(新建 summary) | 低(additive) | **最高**(GraphRAG 核心) |
| **P2** | **merge** | 冗余 | weekly batch(基于 P0) | 是(合并 + retarget) | 高(lossy) | 中(消除可见冗余) |
| **(独立)** | **split** | 过载 | inline 写后(D3) | 是(拆 parent + children) | 低 | 中 |
| **(独立)** | **archive** | 腐败 | daily batch | 软(meta 标记) | 0 | 中 |
| ~~P3 reinforce~~ | **已并入 dream synapse** | 稀疏 wikilink | 由 dream Phase 2 step 4 织突触承担 | (不在 consolidate 范围内) | — | — |

**关键论断**:
- **P1 比 P2 优先** —— abstract additive 失败可逆且回报最大;merge lossy 失败要回滚 inbound,价值是消除冗余(必要但不增能力)。
- **reinforce 已取消**(2026-06-02)—— 详 §4 标作废说明;wikilink 稀疏的解决方案是 dream Phase 2 在写入瞬间多召回 + 织突触(详 `auto_dream_design.md` §4.2.2),不再由 consolidate 周期补救。

### 0.2 实施路径

```
M0:  P0 community detection (基础设施)
     + split (已实现)
     + archive (软标记,完全可逆)

M1.1: P1 abstract (additive,最低风险开始改 vault)
M1.2: P2 merge   (lossy,高门槛 + 多数票)

M2+: 多层 abstract (L2 super-community) / delete

reinforce: 不再排期 —— 已由 dream Phase 2 synapse 织突触承担
```

### 0.3 显式排除

- ❌ 重做"抽象判断" —— gate 决策只在 dream(consolidate 不重新判定"该不该记")
- ❌ 重做"语义内容" —— UPDATE 三种 flavor(CORROBORATE / REFINE / CORRECT)只在 dream;consolidate 做结构层,不做语义层
- ❌ 改 daily / resource —— consolidate 只动 digest 节点(I-2 / I-3 仍守)

---

# Part A — community 工作群(本文档核心)

P0-P3 四件套围绕 community detection 协同工作:**community 提供"哪些节点同主题"的判据,abstract / merge / reinforce 各自利用这个判据做不同的解决动作**。

## 1. community detection(P0,基础设施)

**目的**:在 vault wikilink 图上做 community detection,产出"节点 → community_id"映射。这是 P1-P3 三个动作的**唯一前置**。

### 1.1 算法选择:Leiden

| 选项 | 评估 |
|---|---|
| Louvain | 经典,但有 resolution limit + disconnected community 风险 |
| **Leiden** ✅ | Louvain 改进版(2019),稳定性显著好;GraphRAG 采用;Python `igraph.community_leiden` 现成 |
| label propagation | 实现最简,但结果不稳定(随机种子敏感) |

**首版决策:Leiden**,直接对齐 GraphRAG 路线,后续接它的多层抽象更顺。

### 1.2 图的形态

| 维度 | 决策 |
|---|---|
| **节点范围** | **只 digest 节点**;daily / resource 不参与 |
| **边权重** | **首版 unweighted undirected**(所有 wikilink 等权)—— 加权方案(predicate 类型加权)留 M2+ 视效果 |
| **跨桶 community** | **必须允许** —— bucket 是物理归档,community 是语义聚合,二者本就正交。"错桶节点"会被自然纳入 community,可作 audit 信号但不强制 move(F-1 守住)|
| **resolution** | **1.0 起步**(Leiden 默认 / GraphRAG 默认)—— dogfooding 后视 community 平均规模(理想 5-15 节点)调 |
| **更新模式** | **全量重算**;vault 千节点级 Leiden < 1 秒,M0/M1 不引入增量复杂度 |

### 1.3 多层级:M1 只 L1

| 层数 | 适用 | reme 决策 |
|---|---|---|
| 单层 L1(原子 → community)| vault < 500 节点足够 | **M1 起步** |
| 双层 L1 + L2(community → super-community) | vault > 500 节点 / 跨主题大类涌现 | M2+ 视规模 |
| GraphRAG 4 层 | 大规模文档库 | M3+ 不优先 |

理由:GraphRAG 论文证明 L1 拿走 60-80% 效果。先把 L1 跑稳,L2 看实际是否需要。

### 1.4 输出

**`meta/communities.json`**:
```json
{
  "version": 1,
  "computed_at": "2026-06-08T03:00:00Z",
  "algorithm": "leiden",
  "resolution": 1.0,
  "communities": {
    "digest/auth/jwt-rotation.md": "c_07",
    "digest/auth/oauth-flow.md": "c_07",
    "digest/api/rate-limit.md": "c_12"
  },
  "stats": {
    "n_communities": 14,
    "median_size": 7,
    "max_size": 23
  }
}
```

**`meta/community_changes.json`**(供 abstract 稳定度判据):
```json
{
  "computed_at": "...",
  "previous": "...",
  "stability_per_community": {
    "c_07": 0.92,   // 1 - (Jaccard 距离与上周该 community 节点集)
    "c_12": 0.45    // 不稳定,abstract 跳过
  }
}
```

### 1.5 community_id 不需要稳定

下游(abstract / merge / reinforce)只关心"两节点是否同 community";id 本身可重排。每周重算后 id 不需要保持与上周对齐。stability 信号通过节点集 Jaccard 距离计算,不依赖 id。

### 1.6 用途总览

| 下游 | 用法 |
|---|---|
| **abstract**(§2)| 判据"该 community 节点数 ≥ N + 稳定度满足 + 无 hub" → 创建 summary |
| **merge**(§3)| 候选 pair 必须在同 community(降错率;不同 community 的相似 description 多是同名异义)|
| **reinforce**(§4)| 候选 wikilink 必须在同 community(避免假关联)|
| **recall**(`auto_recall_design.md` §3) | 同 community 节点 boost |

---

## 2. abstract(P1,抽象提升)

**类比**:NREM systems consolidation —— 跨多次睡眠把分散事件抽出共同 schema,从 episodic 升到 semantic。

**目的**:vault 演化到一定规模后,某些 community 形成稳定主题群,需要一个 hub 节点统领,让 retrieve 能召回到"主题概览"而非散点。

### 2.1 等价处理立场(关键)

**summary 节点完全等同普通节点**:

| 维度 | 决策 |
|---|---|
| **路径** | LLM 选桶,正常 slug 命名(如 `digest/auth/authentication-mechanisms.md`);**无 `__community__` / `__hub__` 等结构性标识** |
| **frontmatter** | 仅 `name + description`(reme 核心保留);**无 `kind: community_summary`、无 `auto_generated`** |
| **summary 性质** | 完全体现在 **body 形态** —— 主题概述 + 列出 source 节点 wikilink + 跨节点 pattern;但这是内容自然形态,不是结构性宣告 |
| **后续维护** | **无** —— 跟其它节点等价,被 dream / split / merge / archive 自然演化(参见 §2.6) |

这跟 dream 的核心立场对齐:"节点角色由 body 内容决定,不由 frontmatter 类型标记"。abstract 是"用一种新方式创造节点",不是"创造一种新节点类型"。

### 2.2 触发判据(组合门槛)

```
weekly batch:
  for community in communities.json:
    if community_has_hub(community):           # §2.5 结构化判据
      continue
    if len(community) < MIN_NODES (5):         # 节点数门槛
      continue
    if stability(community) < 0.7:             # 稳定度门槛
      continue
    if active_node_count(community, 30d) < 3:  # 活跃度门槛
      continue
    if name_diversity(community) < 0.5:        # 多样性门槛
      continue
    → enqueue abstract job
```

| 门槛 | 默认 | 含义 | 防的是 |
|---|---|---|---|
| **节点数** | ≥ 5 | community 大小 | 给 2-3 节点造 hub 不划算 |
| **稳定度** | ≥ 0.7 | 与上周边界 Jaccard 距离 | 给短命 community 造 hub 浪费 |
| **活跃度** | ≥ 3 节点近 30 天 hit | community 仍在用 | 给死社区造 hub(下次没人看)|
| **多样性** | name 差异度 ≥ 0.5 | frontmatter `name` 互不相同 | 给"一组重复节点"造 summary —— 那是 merge 的事 |

### 2.3 创建动作 + grounding 守恒

```
LLM 看 community 内所有节点 (frontmatter + body)
  ↓
产 planned summary body (三段):
  1. 主题概述 (1-2 段,跨多节点共同主题)
  2. 关键支柱 (列表,3-5 节点 + 一句话 + wikilink)
  3. 不在概览的细节 (明说哪些细节留原节点)
  ↓
长度限制: summary body < 1500 token
  (防 abstract 创建后立刻被 split 触发,§5)
  ↓
LLM 决定 path: digest/<bucket>/<slug>.md
  ↓
CAS 写入 (§9) + 双重守恒校验:
  - 机械: 出边集合 ⊇ "关键支柱"声称引用的节点 (防套话)
  - 机械: 出边集合 ⊇ source_nodes 的至少 60% (allow LLM 漏列少数)
  ↓
audit 记录: audit/<date>/consolidate_actions.md
```

**grounding 守恒**:summary body 中**声称引用某节点必须真写 wikilink**。LLM 不能仅口头提及"我们在 X 中看到..."而不带 `[[X.md]]`。这是机械可校验的,LLM 跑不掉。

### 2.4 长度限制为什么重要

summary body < 1500 token 是**与 split 互锁的机制**:

- 不限长 → LLM 会写"完整覆盖" → 最终 body 累积接近 split 阈值(2000 token)→ 下次 D3 触发拆 → 拆出来的 children 又被 community 视为同主题 → 下次 abstract 又造一个 hub → 循环
- 限长 1500 → summary 留出 split 阈值的 25% buffer,稳定不触发拆

### 2.5 "community 已有 hub"的结构化判据

不靠 frontmatter / 路径标识,靠**结构**:

```
def community_has_hub(community):
    for node in community:
        out_targets = outbound(node) ∩ community
        if len(out_targets) / len(community) >= 0.6:
            return True   # 该节点出边覆盖 community 60% 以上 → 它已是 hub
    return False
```

**好处**:
- split parent overview 自然被识别为 hub(split parent 出边覆盖大部分 children)→ abstract **复用** split 的工作,不重复创建
- 已有 abstract 创建过的节点,只要它出边没退化,下次 batch 自然识别为 hub,不重复创建
- 节点被 dream update 后形态变化,出边变了 → 自动重新评估

**M1 实施关键验证点**:跑实测验证这个涌现 —— split parent 是否真被识别为 hub。如有 corner case,调阈值 0.6 → 0.5 / 0.7。

### 2.6 后续维护:无 —— 完全靠 5 大动作演化

abstract 创建即放归 vault,**consolidate 不再"管"它**。后续命运:

| 演化路径 | 结果 |
|---|---|
| 新材料触及该主题 | dream update 自然修正 body(走 CORROBORATE / REFINE / CORRECT)|
| 老 summary 长期不被引用 | archive 自动归档(§6)|
| community 边界变了 → 下次 batch 创建新 summary | 新老 summary 描述同主题 → merge 自动合并(§3)|
| summary body 累积过长 | split 自动拆(§5)|

这是真正的"vault 自我代谢"。**没有特殊维护通道**。

---

## 3. merge(P2,同概念合并)

**类比**:NREM 跨多次睡眠识别"同一件事" → 合一个记忆痕迹。

**目的**:消除 vault 内的冗余 —— 同概念多节点。

### 3.1 候选挖掘(community 内三层过滤)

```
weekly batch (依赖 community detection):
  for community in communities:
    pairs = all_pairs(community)
    for (A, B) in pairs:
      if description_sim(A, B) < 0.6:        # 第一层: frontmatter 相似
        continue
      if body_topic_overlap(A, B) < 0.5:     # 第二层: body 主题词重合
        continue
      if cooldown_active(A) or cooldown_active(B):  # 第三层: cooldown 检查
        continue
      candidates.append((A, B))
```

**关键约束**:候选必须在**同 community**(降错率)。

### 3.2 多数票决策

merge 是高风险动作(lossy + 改 inbound),用多数票降错:

```
for (A, B) in candidates:
  votes = parallel_run(N=3, prompt="A 和 B 是否同一概念? 返回 {is_same, confidence}")
  agree = sum(v.is_same and v.confidence >= 0.8 for v in votes)
  if agree >= 2:
    → enqueue merge job
  elif agree == 1:
    → 写 audit/<date>/dups_uncertain.md (灰色地带,人介入)
  else:
    → 丢弃
```

### 3.3 merge 动作:body 重写归 consolidate(方案 B)

**关键决策**:merge 后的 body 由 **consolidate 自跑合并 prompt**,不走 dream update 路径。

| 方案 | 评估 | 决策 |
|---|---|---|
| A. 走 dream update 路径(把 loser body 作"新材料")| 优雅但跨阶段;dream 不应知道 caller 是 consolidate 还是新材料 | ❌ |
| **B. consolidate 自跑合并 prompt** | 简单自包含;通过严格 prompt 约束化解"做语义工作"张力 | ✅ |
| C. 不重写 body(留 redirect stub) | 完全不做语义,但 vault 留无用节点 | ❌ |

**B 方案的边界守住**(避免 consolidate 真在做语义判断):

| 边界 | 含义 |
|---|---|
| **prompt 严格约束** | "只合并不精化" —— 不重写措辞、不加新内容、不做精化决策 |
| **机械守恒** | 出边 ⊇ A.outbound ∪ B.outbound + provenance 全保留(LLM 跑不掉) |
| **信息守恒抽样** | LLM 自检 "merged.body ⊇ A.body ∪ B.body 全部信息";audit 抽样人审 |
| **失败拒写** | 守恒校验失败 → LLM 重试一次 → 二次失败拒写 + audit |

### 3.4 完整动作流

```
A, B → 选择 winner (path):
  - inbound 数大者赢 (保护既有 inbound,降 retarget 量)
  - 平局取路径短者
  ↓
LLM 跑 merge prompt → planned merged_body (B 方案)
  ↓
机械 retarget 准备:
  - 扫所有 inbound(loser): [[loser.md]] → [[winner.md]]
  - alias 保留;predicate 保留
  - 这是机械算子,非 LLM
  ↓
事务式 CAS 写入:
  1. winner body 改写
  2. 所有 inbound 节点 body 改写 (retarget)
  3. 删除 loser 文件
  任一步失败 → 全部回滚
  ↓
audit 记录 + cooldown 设置 (winner 进 cooldown 2 weeks)
```

### 3.5 灰色地带:报告

- 多数票通过(agree ≥ 2)→ 自动 merge
- 仅 1 票通过 → 写报告 `audit/<date>/dups_uncertain.md`,人 / agent 介入
- 0 票 → 丢弃

报告格式:
```markdown
# dups uncertain 2026-06-08

## pair 1 (1/3 votes)
- A: digest/auth/jwt-rotation.md ("JWT 密钥轮换")
- B: digest/security/key-rotation.md ("密钥轮换原则")
- vote 1 (yes, 0.85): "同一概念,A 偏 JWT 场景"
- vote 2 (no, 0.72): "B 是通用原则,A 是具体应用"
- vote 3 (no, 0.68): "粒度不同,不应合并"

建议:走 dream update 通道把 A 内容作为 B 的实例并入。
```

---

## 4. ~~reinforce~~(**已作废,2026-06-02**)

> ⚠️ **本节作废,reinforce 已并入 dream Phase 2 synapse 织突触**(详 `auto_dream_design.md` §4.2.2)。理由:
> - reinforce 的本质 = "找语义相关但 wikilink 缺失的节点对,补 wikilink"
> - 但 dream Phase 2 在写入新节点瞬间已经在做同样的事(多召回 + 内化判 related + 织 `[[Y.md]]`)
> - 让 consolidate 周期事后补 wikilink = dream RECALL 不充分的兜底,与其兜底不如把 dream 召回做强
> - F-2 自然守住:dream 只动新节点 body(自己的 subject),不需要 consolidate 改 leaf body 这种 F-2 破例
>
> **新立场**:wikilink 的稀疏由 dream Phase 2 在写入瞬间一次性解决,vault 不维护"事后周期补 wikilink"的通道(`auto_cognition_design.md` §9.2 立场:关系建立在写入瞬间)。详 `hierarchical_summary.md` §13.2 Q4。
>
> 以下保留原 reinforce 设计内容作为历史快照,**不实施**。

**(以下内容已作废,仅作历史快照)**

**类比**:NREM 突触强化 LTP —— 反复共激活的连接被强化。

**目的**:vault 演化中,某些节点对应该有 wikilink 但 dream 写入时漏召。reinforce 周期检测并 additive 补。

### 4.1 候选挖掘(三层过滤)

```
weekly batch (依赖 community detection):
  for community in communities:
    for (A, B) in all_pairs(community):
      if has_wikilink(A, B):
        continue
      # 第一层: 字符串 mention 锚点
      if not has_mention(A.body, B.frontmatter.name):
        continue
      # 第二层: embedding 相似度验证
      if embedding_sim(A.context_around_mention, B.body) < 0.7:
        continue
      # 第三层: 同 community (已经是,但显式说明)
      candidates.append((A, mention_pos, B))
```

**三层过滤的角色**:

| 层 | 防的是 |
|---|---|
| 字符串 mention | 大幅降候选数(从 O(N²) 降到 O(实际共现)) |
| embedding 相似度 | 防同名异义("Apple" 公司 vs 水果)|
| 同 community | 防表面术语共现但语义无关 |

### 4.2 决策(单票即可,门槛较高)

reinforce 是 additive 低风险动作,不需要多数票:

```
for (A, mention_pos, B) in candidates:
  vote = LLM("A.body 在该位置提到 B 的概念。是否合理加 [[B.md]] 链接?")
  if vote.confidence >= 0.85:
    additive_wikilink(A, mention_pos, target=B.path)
    → CAS 写入 (E-1 自动满足:additive 只增不删)
    → audit 记录
  else:
    丢弃
```

### 4.3 边界

| 维度 | 决策 |
|---|---|
| **只 additive 加 wikilink** | 不改 body 文字,不升级 typed predicate(predicate 升级是语义判断,留 dream)|
| **alias 保留原文** | `[[B.md\|<原文 mention>]]`;原文一字不改 |
| **写入位置** | mention 第一次出现处加;后续保持原文(防 wikilink 满文) |
| **不动 anchor** | 与 dream 一致 |
| **守恒** | E-1 天然满足(纯增) |
| **rollback** | 误链发生时,人 / agent 直接编辑 body 删除 wikilink 即可;reinforce 不维护"我加过哪些"audit log(每次动作进 `audit/<date>/consolidate_actions.md`)|

### 4.4 reinforce 与 dream 的边界

dream 写入时 LLM 应已尽力召回相关节点 + 加 wikilink。reinforce 是**周期性兜底** —— 写入瞬间漏的、术语后才一致的、被 split 拆出来后才相关的,在 reinforce batch 里被检出。

这不违反"consolidate 不修 dream 漏的"立场 —— **dream 漏的 wikilink 在巩固阶段补,是合法工作**(它的依据是 dream 单点视角永远做不到的"周期统计 + 全局视角");**dream 漏的语义抽象在巩固阶段不补**(那是 dream 的语义判断,consolidate 不重做)。

---

# Part B — 独立工作

P0-P3 围绕 community,这两个动作独立运行。

## 5. split(过载分化:inline 写后)

**类比**:海马表征过密 → 分化新单元。

**目的**:节点 body 累积过长 / 主题离散后,拆成 parent overview + N children,保持单节点"一个原子语义单元"的粒度。

### 5.1 触发模型(写后立即,inline)

split 是 5 大动作中**唯一 inline** 的 —— 跟 dream 写入流强耦合,不走 weekly batch:

```
dream / split 写 body 成功 (CAS 通过)
  └─ if len(body) > T_token (default 2000):
        └─ LLM 判离散度
            └─ if is_overloaded:
                  └─ enqueue split job (FIFO, CAS-protected)
  └─ return (不阻塞 dream)
```

理由:节点过载是**写入瞬间的本地信号**(token + 离散度),延后无价值;反应即时。

### 5.2 split 动作

```
LLM 看 parent body:
  - 拆成 1 个 parent overview body + N 个 children body
  - 每个 child 自带 [[parent]] 反向链接
  - inbound 不动 (F-10)
  ↓
机械 outbound 守恒校验 (E-2):
  (parent_new ∪ ∪children_outbound) ⊇ parent_old
  失败 → LLM 重试 → 二次失败拒写 + audit
  ↓
事务式 CAS 写入: parent body 改写 + N 个新 children 文件创建
  ↓
audit + cooldown 设置 (parent + children 进 cooldown,与 merge 互锁)
```

### 5.3 split 与 abstract 的协同(关键)

| | 起源 | 方向 | 触发 |
|---|---|---|---|
| split overview | 单节点过载分化 | 自上而下(一拆多)| inline 写后 D3 |
| abstract summary | 多节点抽象凝聚 | 自下而上(多归一)| weekly batch + 稳定度阈值 |

**协同**:split 产出的 overview 节点会被 §2.5 的"已有 hub"判据识别,abstract 不重复创建。两者互补,不冲突。

---

## 6. archive(时效衰减:让长期不激活的节点淡出)

**类比**:突触代谢稳态 —— 长期不用的连接被减弱,但不删除。

**目的**:让 retrieve 默认排除"已不活跃"的节点,提升信噪比;不删 vault 文件,保持可逆。

### 6.1 recency_score:连续衰减信号

```
recency_score(node) =
  exp(-(now - last_update) / τ_update)        # 时间衰减
  × (1 + log(1 + last_hit_count_30d))         # 活跃度增强
  × (1 + log(1 + inbound_count) / SCALE)      # 中心性 cushion(避免 hub 被冷藏)
```

| 参数 | 默认 | 含义 |
|---|---|---|
| τ_update | 60 days | 时间衰减常数 |
| SCALE | 10 | 中心性 cushion 缩放 |

输出:`meta/recency.json`,每节点 0.0~1.0 连续值。

### 6.2 archived 派生快照

archived 是 recency_score 的二元化派生:

```
archived = {node | recency_score(node) < 0.15}
```

输出:`meta/archived.json`,recall 默认过滤这个列表。

### 6.3 解冻

任何动作触及节点 → 自动从 archived 移除:
- retrieve 命中(写 access_log)
- dream update 触及
- merge / reinforce 触及

下次 batch 时 recency_score 重算自然超过阈值。

### 6.4 daily 节奏

archive 是唯一不需要 community detection 的动作 → 节奏可以更快(daily batch),让冷启动后第二天就能影响 recall。

```
daily batch:
  1. 读 access_log (retrieve / dream / consolidate 钩子记录的命中事件)
  2. 重算 recency_score for all digest nodes
  3. 输出 meta/recency.json
  4. 阈值过滤 → meta/archived.json
```

---

# Part C — 共享基础设施

## 7. F-invariants 松绑与守恒规则

旧 F-invariants(`auto_dream_design.md` §4.3)在"vault 只读"立场下定义,新立场要松绑。但松绑不是"自由改",是用**动作级守恒规则**换"一刀切禁令"。

### 7.1 F-invariants 修订

| # | 旧约束 | 新立场 |
|---|---|---|
| **F-1** | 0 文件移动 | **改为**:"非 consolidate 动作不移动文件";merge 删除 loser 文件是**合法移动**(逻辑上等价 retarget) |
| **F-2** | 改正文限定 subject | **改为**:"dream / split / reinforce 改 subject body;merge 在受控算子内可改 inbound 节点 body";其它阶段(recall)绝不改 |
| **F-3** | maintainer 只做 split | **作废** —— consolidate 5 大动作合法 |
| **F-10** | inbound 不动 | **改为**:"split 时 inbound 不动";merge 必须 retarget inbound(机械算子) |
| **F-11** | wikilink 是 body 一部分 | **保留** —— 没有"独立的边"基础设施 |

### 7.2 动作级守恒规则矩阵

| 动作 | 置信度门槛 | 守恒规则 |
|---|---|---|
| **abstract** | community 节点 ≥ 5 + 稳定度 ≥ 0.7 + 活跃度 ≥ 3 + 多样性 ≥ 0.5 + 无 hub | 出边 ⊇ "关键支柱"列表 + 出边 ⊇ source 节点 60%(机械)|
| **merge** | LLM 多数票 ≥ 2/3 + similarity ≥ 0.6 + body overlap ≥ 0.5 | 信息守恒(merged.body ⊇ A ∪ B)+ 出边 ⊇ A.out ∪ B.out + inbound 全 retarget(机械)|
| **reinforce** | LLM 单票 ≥ 0.85 + 同 community + mention 锚点存在 + embedding ≥ 0.7 | E-1 天然(additive)|
| **archive** | recency_score < 0.15 | 软标记,无破坏性 |
| **split** | token > T + LLM 判离散 | E-2(parent ∪ children ⊇ parent_old)+ inbound 不动 |

---

## 8. cooldown 与防循环

5 大动作之间的潜在循环:

```
A merge B → AB body 长 → split AB 回 A' + B' → 又 merge → ...
```

防御:

| 互锁对 | 窗口 | 实现 |
|---|---|---|
| **split → merge** | 2 weeks | 刚 split 出的兄弟节点不参与 merge 候选 |
| **merge → split** | 2 weeks | 刚 merge 的节点不参与 split 评估(D3 检测时跳过)|
| **merge → merge**(同对反复) | 12 weeks | 同一 path 12 周内被 merge 又被识别为新 merge 候选 → audit 警报,人介入 |
| **abstract → merge**(同主题反复 abstract) | 4 weeks | 刚 abstract 出的 hub 节点 4 周内不参与 merge 候选 |

cooldown 状态外置 `meta/cooldowns.json`,不污染 vault。

---

## 9. CAS 写入协议(共享基础设施)

CAS 是 dream(`auto_dream_design.md` §4.2)、split / merge / reinforce / abstract(本文档)**多方共用**的 vault 写入协议。归本文档因 consolidate 是写入主战场。

archive 不写 vault → 不走 CAS;它写 `meta/`,各任务的 atomic write(write-temp + rename)即可。

### 9.1 协议

```
1. 读 + 记戳: read body → version_stamp = sha256(body) | mtime
2. 决策: LLM / 算法 → 产 planned new_body
3. CAS 写入: 重读 body 比 version_stamp
   - 未变: 跑动作级守恒校验 → 通过 → atomic write (write-temp + rename) → done
   - 已变: 丢弃 planned new_body, 带最新 body 重走 step 1
4. 守恒校验失败: LLM 重试一次, 二次失败拒写 + audit
5. 重做次数上限: 3 次 → 跳过候选 + audit log
```

### 9.2 事务式 merge / split 写入

merge 涉及多文件写入(winner body + N 个 inbound retarget + loser 删除);split 涉及多文件创建(parent body + N children)。需要事务语义:

- 准备阶段:全部 planned new_body 写到 temp 区(带 version_stamp)
- 提交阶段:逐个 CAS 检查 + atomic write(write-temp + rename)
- 任一 CAS 失败 → 全部回滚(temp 区清理,已 rename 的恢复)

实现细节:可借 fs-level 事务库(如 `pyrsistent` 模式)或自实现 journal。M0 起步用最简的"先全部检查 → 再全部写入"两阶段,接受窗口期(检查到写入间)的极小并发风险。

### 9.3 create 路径 race

merge / abstract 都可能并发 create 同一 path → atomic create(`O_CREAT | O_EXCL`)只让一个赢;输者 EEXIST → 重走 step 1(此时大概率改判 update 或丢弃)。

### 9.4 不解决

- 跨进程并发(多 reme 实例同 vault)→ 不在 M0,需 fs lock(M1+)
- 高冲突 workload(同候选反复触发)→ 重做上限触发后 audit

---

## 10. D 健康检查(D1 / D10)

不属"巩固"主语义,但跟 consolidate 同节奏(周期 batch 顺手跑),归本文档:

| # | 信号 | 节奏 | 修复策略 |
|---|---|---|---|
| **D1** | 断链(wikilink → 不存在 path) | 写时 inline + weekly batch 巡检(双重保险)| 就地删 wikilink 或保留 alias 文本 → audit |
| **D10** | provenance 断裂(digest 反指的 daily/resource 不可达)| 同上 | I-不变量违反 → 严重告警 + 人介入 |

D1 / D10 不算 5 大动作之一(它们不解决"vault 演化问题",只检测异常)。但它们的修复(就地删 wikilink)需要走 CAS,所以协议共享。

---

# Part D — 契约与实施

## 11. 维护 → 检索契约

5 大动作产物给 retrieve 消费(详细 retrieve 逻辑见 `auto_recall_design.md`):

| 产物 | 路径 | 写入者 | 读取者 | 缺失行为 |
|---|---|---|---|---|
| **vault 节点变化** | `digest/**.md` | merge / split / reinforce / abstract | recall(图遍历 / 命中) | — |
| **communities** | `meta/communities.json` | community detection | recall + abstract / merge / reinforce | 不做同社区 boost / 三个动作跳过 |
| **community changes** | `meta/community_changes.json` | community detection | abstract 决策 | abstract 跳过(无稳定度判据)|
| **recency** | `meta/recency.json` | archive daily batch | recall | recency_factor = 1.0 |
| **archived** | `meta/archived.json` | archive daily batch | recall(默认过滤)| 不过滤 |
| **cooldowns** | `meta/cooldowns.json` | split / merge | consolidate 内部 | 无防御循环 |
| **access_log** | `meta/access_log.json` | recall(写命中) + archive(聚合) | archive(读 recency) | recency 不衰减 |
| **dups uncertain** | `audit/<date>/dups_uncertain.md` | merge | 人 / agent | — |
| **consolidate actions** | `audit/<date>/consolidate_actions.md` | 全部 5 动作 | 审计 | — |
| **D1 / D10 健康** | `audit/<date>/health_*.md` | inline check + weekly | 人 / agent | — |

**契约稳定性**:`meta/*.json` 都带 `version` + `computed_at`;recall 启动时校验 version,不兼容则降级。

---

## 12. 与 dream 模型的引用关系

本文档松绑了部分 F-invariants(§7),但仍在 dream 定义的底层模型上工作:

| 引用 | 来源 |
|---|---|
| wikilink 基础语法 | `auto_dream_design.md` §3 |
| 节点 / 边模型 | `auto_dream_design.md` §4 / §2 / §3 |
| F-invariants 原始定义 | `auto_dream_design.md` §4.3(本文档 §7 修订)|
| 边守恒 E-1 / E-2 / E-3 | `auto_dream_design.md` §4.4 |
| 路径即 ID / rename | `auto_dream_design.md` §2 |
| anchor 不引入 | `auto_dream_design.md` §3 |
| provenance 载体形态 | `auto_dream_design.md` §4.2 |
| dream 写入路径 | `auto_dream_design.md` §4.2 |

---

## 13. 下一步(M0 → M1.1 → M1.2 → M1.3 → M2)

实现进入 `reme4/steps/consolidate/` 时,本文档与 `auto_dream_design.md` / `auto_cognition_design.md`(顶层)/ `auto_recall_design.md` 共同作为契约依据。

### M0:基础设施 + 完全可逆动作

- ✅ split inline 触发 + LLM 离散度判 + E-2 守恒(基础部分)
- ⏳ **community detection weekly batch**(Leiden via `igraph`)+ `meta/communities.json` + `meta/community_changes.json`
- ⏳ **archive daily batch** + recency_score + access_log 收集
- ⏳ CAS 写入框架 + version_stamp + EEXIST race + 重做上限 + audit
- ⏳ D1 / D10 写时 inline 检测 + weekly 巡检

### M1.1:abstract(P1,additive 最低风险)

- ⏳ abstract 候选挖掘(community 大小 + 稳定度 + 活跃度 + 多样性 + 无 hub 五重判据)
- ⏳ abstract LLM prompt(三段输出 + 长度限制 1500 token)
- ⏳ grounding 守恒校验(出边 ⊇ 关键支柱 + 出边 ⊇ source 60%)
- ⏳ "已有 hub" 结构化判据(outbound 覆盖度 ≥ 60%)
- ⏳ **关键验证点**:实测 split parent 是否被识别为 hub

### M1.2:merge(P2,lossy 高门槛)

- ⏳ 候选挖掘(community 内 description 相似 + body 重合 + cooldown 检查)
- ⏳ 多数票框架(N=3 LLM,2/3 通过)
- ⏳ merge prompt(B 方案:"只合并不精化")
- ⏳ inbound retarget 机械算子(扫所有 `[[loser.md]]` → `[[winner.md]]`,alias / predicate 保留)
- ⏳ 事务式多文件 CAS 写入
- ⏳ 灰色地带报告(`audit/<date>/dups_uncertain.md`)
- ⏳ cooldown 框架(`meta/cooldowns.json` + 各动作互锁)

### M1.3:reinforce(P3,价值最低,可缓做)

- ⏳ 候选挖掘(三层过滤:mention + embedding + 同 community)
- ⏳ 单票决策(门槛 0.85)
- ⏳ additive wikilink 写入(alias 保留原文)

### M2+:演进

- ⏳ 多层级 community(L2 super-community)+ L2 abstract
- ⏳ delete(永久删除 vault 文件)—— 视 dogfooding 效果决定是否开启
- ⏳ predicate upgrade(typed link reinforce —— 当前 reinforce 只 additive 加无谓词)
- ⏳ PageRank 替代 simple inbound count(若 retrieve 质量瓶颈在中心性)
- ⏳ 跨进程并发(fs lock 支持多 reme 实例同 vault)
- ⏳ Leiden 边权重(按 predicate 类型加权)
