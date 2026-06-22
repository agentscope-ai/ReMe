# auto-recall 设计(Stage 3 检索:信号融合 + 召回增强)

> 本文档:reme 中 **auto-cognition 三阶段** 的 **Stage 3 — 检索阶段** 实现。覆盖 query 到来时如何把 workspace 一等公民信号(wikilink 图 / frontmatter)与维护阶段产出信号(centrality / community / recency / archived)融合,生成最终召回。
>
> 配套阅读:
> - `auto_cognition_design.md`:三阶段顶层思想(本文档是 Stage 3)
> - `auto_dream_design.md`:Stage 1 写入 / 节点 + 边模型
> - `auto_consolidate_design.md`:Stage 2 维护 —— **本文档消费它产出的所有 `meta/*.json`**
> - `structure.md` §4(retrieve 三种问法)/ §7.4(为什么没有 retriever 模块)
> - `reme/steps/index/search.py` / `traverse.py`:现有原子实现
>
> **核心立场**:
> - retrieve **不引入新 L4 模块**(`structure.md` ✗-15)—— 三种问法各自由 L3 原子工具(`list_step` / `search_step` / `traverse_step`)直接覆盖
> - 本文档增强**集中在 `search_step` 内部**:把维护信号融入打分 / 排序 / 过滤;`traverse_step` 仅做小幅参数扩展
> - retrieve **只读 workspace,不写 body / 不写 frontmatter**;唯一写入是 `meta/access_log.json`(命中计数,供下次 recency 计算)

---

## 0. 问题陈述

`structure.md` §4 已规定 retrieve 三种问法(state / semantic / topological)正交分立(R-1)。本文档**只增强 semantic 问法**;state 问法已被 `list_step` 覆盖,topological 问法已被 `traverse_step` 覆盖。

semantic 问法当前在 `reme/steps/index/search.py` 实现:

| 已就绪 | 缺口 |
|---|---|
| ✅ vector + keyword 并行召回 | ❌ 节点中心性加权(高权威节点不被 boost) |
| ✅ RRF fusion(vector_weight=0.7) | ❌ 同社区 boost(`meta/communities.json` 未消费) |
| ✅ 一跳 expand_links(向前向后,max=10) | ❌ 时效衰减 / 冷藏过滤(`meta/access_log.json`、`meta/archived.json` 未消费) |
| ✅ min_score 过滤 + limit 截断 | ❌ 同 file 多 chunk 冗余(top-K 可全来自同节点) |
| ✅ chunk-level 命中(start_line / end_line) | ❌ 节点级 surface(frontmatter `name + description` 未与 chunk 命中合并展示) |
| ✅ 二跳 traverse 作为独立工具 | ❌ search 内 multi-hop expand(只一跳,跨术语关系到不了) |
| | ❌ query rewrite / multi-query(单一表达式漏召) |

**本文档的工作 = 设计这些缺口怎么填**,在 `search_step` / `traverse_step` 现有形态上增量。

---

## 1. 三种问法分立(继承 R-1)

```
┌─────────────┐  state 问 ──────► list_step + frontmatter filter
│   agent     │  semantic 问 ──► search_step  (本文档主要增强)
└─────────────┘  topological 问 ► traverse_step (小幅参数扩展)
```

| 问法 | 原子工具 | 本文档涉及 | 备注 |
|---|---|---|---|
| **state** | `list_step` / `daily_list_step` / `frontmatter_read_step` | 不涉及 | frontmatter 过滤无需维护信号 |
| **semantic** | `search_step` | **主战场**(§3-§7) | RRF fusion + 信号加权 + multi-hop + query rewrite |
| **topological** | `traverse_step` | 小幅(§8) | 起点选择可借助维护信号 |

**关键约束**(继承 `structure.md` ✗-8):**绝不合并三种问法成单一 read verb**。本文档增强 search_step,但不把 list / traverse 揉进 search;agent 按需各自调用。

---

## 2. 维护信号契约消费总览

`auto_consolidate_design.md` §11 列出维护产出。retrieve 端按以下方式读:

| 信号 | 来源 | 加载时机 | 缺失行为(降级) |
|---|---|---|---|
| **centrality** | `file_graph` 反向索引(实时) | search_step init 时引用 file_store | 总在线(file_graph 是核心组件) |
| **community** | `meta/communities.json` | search_step 启动 lazy load(LRU 缓存,文件 mtime 失效) | 缺失 → 不做同社区 boost |
| **recency** | `meta/access_log.json` | 同上 | 缺失 → recency_factor = 1.0 |
| **archived** | `meta/archived.json` | 同上 | 缺失 → 不过滤,所有节点参与 |
| **wikilink 图** | workspace 自身(file_graph) | 实时 | 总在线 |
| **frontmatter** | workspace 自身(`name` / `description`) | chunk 已带 metadata | 总在线 |

**version 校验**:`meta/*.json` 加载时检查 `version` 字段,与本文档约定的 schema 版本不匹配 → 走"该信号缺失"降级,日志告警(不崩)。

**新鲜度**:每个信号文件的 `computed_at` 暴露给调用者(metadata 中带 `signals_freshness`),调用方知道当前权重基于多久前的快照。超过阈值(默认 14 days)→ logger.warning + 仍使用(避免维护偶尔失效就拒绝服务)。

---

## 3. semantic 问法增强:打分公式

**目标**:把维护信号融入 fused chunk 的最终 score,让排序兼顾"文本相关 + 节点权威 + 同社区 + 时效"。

### 3.1 当前打分(基线)

```
score = RRF_fused(vector_rank, keyword_rank, vector_weight=0.7)
```

仅文本相似度。

### 3.2 新打分公式

```
final_score = base_score
            × centrality_factor(path)
            × community_factor(path, query_seed_paths)
            × recency_factor(path)
```

| 因子 | 公式 | 默认参数 | 来源 |
|---|---|---|---|
| **base_score** | RRF 融合分(现状) | vector_weight=0.7 | search.py |
| **centrality_factor** | `1 + α · log(1 + inbound_count)` | α = 0.15 | file_graph 实时 |
| **community_factor** | 同 community 命中节点 → ×β,否则 1.0 | β = 1.20 | `meta/communities.json` |
| **recency_factor** | `exp(-Δt / τ)`,Δt = 距 last_hit_or_update | τ = 60 days | `meta/access_log.json` |

**为什么乘法而非加法**:
- 各因子量级不同(base_score ≤ 0.02,centrality 与 query 无关),加法需大量 normalization;乘法天然处理量级差
- 任一因子接近 0(极冷藏 / 极孤立)→ 整体压低,符合"弱信号一票否决"直觉
- 默认 α/β/τ 让 factor 落在 [0.5, 2.0] 区间,不会让 base_score 完全失声

**已排除**:LLM rerank。它是 query-time 多调一次 LLM,成本高,M0 不引入;留 M1+ 视 dogfooding 决定。

### 3.3 query_seed_paths 的角色

community_factor 需要"query 主关注的节点是哪些"才能判断同/异社区。做法:
1. RRF 融合后取 top-N(N=3)的 fused chunk 的 path 作 seed
2. 后续每个候选 chunk 的 path → 查它和任一 seed 是否同社区 → boost
3. 不需要 query 自身被映射到 community(query 是字符串,不在图里)

**边界**:N=3 是经验起点;N 太大会让"同社区"几乎等于"全召回"失去区分度。dogfooding 后调。

---

## 4. semantic 增强:节点级合并(unique_paths)

**问题(gap 5)**:fused 列表里 top-5 可能是同 file 的 5 个 chunk,信噪比退化。

**当前**:`expand_links` 已用 `unique_paths = list(dict.fromkeys(c.path for c in fused))`,但 fused 本身没去重,limit=5 仍可全是同节点。

**新方案**(节点级 dedupe + 节点级 surface):

```
fused (chunk-level) → group by path → 每组保留 top_chunks_per_path 个
                   → 每组追加节点 frontmatter (name + description) 作"节点级 surface"
                   → 再按节点 best_score 排序 → limit
```

| 参数 | 默认 | 含义 |
|---|---|---|
| `top_chunks_per_path` | 2 | 同节点最多保留多少 chunk |
| `surface_node` | true | 是否在每组前追加 frontmatter `name + description` |

**为什么**:
- 节点是 retrieve 的语义单位(`auto_dream_design.md` §2 路径即 ID),chunk 只是"展示窗口"
- frontmatter 是节点级摘要(name + description)—— 已是 dream 写入时认证过的信号,不召它浪费
- 同节点多 chunk 时,frontmatter + top-2 chunk 比 5 个 chunk 信息密度高

### 4.1 答案展示形态

```
========== digest/auth/jwt-rotation.md ==========
[node] JWT Key Rotation
       Process for rotating JWT signing keys without downtime.
[score=0.0241 centrality=2.1 community=1.2 recency=0.91]

---------- chunk @5-23 ----------
<chunk text>

---------- chunk @45-60 ----------
<chunk text>

[expansion] 1 inbound, 2 outbound (...)
```

**对照旧形态**:每个 chunk 独立成块,无节点级 surface,scores 散在 chunk 头。新形态以**节点为视觉单位**,人 / agent 看到的第一眼是"哪个节点中了",而非"哪段文字中了"。

---

## 5. semantic 增强:multi-hop expand

**问题(gap 4)**:当前 expand_links 只展一跳,跨术语关系("分布式锁" → 一跳到"租约机制",再一跳才到"心跳协议")到不了。

**新方案**:expand_links 支持 `depth` 参数;默认仍 1(保守),agent / 配置可调到 2。

| 参数 | 默认 | 限制 |
|---|---|---|
| `expand_depth` | 1 | 最大 3(避免组合爆炸) |
| `max_links_per_direction` | 10(现状)| 每跳每方向上限,深度不展开时限到当跳总数 |
| `expand_path_budget` | 30 | 总扩展节点数硬上限,优先深度优先(深度浅但条数少) |

**为什么默认仍 1**:
- 二跳延迟不可忽略(N × 10 × 10 = 100 候选 IO)
- agent 需要"再深一层"时显式调 `traverse_step(depth=2)` —— 三种问法分立(R-1)
- 默认深拉会让"语义召回"变成"图召回",违背 R-1

**何时调 2**:dogfooding 发现 workspace 节点平均出度低 / 跨术语关系频繁 → 调到 2(改 search_step 配置,不改协议)。

---

## 6. semantic 增强:query rewrite / multi-query

**问题(gap 6)**:用户 query "JWT 怎么轮换" 可能错过 body 写"密钥定期更换"的节点(术语不同)。

**方案矩阵**:

| 方案 | 成本 | 效果 |
|---|---|---|
| **(a) 不做** | 0 | 漏召部分跨术语 |
| **(b) embedding 多 query**(用同 LLM 生成 N 个表述) | LLM 调用 1 次(query → N 表述)+ N 次 vector_search | 中等 |
| **(c) BM25 同义词扩展**(用静态词表 / 嵌入式词表) | 0(若有词表) | 弱(中文场景词表缺) |
| **(d) HyDE**(LLM 生成假设答案 → 嵌入这个答案而非 query) | LLM 1 次 | 高,文献证实 |

**首版决策**:**(a) 不做**。理由:
- workspace 本身规模 M0 不大,推断增加召回但增 LLM cost 不划算
- 维护阶段的 community 聚类已部分弥补"跨术语关系"(同社区 boost)
- 真要做,优先 (d) HyDE,延 M1+ 再启,实施只需加一层 query 预处理

**契约预留**:search_step kwargs 加 `query_rewrite: str | None`(默认 None;非 None 则用此重写代替原 query 做 vector_search,keyword_search 仍用原 query)。SDK 层可调用 LLM 生成重写后传入,reme 核心不强加 LLM 依赖。

---

## 7. semantic 增强:archived 过滤

**问题**:长期未访问的旧节点应该默认排除。

**方案**:search_step kwargs 加 `include_archived: bool`,默认 false。

```
fused → drop where path in archived_set → 后续打分 / unique_paths
```

**何时绕过**:
- agent 显式 `include_archived=true`(找历史 / debug)
- query 命中节点本身在 archived → boost 推回(冷节点突然被命中,说明不是真冷)
  - **首版不做**,过滤即过滤;如有需要,M1+ 加"intent override"机制

**冷启动**(`meta/archived.json` 缺失)→ 不过滤,等同 `include_archived=true`。

---

## 8. topological 问法的小增强

`traverse_step` 当前完整:BFS / 多 seed / direction / depth / per-edge 输出。本文档不重构,仅:

### 8.1 起点选择借助维护信号(可选 hint)

agent 调用 traverse 时往往不知道"哪个节点是该主题的中心";维护阶段产出的 centrality 可作 hint:

| 用例 | 做法 |
|---|---|
| traverse 给定 seed | 不变,直接 BFS |
| traverse 给定主题字符串(SDK 上层语法糖) | 先 search_step 找 top-1 → 用其作 seed → traverse depth=2 |

**位置**:这个组合在 SDK 上层做,不进 traverse_step;reme 核心保留 traverse 原子形态。

### 8.2 traverse 输出消费 archived

traverse_step 当前不知道 archived 信号。改造:加 `exclude_archived: bool` kwarg 默认 false(traverse 默认不过滤,因为它是图问法,过滤会破坏图视角)。SDK / agent 可显式开启。

---

## 9. retrieve 写访问日志(唯一对外写入)

**问题**:`meta/access_log.json` 的 `last_read` / `last_hit_count_30d` 谁写?

**约定**:retrieve 命中节点 → 异步 append 到访问日志缓冲区;由 maintain daily batch 聚合写入 `meta/access_log.json`。

| 路径 | 实现 |
|---|---|
| **同步写**(每 query) | retrieve 把命中 path 写入内存 ring buffer(进程级)|
| **异步落盘** | 进程退出 / 维护 daily batch / 周期 flush(默认 10 min)|
| **聚合** | maintain 在 daily access_log 重算时:读 ring buffer + 上一份 access_log → 合并写新版 |

**幂等**:同 query 多次重读同节点不应放大 last_hit_count;ring buffer 按 (path, day) 去重,每天每节点最多记一次"被读"。

**降级**:ring buffer 写失败 / flush 失败 → 不影响 retrieve 返回,只是日志少一条;recency 信号略迟。

---

## 10. 不变量 / 边界

| # | 约束 | 含义 |
|---|---|---|
| **R-1**(继承)| 三种问法分立 | 不合并 list / search / traverse 成单一 verb |
| **R-2**(继承)| 默认 `digest > daily > resource`,可覆盖 | search_step 通过 `search_filter` 支持限层 |
| **R-3**(继承)| 拓扑问与层无关 | traverse 跨三层(I-4) |
| **R-4**(继承)| Provenance 默认 lazy | retrieve 不自动 traverse(R-4);expand_links 是性能优化非语义展开 |
| **Re-1**(本文档)| retrieve 不引入 L4 模块 | 增强限定在原子 step 内部 |
| **Re-2**(本文档)| retrieve 只读 workspace | 不改 body / frontmatter / 文件位置 |
| **Re-3**(本文档)| retrieve 唯一对外写入是 `meta/access_log.json` | 通过 ring buffer + maintain 聚合,不直接写 |
| **Re-4**(本文档)| 任一维护信号缺失 → 降级不崩 | `meta/*.json` 缺 → 跳过对应因子,系统始终可用 |
| **Re-5**(本文档)| version 不兼容 → 降级 + warning | 不阻断 retrieve |

---

## 11. 与其它文档的引用关系

| 引用 | 来源 |
|---|---|
| 三种问法 / R-1..R-5 | `structure.md` §4 |
| 没有 retriever 模块 | `structure.md` §7.4 |
| 节点 / 边 / wikilink 模型 | `auto_dream_design.md` §2 / §3 |
| 维护信号契约 | `auto_consolidate_design.md` §11 |
| centrality / community / recency / archived 输出 | `auto_consolidate_design.md` §3-§5 |
| 路径即 ID | `auto_dream_design.md` §2 |

---

## 12. 下一步

实现进入 `reme/steps/index/` 时,本文档与 `auto_cognition_design.md`(顶层)/ `auto_dream_design.md` / `auto_consolidate_design.md` 共同作为契约依据。

**search_step 增强(§3-§7)**:
- ⏳ **打分公式**:加 centrality_factor / community_factor / recency_factor;config 化 α / β / τ(§3)
- ⏳ **节点级合并 + surface**:group-by-path + frontmatter surface + top_chunks_per_path(§4)
- ⏳ **multi-hop expand**:`expand_links` 支持 depth 参数,加 `expand_path_budget` 硬上限(§5)
- ⏳ **query_rewrite kwarg**:契约预留,reme 核心不强加 LLM(§6)
- ⏳ **archived 过滤**:`include_archived` kwarg,默认 false(§7)

**traverse_step 增强(§8)**:
- ⏳ **`exclude_archived` kwarg**(默认 false)

**信号加载基础设施(§2)**:
- ⏳ **`meta/*.json` lazy loader + LRU 缓存 + mtime 失效**
- ⏳ **version 校验 + 降级路径 + warning logger**
- ⏳ **signals_freshness metadata 暴露**

**access log 写入路径(§9)**:
- ⏳ **进程级 ring buffer**(命中 path 异步 append)
- ⏳ **周期 flush + (path, day) 幂等**
- ⏳ **maintain daily 聚合接口**(读 ring → 合并旧 access_log → 写新版)

**性能与回归**:
- ⏳ **基准测试**:打分公式启用前后的 召回 P@5 / MRR(用合成 workspace + ground-truth query)
- ⏳ **延迟监控**:维护信号读取 + multi-hop expand 的 p50 / p95
