# auto-dream 设计(桶 / 节点 / 边 / 演化)

> 本文档:digest 沉淀层的**桶**(物理布局)/ **节点**(原子单元)/ **边**(wikilink)/ **演化**(dream create_or_update;split 归 maintain)。
>
> 配套阅读:
> - `structure.md` §1.2(数据视角)/ §2(三层存储)/ §3.5(digest 动作)
> - `auto_memory_design.md`:daily 实时事件 = dream 的入流之一
> - `auto_consolidate_design.md`:M split / D 检测 / CAS 写入协议(dream 模型的运行时实现)
> - `auto_cognition_design.md`:auto-cognition 三阶段顶层思想 —— dream 是其 Stage 1(写入阶段)的实现
>
> **核心**:digest = **浅桶(shallow bucket)+ flat .md** + **一张图(节点 + 边)**;dream 定义模型与主流程(create_or_update),maintain 负责 split / 写入运行时。
>
> **关键收敛**:digest 不分"逻辑层"。所有 .md 文件都是同一种节点,内容决定它扮演什么角色(主题概览 / 概念定义 / 方法描述 / 实体记录 ...)。"主题"从图中涌现,不是结构性宣告。

---

## 0. 问题陈述

digest 是 agent 长期记忆的"组织化沉淀"层,与三层架构的另两层职责互补:

| 层 | 组织主轴 | 形态 |
|---|---|---|
| resource/ | 时间(`<date>/<name>`) | 外部原始资料,不可变 |
| daily/ | 时间 + 任务(`<date>/<event-slug>/`) | agent 任务过程,半可变 |
| **digest/** | **语义** | **跨任务知识,可重组** |

dream 设计回答四个问题:**桶**怎么布局 / **节点**长什么样 / **边**怎么连 / **演化**谁负责怎么做。

---

## 1. 桶(物理布局)

| 维度 | 决策 |
|---|---|
| **物理几何** | `digest/<bucket>/<slug>.md`;**浅桶一层**(顶多两层),桶内 flat |
| **bucket 角色** | **仅承担物理归档 + OS-level 浏览锚点**;不承担语义本体角色 —— 主题由图中节点表达 |
| **bucket 集合** | **代码内 hard-coded**(`reme/steps/evolve/dream.py` 的 `BUCKETS` 常量),不通过配置外置,不由 dreamer / maintainer 动态生成 —— 三桶设定是 dream 模型本身的一部分(Phase 2 prompt 按 bucket 专化),不是可调参数 |
| **集合视图** | 桶名内嵌在 prompt 中(extract 阶段三桶判别启发 + 三份独立 integrate prompt);不再生成独立 `_buckets.md` 视图 |
| **初始化** | opinionated **三桶**,按"答什么问 + 谁在问"划分:`procedure`(答"怎么做 X" —— 步骤 / 方法 / runbook)/ `personal`(答"X 是谁 / 喜欢什么 / 不要做什么" —— 用户 / 团队 specific 身份 + 偏好)/ `wiki`(答"X 是什么 / 发生了什么 / 决策依据是什么" —— 通用知识 / 定义 / 原则 / 观察 / 决策先例;**也是默认兜底**) |
| **bucket 主页** | 不强制存在;split 累积出层级时 parent 节点天然成为浏览主页(中心性涌现,非架构必需) |
| **新节点归属** | bucket 由 **Phase 1** 在 unit 级别分配(写进 `MemoryUnit.bucket`),Phase 2 据此分发到对应 bucket 的专用 prompt;LLM 不能造新桶 |
| **未归类节点** | Phase 1 找不到更明确归属时强制归入 `wiki` —— 它就是默认兜底,不是失败状态 |
| **跨桶 move** | F-1 已禁止;若必须做(人工介入修错桶),走一次 `wikilink_handler.retarget_links(old, new)` |

**`wiki` 兜底桶**:

| 维度 | 内容 |
|---|---|
| **语义** | "通用知识 / 默认归属" —— `wiki` 在三桶中 scope 最广(定义 / 原则 / 观察 / 决策先例),Phase 1 没有更明确归属(不属于 `procedure` 的可执行流程,也不属于 `personal` 的用户 specific 偏好)时归入此桶;**是合法常态,不是故障状态** |
| **路径** | `digest/wiki/<slug>.md`,与其它 bucket 完全等同;节点演化与其它桶一致 |
| **错桶后续** | 不主动跨桶 move;若严重,人工 mv + `retarget_links(old, new)` |

**为什么 `wiki` 兜底,而不是另设 `unknown`**:三桶设计中 `procedure` / `personal` 都有明确语义边界,剩下的"X 是什么 / 决策依据 / 一般原则"自然落在通用知识那一边 —— 这恰好就是 `wiki` 的本职。再设独立 `unknown` 会出现两类语义重叠的兜底(`wiki` 的"通用知识" vs `unknown` 的"分类未定"),反倒让 LLM 在 Phase 1 多一道无意义的犹豫。`wiki` 节点本身就是合法常态,不需要后续清理。

**为什么是浅桶而不是深树**:
- 物理浏览有"主题轮廓"(打开 `digest/wiki/` 能看到这一族节点),不像纯 flat 那样毫无锚点
- 节点不被深路径绑死("属 wiki/auth 还是 wiki/session"这种归属焦虑被消解 —— 一个节点可以同时被多个主题通过 wikilink 引用)
- F-1(0 文件移动)+ 平铺后,深树的核心收益(子树重组)消失,只剩深路径维护负担
- **固定三桶的关键意义**:LLM 在 dream 桶决定时只做"分类"(三选一),不做"造类" —— 决策面坍缩,跨任务跨时间稳定;不会出现 "knowledge" / "wiki" / "concepts" 三个语义重叠的桶共存。三桶覆盖 personal-knowledge 的核心切片(做什么 / 谁喜欢什么 / 知识本身),进一步细分由桶内 wikilink 图自然涌现

**已排除**:动态扩桶 / 拒绝写入(候选丢失)/ 强行选最近似专属桶(本体污染) / 把 bucket 数推回 6+(决策面失控)。

---

## 2. 节点

| 维度 | 决策 |
|---|---|
| **粒度** | atomic;一个 .md 文件 = 一个原子单元(概念 / 方法 / 实体 / 案例 / 原则 / 主题概览)|
| **节点角色** | **由 body 内容决定,不由 frontmatter 类型标记**;同一节点扮演"主题概览"还是"具体方法",看它的 body 写了什么 |
| **身份(ID)** | **workspace-relative 路径(含 `.md`)即节点身份** —— `digest/auth/jwt-rotation.md` |
| **`name` frontmatter** | 文件名 basename(不含扩展名),与文件名同步 —— 检索 hint / 人读标签,**不当 ID 用** |
| **frontmatter 保留字段** | 只有 `name` + `description`(reme 核心保留)|
| **可选 `kind` 字段** | 例:concept / procedure / preference / observation / ...;**消费层 schema 提示**,reme 核心透明,不读它做结构决策。与 bucket 是不同概念 —— bucket 决定物理归档(三桶)+ Phase 2 prompt 走哪份;`kind` 是更细粒度的 frontmatter 标签,留给消费层自由使用 |
| **文件名冲突** | 同 bucket 内文件名冲突 → 文件系统层断言(写入即拒);不需要独立检测信号 |
| **rename** | 一次 `wikilink_handler.retarget_links(old_path, new_path)`(机制现成);无 alias 表,无透明展开 |

**为什么 atomic + 路径即 ID**:
- **节点粒度 = retrieve 精度上限** —— semantic 检索召回 "一个原子单元" 远比召回 "一个 5000 字的主题文档" 信噪比高
- **wikilink 在 atomic 粒度才真有意义** —— `[[digest/auth/jwt-rotation.md]]` 指向"一个具体方法"比指向"auth 主题文档"精确一个数量级
- F-1 + 平铺 + 下层 immutable 后,slug abstraction 的核心价值(移动鲁棒性)蒸发;路径作 ID 与 `wikilink_handler.py` 默认形态完全对齐(*Recommended form: full path relative to the workspace with extension*)
- provenance wikilink 反指 daily/resource 本来就用路径,统一后整个 workspace 一种 wikilink 形态

**"主题概览节点"靠内容识别,不靠前缀 / kind**:`hub__` / `topic__` 前缀**不存在**;文件名自然命名(`auth-fundamentals.md` / `jwt-rotation.md`)。主题概览身份是图位置(中心性 / split parent)+ body 形态共同涌现。

---

## 3. 边

参考实现:`reme/utils/wikilink_handler.py` + `reme/schema/file_link.py`。

| 形态 | 写法 | 说明 |
|---|---|---|
| **基础** | `[[<workspace-path>.md]]` | literal,不隐含 `.md`,不自动短链补全 |
| **alias** | `[[path.md\|display-text]]` | rewrite 时 alias 保持 |
| **image** | `![[image.png]]` | 资源引用,不是知识边 |
| **可选谓词** | `predicate:: [[path.md]]`(行级)/ `[predicate:: [[path.md]]]`(内联) | Dataview 风格;谓词在 `[[]]` 外,`[[]]` 内只保留纯目标 |
| **谓词标识符** | `[A-Za-z][A-Za-z0-9_]*`(`is_a` / `extends` / `causes` / `references` ...) | 词表**开放**,任意标识符 |
| **未类型化合法** | 绝大多数 wikilink 不加 predicate;`predicate=None` 是默认 / 常态 | |
| **边唯一性键** | `(target_path, predicate)` 二元组 | 同源同标不同 predicate = 不同边 |
| **不引入 anchor** | digest 设计层不使用 `[[path.md#section]]` | `FileLink.target_anchor` schema 保留(供其它消费层),digest 层永远写 `None` |

**reme 核心对 predicate 的"透明"边界**(关键):
- 横向 link/ retrieve 中心性 —— 都**聚合所有 predicate** 算,不分桶
- 只有 edge 唯一性 / 反向索引会用到 predicate(否则 `[[A]]` 和 `is_a:: [[A]]` 会被当作同一条边互相覆盖)
- 消费层若要按 predicate 做更精细的推理(如"taxonomic 路径只走 `is_a` 边"),自己读 `FileLink.predicate` 即可

**与 `kind` 一致的立场**(与 [[reme_schema_layering]] 对齐):reme 核心**只有节点 + 边两种结构类型**;`kind` / `predicate` 都是内容标签,绝不参与"hub / topic / leaf"这类结构角色判断。

**为什么不引入 anchor**:LLM 想"指向具体子主题"时,**正确做法是让那个子主题升级为独立节点**(必要时通过 split),不在过载 parent 内部用 anchor 凑合。anchor 在 digest 层无语义;prompt 必须明确告知 LLM 写 wikilink 时不带 `#section`。

---

## 4. 演化

### 4.1 演化只做两件事

| op | 谁 | 何时 | 改什么 |
|---|---|---|---|
| **dream**(create_or_update) | dreamer(本文档 §4.2) | 入流(新材料进入) | 创建新节点 / update 已有节点 body(语义守恒重写;UPDATE 内分 **CORROBORATE / REFINE / CORRECT** 三种 flavor,详 §4.2.3) |
| **M split** | maintainer(`auto_consolidate_design.md` §1) | 节点过载(token / 主题离散度超阈值) | 把 parent body 拆成 parent overview + N children;parent 文件原地 |

> **关键观察**:"主题概览节点"不是一种 kind,也不是 maintainer 主动涌现的产物 —— 它是 split 的副产品(parent 节点天然成为该 cluster 的 overview,中心性自然高)。

显式排除:
- ❌ merge / dissolve / re-edge / unify —— 跨节点重组不做(同概念二次进入靠 dream update;错桶节点不主动 move)
- ❌ 完美归簇 —— F-5 留白,不确定就不动
- ❌ 实时一致 —— 异步 / eventual

### 4.2 dream(create_or_update)流程

**dream = dreamer 入流唯一改 body 的操作,且只改 subject node。**

#### 4.2.0 digest 是抽象记忆层

Digest 是 agent 长期记忆的**抽象层** —— 类比前额叶对认知的聚合。原始细节(数字、流程文本、谁说了什么)留在材料(daily / resource),digest 只承载细节淡忘后仍想调取的那一层:原则、模式、可作为先例的决策、认知要点。这一立场决定了 dream 流程的形态:**Phase 1 识别抽象,Phase 2 把抽象登记到 digest 节点**。

#### 4.2.1 两阶段流程

```
material 进入(daily / resource 选定 scope)
  │
  ▼
Phase 1 — extract (轻量)
  LLM 读材料 → 识别其中教导的"抽象"(原则 / 模式 / 先例)
  → 为每个 unit **分配 bucket**(procedure / personal / wiki)
  → 发出 ExtractedUnits 结构化输出 = K 个 sub-unit
    (每个: {name, bucket, summary})
  说明:多个支撑事实说明同一抽象 → 合并为同一 sub-unit
       (倾向少而精);Phase 1 是 gate ——
       无新抽象时发空列表,Phase 2 跳过整轮;
       bucket 由 Phase 1 一次性决定,Phase 2 不再回选
  │
  ▼ (Python 外循环,K 次)
Phase 2 — integrate (per sub-unit,**按 bucket 分发到独立 prompt**)
  │  system prompt = integrate_system_prompt_<unit.bucket>
  │  procedure / personal / wiki 三份独立 prompt,**不共用一套**
  │  sub-unit ↔ digest 节点 1:1;Phase 2 必写,无 SKIP 出口
  │
  ├─ RECALL: search(关键词 + 向量 + RRF) + traverse(对 top hit
  │  做图扩展,**跨 bucket**) → 候选路径集
  │
  ├─ HIT: frontmatter_read 廉价 triage → read 完整 body
  │  确认候选是否承载同一抽象 → hit 集合
  │
  ├─ 决策:
  │   ├─ hit 空    → CREATE 在 digest/<unit.bucket>/<slug>.md
  │   └─ hit 非空  → UPDATE 路径 (CORROBORATE / REFINE / CORRECT;
  │                  目标可在任意桶 —— 召回是跨桶的)
  │
  ▼
写入(canonical write 创建 / canonical edit 改正文)
  │
  ▼
agent 上报 IntegrateOutcome {action, target_path}
```

**两阶段 trade-off**:Phase 2 把完整材料发 LLM K 次(一次一 sub-unit),不做 summary loss;代价是 K 倍 prompt token。换来的是 Phase 1 只做"识别抽象 + 分类 bucket"两件事(粒度集中在一个 prompt),Phase 2 每次会话上下文干净、bucket-specific prompt 让推理聚焦于"这一桶要怎么写 / 怎么改"。

**Phase 2 的 bucket 专化**:三桶各有独立 system prompt,因为各桶的 body 形态、决策偏置不同 —— `procedure` 节点是 runbook 风(触发 / 步骤 / 前置 / 失败模式),`personal` 节点是规则风(rule + Why + How to apply),`wiki` 节点是百科风(定义 + 性质 + 关系)。共用一份通用 prompt 会让"应该写成什么样"的指导被稀释,bucket 信号靠一段 if-this-then-that 散文承载,效果劣于让每桶自带专属 prompt。

#### 4.2.2 召回 → 内化分类 → 决策 → 织突触(ReAct agent 一体完成)

Phase 2 是单个 ReAct agent 在一个 loop 内完成 4 件事 —— **不拆 stage,不引入外部机械步骤**,只通过 prompt 引导 agent 把 dedup 与 synapse 这两类判断都做透。当前默认 `search(limit=5)` 不够,prompt 已显式引导更深召回。

**4 步流程**(整段由 ReAct agent 自主组织调用):

| # | 步 | 关键动作 |
|---|---|---|
| 1 | **召回 —— 多角度宽召** | 显式 `limit=20-30` × 两轮 search(一次 hybrid,一次 `vector_weight=1.0` 纯语义)+ `traverse depth=2` 拓扑补充 |
| 2 | **内化分类** | `frontmatter_read` triage + 必要时 `read` body;对每个候选**内化打 label**(只在思考中分类,不输出):`same_abstraction` / `related` / `unrelated` |
| 3 | **决策** | 0 个 `same_abstraction` → CREATE;1 个 → UPDATE(选 flavor) |
| 4 | **织突触** | CREATE 或 UPDATE 都把所有 `related` 候选织入 body 作 `[[Y.md]]`;CREATE 一次性织全;UPDATE additive 加 wikilink |

**两类内化判断的本质**:

| 判断 | 服务 | 输出形态 |
|---|---|---|
| **同抽象?**(dedup)| 决定 CREATE / UPDATE | 0/1 个 target(决策面排他) |
| **相关?**(synapse)| 决定织哪些 wikilink | N 个 related 候选(决策面累加) |

两者是同一个 ReAct agent 在看完 candidates 后的**两层独立判断**,共享同一批召回结果,**不需要分两轮 LLM 调用**。

**召回**(对应 prompt step 1):dream 用专属的 `node_search`(`reme/steps/index/node_search.py`),**不**用通用 `search`,**也不用 `traverse`** —— 详 §4.2.2.1(traverse 是 retrieve-time 子图挖掘工具,跟 dream 写入场景错位)。

| 调用 | 找什么 |
|---|---|
| `node_search(query=<...>, limit=20-30)` | digest 内节点级 hybrid 召回(vector + BM25 RRF),返回 path + frontmatter |

**召回结果服务两类判断**:dedup(`same_abstraction` label,是否同抽象 → CREATE / UPDATE)和 synapse(`related` label,是否相关 → 织 wikilink)是 LLM 在**同一批 candidates** 上的两类内化 label。原"两轮 search(hybrid + vector_only)"是设计冗余 —— 同一批候选 LLM 自己能判 same/related/unrelated,模式切换无意义。**调用次数由 agent 自决**:一次通常够;若 unit 跨多个概念维度,agent 可发起多次不同 query 的召回,prompt 不强约束。

**HIT = `node_search` 返回 + read**:`node_search` 已内嵌返回每个 hit 的 frontmatter(`name + description`),agent 直接据此 triage,**不需要额外调 `frontmatter_read` 批量取 metadata**;仅对需要看 body 的少数候选用 `read`。**不可仅凭 frontmatter 决定 UPDATE**,body 才是判定依据。

##### 4.2.2.1 node_search vs 通用 search 的差别 + 为什么 dream 不用 traverse

**node_search vs 通用 search**:dream 的召回需求跟外部 agent 的 RAG 检索**结构性不同**,因此用专属 step 而非复用 `search`:

| 维度 | 通用 `search`(外部 agent)| `node_search`(dream Phase 2) |
|---|---|---|
| 用户 | 用户/外部 agent 的自然语言 query | dream 内部生成的 unit.summary |
| 结果粒度 | **chunk 级**(可能同一 node 多个 chunk)| **node 级**(同 path 聚合 max score)|
| 返回信息 | 完整 chunk text + scores | **path + name + description**(frontmatter 内嵌,无 body)|
| 范围 | 全 workspace(daily / resource / digest) | **digest-only**(dream 永远只在 digest 找候选)|
| expand_links | 默认 `True`(给 agent 更多上下文)| **永远 `False`**(synapse 找的就是未 link 的)|
| 默认 limit | 5 | **20**(dream 需要宽召覆盖 synapse)|

复用通用 `search` 会让 dream 拿到的候选**既粒度不对**(chunk 级,同 node 多次出现)**又信息冗余**(chunk text 不必要)**又被噪声污染**(daily / resource hits 永远不是 dream 的 UPDATE 候选)**又召回偏窄**(expand_links 把已 link 的拖回来,挤掉真正未 link 的 synapse 候选)。所以 dream 需要自己的 `node_search`。

**为什么 dream toolkit 不包含 traverse(或 dream_traverse)** —— traverse 是 **retrieve-time 子图挖掘工具**,跟 dream 写入场景**结构性错位**:

| 维度 | traverse 的本性(retrieve / RAG)| dream 的真实需求(写入)|
|---|---|---|
| 方向 | 从已知中心向外扩散 | 从外部新材料找 workspace 内相关候选 |
| 输入 | 已知种子节点 | 新材料的 unit.summary |
| 输出语义 | "X 的子图"(给读者上下文) | "X 应该 link 到哪些 Y" |
| 图遍历的角色 | 主操作 | 召回兜底(可有可无) |

dream 写新节点要回答"workspace 中谁跟我相关",这是**召回**问题(给 query 找相关),不是**遍历**问题(给中心找邻居)。**召回工具 = node_search;遍历工具 = traverse(留给 retrieve / 外部 agent 用)。dream 不需要遍历**。

(早期曾实现 `dream_traverse` 准备作为 dream toolkit 一员,后撤销 —— 实测拓扑遍历 vs vector 召回重叠率 ~95%,真正独特贡献 < 2%,且引入 LLM 调用 / 上下文 / 复杂度成本。详 git log。)

**node_search 参数极简**(`query / limit` 两个):**mode 不需要**(同一批候选服务双判断);**exclude_paths 不需要**(self 由 LLM 自己识别,frontmatter 内嵌让 agent 一眼看出"这就是我");**min_score 不需要**(RRF 分数范围 0~0.025,跟 cosine 0~1 量纲完全不同,召回深度由 `limit` 控制就够)。**调用次数 agent 自决**:prompt 不约束"必须一次",unit 跨多个概念维度时 agent 可多次召回。

**node_search 召回算法:weighted node-level RRF**(vector + BM25 hybrid):

- vector + BM25 各自独立召回 → 各自得到 chunk list(按各自 score 排序)
- 同 path 多 chunk 合并:取该 path 在两个 list 中的 max chunk score 位置作为 node rank
- RRF 融合:`score(path) = vector_weight × 1/(60 + rank_v) + (1-vector_weight) × 1/(60 + rank_k)`
- `vector_weight=0.7`(默认),vector 主导,BM25 作为兜底(覆盖专有名词 / 缩写等 embedding 可能 struggle 的字面 case)
- 输出 score 是 RRF 分(0~0.025 量级,不是 cosine);LLM 不依赖具体分数,内化判 same/related/unrelated

**reinforce 并入立场**(对照 `auto_consolidate_design.md` §4 标作废):reinforce 不再是独立的 consolidate 动作 —— 它就是 step 4 的"织突触"。新节点写入瞬间一次性建立关系,workspace 不维护"事后周期 batch 补 wikilink"的通道。F-2 自然守住 —— dream 只动新节点 body,不动其它节点。

**关键约束**(诚实承认):
- **写入即定型** —— 今天没织的 wikilink 以后没机会再织;workspace 单调演化
- **一次性 commit,无事后兜底** —— prompt 明示"宁可多织"(false positive 一眼能否决;false negative 永远沉默)
- **召回深度取决于 prompt 引导 + agent 配合** —— 不引入外部机械召回 step;prompt 已明示 `limit=20-30 × 两轮`,但仍是 ReAct agent 的开放执行
- **dedup 与 synapse 在一次 LLM 调用内完成** —— 不拆独立 stage,共享召回结果,内化分类是免费的

#### 4.2.3 UPDATE 三种 flavor

| flavor | 何时 | body 怎么动 |
|---|---|---|
| **CORROBORATE**(最常见)| 已有节点已覆盖此抽象,材料是又一个实例 | body 实质不变 —— 追加 `derived_from::` 溯源,可选强化措辞("似乎"→"确实") |
| **REFINE**(常见)| 已有节点覆盖了核心,但材料揭示新的范围 / 边界 / 维度 | 改相关片段使更精确,加新维度,加 `derived_from::`。正文在**精度**上长,不在**细节**上膨胀 |
| **CORRECT**(少见)| 材料与已有抽象矛盾 / 表明它被夸大 | 收紧到新旧证据都支持的窄形式,或内联标注 `> note: contradicted by [[...]]` 不仲裁。仍加溯源 |

三种都受 §4.4 E-1 强守恒约束(出边集合不能缩)。

#### 4.2.4 关键边界

- **Phase 1 是 gate + 分类器** —— "不值得记忆"在 Phase 1 过滤(空列表);此外 Phase 1 还为每个进入 Phase 2 的 unit 分配 bucket(procedure / personal / wiki),决定 Phase 2 走哪份专用 prompt;Phase 2 必然写,sub-unit 与 digest 节点 1:1
- **Phase 2 prompt 按 bucket 分发** —— `integrate_system_prompt_procedure` / `_personal` / `_wiki` 三份独立 system prompt,各自承载该桶的 body 形态指南与决策偏置,**不共用一份通用 prompt**
- **CREATE 写入桶 = Phase 1 分配的桶**;**UPDATE 目标可在任意桶**(召回跨桶,UPDATE 命中谁就写谁)
- **dream update 必须语义守恒** —— LLM 重写 body 时只能"融入"新内容,不能删除已有信息(只增不删 / 不改原意;冲突标注 `> 注:不同来源记载...`,不擅自仲裁);**当前实现下 E-1 强守恒是 prompt-only 自律**(canonical edit 不做机械 outbound diff;早期 `digest_edit` 子类的机械校验已在切到 canonical 工具时移除,详 §4.4)
- **Phase 2 用 canonical write / edit** —— 不再有 `digest_write_step` / `digest_edit_step` 子类;桶归位与边守恒都是 prompt-level 纪律
- **dream 不改其它节点正文**(F-2) —— 只动 subject
- **dreamer 不做事件级伞节点** —— 材料本身(daily / resource 文件)就是 fan-out 点,每个 sub-unit 的 `derived_from::` 让材料天然聚合到所有派生节点
- **0 出边节点合法**(没识别到合适邻居),后续 dream 进入时其它节点可以反向链回来 —— 不强求 LLM 一次性给全
- **dream 漏判去重**(同概念建成新节点)→ 不主动兜底,接受重复;若 workspace 累积明显重复,由 auto-consolidate 的 dups 检测周期 batch 产报告(`auto_consolidate_design.md` §3)
- **召回不做 bucket 粗筛** —— LLM 拥有完整跨桶视野,可识别"概念跨桶同抽象"(例如同一原则在 wiki 已有节点而 Phase 1 把新材料归入 personal,此时 UPDATE wiki 节点而非新建 personal 节点)
- **reinforce 已并入 dream synapse recall** —— 不存在独立的 reinforce 动作或周期 batch;突触构建(原 `auto_consolidate_design.md` §4 reinforce 的职责)在 dream Phase 2 synapse recall 阶段完成,新节点写入瞬间织全(详 §4.2.2)
- **workspace 不维护事后补 wikilink 通道** —— 上一条的直接推论;cognition §9.2 立场("关系建立在写入瞬间")在此自然守住

**provenance 写出**:
- 行文中自然带:"... 该模式最早出现在 [[daily/2026/05/15.md]] 的实践中"
- **强制 typed predicate `derived_from::`** —— body 必须织入至少一条 `derived_from:: [[daily/...]]` 或 `[[resource/...]]`,纯散文形式不会被未来的 update / 守恒比对识别为边,下次 update 时会消失
- LLM 直接做语义守恒重写(只增不删) —— 不走"首版 append 起步"的过渡路径

### 4.3 F-invariants(演化的硬约束)

| # | 约束 | 含义 |
|---|---|---|
| **F-1** | **0 文件移动** | dream / split 都不移动现有文件;split 创建的是**新文件**,parent 原地 |
| **F-2** | **改正文限定 subject** | dream update 改 subject body;M split 改 parent body + 创建 children body;**没有任何操作改"其它节点正文"** |
| **F-3** | **maintainer 只做 split** | 没有 summarize / merge / re-edge / link / unify / dissolve |
| **F-4** | **一次一个候选** | M split 一次拆一个;dream 一次处理一个原子单元(N 候选 = N 次 dream) |
| **F-5** | **不确定时不动** | dream 拿不准 create 还是 update → 倾向 create;split 拿不准 cluster → 不拆 |
| **F-7** | **多归属合法** | 一个节点可被多个引用,也可指向多个;**没有"单父"约束** |
| **F-10** | **inbound 目标节点不动** | 所有 inbound 是裸链 `[[<parent-path>.md]]`(digest 不引入 anchor);split 时全部保持,parent 路径未变即天然有效 |
| **F-11** | **wikilink 是 body 的一部分** | 不存在"独立的边";reme 核心机械算子只感知字符层,语义责任在 LLM(prompt 自律);split 写入路径仍带机械 outbound 校验,dream update 当前是 prompt-only(详 §4.4) |

### 4.4 边守恒(E-1 / E-2 / E-3)

**前提**:wikilink 是 body 的一部分(F-11)。"边"不是独立抽象 —— body 一变,边就跟着变。reme 核心**没有"修边"算子**;边的所有变化都是 body 文本编辑的副作用。语义层守恒由两条腿承担:**prompt 自律**(LLM 在 update 时被反复要求 only-add, not-delete)+ **必要时的机械校验**(下文区分了哪些保留、哪些已移除)。

| # | 类别 | 规则 | 谁负责 |
|---|---|---|---|
| **E-1** | dream update 节点出边(subject 自身) | **强守恒**:新 body 出边 ⊇ 原 body 出边(`(target, predicate)` 二元组,predicate 一并守住) | **当前实现:LLM(prompt)自律** —— canonical `edit` 不做机械 outbound diff,prompt 反复强调"never drop wikilinks the old span contained" |
| **E-2** | split parent 出边(parent 拆解) | `(parent_new ∪ ∪children_outbound) ⊇ parent_old` | LLM(split prompt)+ 机械(由 maintainer 在 split 写入路径上实施,见 `auto_consolidate_design.md`) |
| **E-3** | inbound wikilink `[[<parent-path>.md]]` | split 时**不动** —— 仍指 parent;后续 dream 进入若 LLM 觉得 child 粒度更合适,直接加新边到 child(F-10) | 不动 |

**E-1 实现取舍**:早期版本有专用 `digest_edit_step` 子类,在写入前对 body 做 outbound diff 比较,违反守恒时返回 `REJECT_CONSERVATION` 让 LLM 重试。在切到 canonical `edit` 工具(放弃 digest 子类)后,这道机械校验被移除 —— 守恒退化为 prompt-only 自律。trade-off:
- **失**:LLM 偶尔会在 REFINE / CORRECT 时无意丢弃 `derived_from::` 链;系统不再自动拒写
- **得**:Phase 2 工具与系统其它写入路径完全一致(write / edit 是 canonical job),没有 dream-private 写入语义;prompt 复杂度下降,工具表面更小
- **后续**:若 prompt-only 守恒在生产中被证伪(掉链率高),可在 canonical `edit` 上挂一个可选的 conservation 校验 hook(不再走子类化路径),由 dreamer 在调用前后各 read 一次做 diff;但当前不做

**强守恒(集合包含)而非等价**:`new ⊇ old` = 允许加新边(新关联),不允许减边(老内容不能丢);`new == old` 会拒绝任何新出边 → update 失去意义。

**predicate 守住** —— `[[A]]` ↔ `is_a:: [[A]]` 视为不同 key,升降级走显式 audit 路径,不走默认。重排 / 改 alias / 加新边都不被拦下(集合相同或只增)。

**provenance 不单列** —— 节点反指上游 daily/resource 的 wikilink 是 body 正文的一部分,跟其它 wikilink 走同一套 E-1 / E-2;reme 核心没有 provenance 专用算子。

**inbound anchor 这一类不存在** —— digest 不引入 anchor,所有 inbound 都是裸链,走 E-3 即可,无需机械 retarget 子流程。

---

## 5. 与其它层

| 上下游 | 关系 |
|---|---|
| ← **auto-memory**(daily) | dream 读 daily 作为入流;daily 写完即对 dream 可见 |
| ← **resource** | dream 读 resource 作为入流(只读,不写) |

**关键边界**:dream 不写 daily / resource(I-2 / I-3);只写 digest 节点 body(自身 subject)。dream 不感知下游 —— split / 链接增强 / 索引刷新 / rename 等由 `auto_consolidate_design.md` / `auto_cognition_design.md` / `update_store_index_loop` 各自负责。

---

## 6. 下一步

本文档覆盖 dream 模型(桶 / 节点 / 边 / 演化)。组织端实现清单(M split / D 检测 / CAS 框架)见 `auto_consolidate_design.md` §10。

- ✅ **dream step 实现** —— Phase 1 extract(识别抽象 + 分配 bucket)+ Phase 2 integrate(per sub-unit,**bucket-specific prompt 分发**;`reme/steps/evolve/dream.py` + `dream.yaml`,与 `auto_memory` 同级同形)
- ✅ **三桶 hard-coded** —— `procedure / personal / wiki`,`BUCKETS` 常量在 `dream.py` 顶部,Phase 1 通过 `MemoryUnit.bucket: Literal[...]` 由 Pydantic 强制约束
- ✅ **provenance prompt 规范** —— `derived_from:: [[daily/...]]` / `[[resource/...]]` 强制(三桶 prompt 各自重申)
- ❌ ~~**边守恒校验工具**~~ —— 早期 `digest_edit` 子类的 outbound diff 校验已随子类一并移除(切到 canonical `edit`);E-1 现由 prompt 自律,详 §4.4
- ❌ ~~**bucket 集合配置外置**~~ —— 撤销:三桶是 dream 模型本身的一部分,不做配置参数(`workspace.yaml` 不再承载 `digest.buckets`,`_buckets.md` 视图也不再生成)
- 🆕 **Phase 2 召回拆 dedup / synapse**(2026-06-02 沉淀,详 §4.2.2)—— 当前 prompt 共用一次 `search(limit=5)`,既不够 dedup 精度也不够 synapse 覆盖;落地:`dream.yaml` 6 处(en + zh × 3 buckets)Recall 段改写,加 synapse 模式说明 + 写入即定型纪律
- 🆕 **`file_store.default.embedding_model` 启用**(blocker)—— `default.yaml` 当前 `""`,synapse recall 用 vector_weight=1.0 模式必须开启;否则 `search` 退化为纯 BM25,dedup 也劣化
- 🆕 **reinforce 并入立场写入**(详 §4.2.2)—— 与 `auto_consolidate_design.md` §4 标作废同步;`hierarchical_summary.md` §13.2 Q4 标解决

实现进入 `reme/steps/evolve/` 时,本文档与 `auto_memory_design.md` / `auto_consolidate_design.md` / `auto_cognition_design.md` 共同作为契约依据。
