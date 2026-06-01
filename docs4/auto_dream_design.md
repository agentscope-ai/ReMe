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
| **bucket 集合** | **固定预定义**(`vault.yaml` 顶层 `digest.buckets:`),不由 dreamer / maintainer 动态生成 |
| **集合视图** | 自动生成 `digest/_buckets.md` 作为人/LLM 可读视图;dream 时读后者作 prompt context |
| **初始化** | opinionated default(6 桶,按"答什么问"划分):`concept`(答"X 是什么")/ `procedure`(答"怎么做 X")/ `entity`(答"X 是谁/哪个")/ `observation`(答"发生了什么")/ `preference`(答"X 喜欢怎样 / 别做什么";用户记忆主战场)/ **必带 `unknown`** —— 消费层可改桶名,但 `unknown` 不可删 |
| **bucket 主页** | 不强制存在;split 累积出层级时 parent 节点天然成为浏览主页(中心性涌现,非架构必需) |
| **新节点归属** | dream 只能从已有桶集合**挑选**;LLM 不能造新桶 |
| **未归类节点** | 统一落入兜底桶 `digest/unknown/`(详下) |
| **跨桶 move** | F-1 已禁止;若必须做(人工介入修错桶),走一次 `wikilink_handler.retarget_links(old, new)` |

**`unknown` 兜底桶**:

| 维度 | 内容 |
|---|---|
| **语义** | "分类未定 / 暂无专属归属" —— **合法常态,非故障状态**(LLM 没找到合适专属桶时的诚实表达,不是写入失败) |
| **路径** | `digest/unknown/<slug>.md`,与其它 bucket 完全等同;节点演化与其它桶一致 |
| **错桶后续** | 不主动跨桶 move;若严重,人工 mv + `retarget_links(old, new)` |

**为什么是 `unknown` 而不是 `general`**:`general` 听起来像在断言"这个概念真的属于通用类",实际上只是 LLM 没找到合适专属桶;`unknown` 诚实表达"分类未定",不强加伪类目。但 `unknown` 不是"待清理状态" —— 它是合法常态,节点在此演化(被 update / 被 wikilink 引 / 中心性增长)与其它桶完全等同;不会随时间被自动清空。

**为什么是浅桶而不是深树**:
- 物理浏览有"主题轮廓"(打开 `digest/auth/` 能看到这一族节点),不像纯 flat 那样毫无锚点
- 节点不被深路径绑死("属 auth/jwt 还是 auth/session"这种归属焦虑被消解 —— 一个节点可以同时被多个主题通过 wikilink 引用)
- F-1(0 文件移动)+ 平铺后,深树的核心收益(子树重组)消失,只剩深路径维护负担
- **固定集合的关键意义**:LLM 在 dream 桶决定时只做"分类",不做"造类" —— 决策面坍缩,跨任务跨时间稳定;不会出现 "auth-stuff" / "auth" / "authentication" 三个语义重叠的桶共存

**已排除**:动态扩桶 / 拒绝写入(候选丢失)/ 强行选最近似专属桶(本体污染)。

---

## 2. 节点

| 维度 | 决策 |
|---|---|
| **粒度** | atomic;一个 .md 文件 = 一个原子单元(概念 / 方法 / 实体 / 案例 / 原则 / 主题概览)|
| **节点角色** | **由 body 内容决定,不由 frontmatter 类型标记**;同一节点扮演"主题概览"还是"具体方法",看它的 body 写了什么 |
| **身份(ID)** | **vault-relative 路径(含 `.md`)即节点身份** —— `digest/auth/jwt-rotation.md` |
| **`name` frontmatter** | 文件名 basename(不含扩展名),与文件名同步 —— 检索 hint / 人读标签,**不当 ID 用** |
| **frontmatter 保留字段** | 只有 `name` + `description`(reme 核心保留)|
| **可选 `kind` 字段** | 例:concept / procedure / entity / observation / preference / ...;**消费层 schema 提示**,reme 核心透明,不读它做结构决策 |
| **文件名冲突** | 同 bucket 内文件名冲突 → 文件系统层断言(写入即拒);不需要独立检测信号 |
| **rename** | 一次 `wikilink_handler.retarget_links(old_path, new_path)`(机制现成);无 alias 表,无透明展开 |

**为什么 atomic + 路径即 ID**:
- **节点粒度 = retrieve 精度上限** —— semantic 检索召回 "一个原子单元" 远比召回 "一个 5000 字的主题文档" 信噪比高
- **wikilink 在 atomic 粒度才真有意义** —— `[[digest/auth/jwt-rotation.md]]` 指向"一个具体方法"比指向"auth 主题文档"精确一个数量级
- F-1 + 平铺 + 下层 immutable 后,slug abstraction 的核心价值(移动鲁棒性)蒸发;路径作 ID 与 `wikilink_handler.py` 默认形态完全对齐(*Recommended form: full path relative to the vault with extension*)
- provenance wikilink 反指 daily/resource 本来就用路径,统一后整个 vault 一种 wikilink 形态

**"主题概览节点"靠内容识别,不靠前缀 / kind**:`hub__` / `topic__` 前缀**不存在**;文件名自然命名(`auth-fundamentals.md` / `jwt-rotation.md`)。主题概览身份是图位置(中心性 / split parent)+ body 形态共同涌现。

---

## 3. 边

参考实现:`reme4/utils/wikilink_handler.py` + `reme4/schema/file_link.py`。

| 形态 | 写法 | 说明 |
|---|---|---|
| **基础** | `[[<vault-path>.md]]` | literal,不隐含 `.md`,不自动短链补全 |
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

**与 `kind` 一致的立场**(与 [[reme4_schema_layering]] 对齐):reme 核心**只有节点 + 边两种结构类型**;`kind` / `predicate` 都是内容标签,绝不参与"hub / topic / leaf"这类结构角色判断。

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
  → 发出 ExtractedUnits 结构化输出 = K 个 sub-unit
    (每个: {name, summary},summary 标注证据在材料的哪段)
  说明:多个支撑事实说明同一抽象 → 合并为同一 sub-unit
       (倾向少而精);Phase 1 是 gate ——
       无新抽象时发空列表,Phase 2 跳过整轮
  │
  ▼ (Python 外循环,K 次)
Phase 2 — integrate (per sub-unit,每次独立 ReAct 会话)
  │  sub-unit ↔ digest 节点 1:1;Phase 2 必写,无 SKIP 出口
  │
  ├─ RECALL: search(关键词 + 向量 + RRF) + traverse(对 top hit
  │  做图扩展,跨 bucket) → 候选路径集
  │
  ├─ HIT: frontmatter_read 廉价 triage → read 完整 body
  │  确认候选是否承载同一抽象 → hit 集合
  │
  ├─ 决策:
  │   ├─ hit 空    → CREATE 路径 (挑 bucket,写新节点)
  │   └─ hit 非空  → UPDATE 路径 (CORROBORATE / REFINE / CORRECT)
  │
  ▼
写入(digest_write 创建 / digest_edit 改正文,E-1 强守恒,§4.4)
  │
  ▼
agent 上报 IntegrateOutcome {action, target_path}
```

**两阶段 trade-off**:Phase 2 把完整材料发 LLM K 次(一次一 sub-unit),不做 summary loss;代价是 K 倍 prompt token。换来的是 Phase 1 只做"识别抽象"这一件事(粒度集中在一个 prompt),Phase 2 每次会话上下文干净、聚焦单一抽象的写决策。

#### 4.2.2 召回二段

**RECALL = search + traverse**:search 给关键词 + 向量 RRF 命中;只要 search 在 `digest/` 下返回任何 hit,就对 top hit 跑 `traverse depth=2 direction=both`。理由是 search 关键词导向,会漏掉用不同术语归档的语义相邻抽象,那些常常一跳之外。search 在 `digest/` 下完全无命中 → 无 traverse 起点 → 候选集为空 → 直接 CREATE。

**HIT = frontmatter_read + read**:渐进披露 —— 先 `frontmatter_read` 读 `name + description` 廉价 triage 淘汰明显无关候选,剩下的再 `read` 整 body。**不可仅凭 chunk 片段或 frontmatter 决定 UPDATE**,body 才是判定依据。

#### 4.2.3 UPDATE 三种 flavor

| flavor | 何时 | body 怎么动 |
|---|---|---|
| **CORROBORATE**(最常见)| 已有节点已覆盖此抽象,材料是又一个实例 | body 实质不变 —— 追加 `derived_from::` 溯源,可选强化措辞("似乎"→"确实") |
| **REFINE**(常见)| 已有节点覆盖了核心,但材料揭示新的范围 / 边界 / 维度 | 改相关片段使更精确,加新维度,加 `derived_from::`。正文在**精度**上长,不在**细节**上膨胀 |
| **CORRECT**(少见)| 材料与已有抽象矛盾 / 表明它被夸大 | 收紧到新旧证据都支持的窄形式,或内联标注 `> note: contradicted by [[...]]` 不仲裁。仍加溯源 |

三种都受 §4.4 E-1 强守恒约束(出边集合不能缩)。

#### 4.2.4 关键边界

- **Phase 1 是 gate** —— "不值得记忆"在 Phase 1 过滤(空列表);Phase 2 必然写,sub-unit 与 digest 节点 1:1
- **dream update 必须语义守恒** —— LLM 重写 body 时只能"融入"新内容,不能删除已有信息(只增不删 / 不改原意;冲突标注 `> 注:不同来源记载...`,不擅自仲裁);写入前机械校验出边强守恒(E-1,详 §4.4)
- **dream 不改其它节点正文**(F-2) —— 只动 subject
- **dreamer 不做事件级伞节点** —— 材料本身(daily / resource 文件)就是 fan-out 点,每个 sub-unit 的 `derived_from::` 让材料天然聚合到所有派生节点
- **0 出边节点合法**(没识别到合适邻居),后续 dream 进入时其它节点可以反向链回来 —— 不强求 LLM 一次性给全
- **dream 漏判去重**(同概念建成新节点)→ 不主动兜底,接受重复;若 vault 累积明显重复,由 auto-consolidate 的 dups 检测周期 batch 产报告(`auto_consolidate_design.md` §3)
- **召回不做 bucket 粗筛** —— LLM 拥有完整跨桶视野,可识别"概念错分到 unknown"或"跨桶同概念"

**provenance 写出**:
- 行文中自然带:"... 该模式最早出现在 [[daily/2026/05/15.md]] 的实践中"
- **强制 typed predicate `derived_from::`** —— body 必须织入至少一条 `derived_from:: [[daily/...]]` 或 `[[resource/...]]`,纯散文形式不被守恒校验视作边,下次 update 时会消失
- 不走"首版 append 起步"的过渡路径 —— digest_edit 自一开始就跑 E-1 强守恒,LLM 直接做语义守恒重写

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
| **F-11** | **wikilink 是 body 的一部分** | 不存在"独立的边";reme 核心机械算子只感知字符层,语义责任在 LLM(prompt)+ 守恒校验(outbound diff) |

### 4.4 边守恒(E-1 / E-2 / E-3)

**前提**:wikilink 是 body 的一部分(F-11)。"边"不是独立抽象 —— body 一变,边就跟着变。reme 核心**没有"修边"算子**;边的所有变化都是 body 文本编辑的副作用。但语义守恒不能放任 LLM:守恒责任在 prompt + 机械校验。

| # | 类别 | 规则 | 谁负责 |
|---|---|---|---|
| **E-1** | dream update 节点出边(subject 自身) | **强守恒**:新 body 出边 ⊇ 原 body 出边(`(target, predicate)` 二元组,predicate 一并守住) | LLM(prompt)+ 机械(outbound diff) |
| **E-2** | split parent 出边(parent 拆解) | `(parent_new ∪ ∪children_outbound) ⊇ parent_old` | LLM(split prompt)+ 机械 |
| **E-3** | inbound wikilink `[[<parent-path>.md]]` | split 时**不动** —— 仍指 parent;后续 dream 进入若 LLM 觉得 child 粒度更合适,直接加新边到 child(F-10) | 不动 |

**机械 outbound diff 校验**(E-1 / E-2)伪码:
```
write_subject_body(subject, new_body):
    old = extract_links(old_body)
    new = extract_links(new_body)
    if old - new:
        new_body = llm_retry_with_missing(old - new)
        if old - extract_links(new_body):
            raise ConservationViolation(...)  # 拒写 + audit
    write(subject, new_body)
```

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

- ✅ **dream step 实现** —— Phase 1 extract + Phase 2 integrate(per sub-unit)+ E-1 守恒校验(`reme4/steps/evolve/dream/`)
- ✅ **边守恒校验工具** —— `digest_edit` 的 outbound diff 比较器 + REJECT_CONSERVATION 重试 + 违规上报
- ✅ **provenance prompt 规范** —— `derived_from:: [[daily/...]]` / `[[resource/...]]` 强制
- ⏳ **bucket 集合配置外置** —— 当前 hardcoded 在 `digest_write.py` 的 `DEFAULT_BUCKETS`;目标 `vault.yaml` schema + `_buckets.md` 视图生成

实现进入 `reme4/steps/evolve/` 时,本文档与 `auto_memory_design.md` / `auto_consolidate_design.md` / `auto_cognition_design.md` 共同作为契约依据。
