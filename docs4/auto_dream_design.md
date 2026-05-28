# auto-dream 设计(digest 沉淀:物理 + 图)

> 本文档记录 reme4 中 **auto-dream**(digest 沉淀知识层)的设计讨论 —— 含物理布局、图模型(节点 + 边)、入流端 digester(G\*)。
>
> 配套阅读:
> - `structure.md` §1.2(数据视角)/ §2(三层存储)/ §3.5(digest 动作)/ §7(L4 模块)
> - `auto_memory_design.md`:auto-memory(daily 实时事件)是 dream 的入流之一(G1 scope 拉取);auto-memory 写完即对 dream 可见
> - `auto_maintain_design.md`:digest 的组织 / 重组 / 写入运行时(M split / D 检测 / CAS 写入协议) —— 本文档定义节点 + 边模型与 G\*,maintain 定义运行时
> - `auto_link_design.md`:auto-link 是 dream 写完之后的后置增强(实体识别 + wikilink 写回);复用 maintain 的 CAS 协议
>
> **四层对应**:reme 服务整体四份设计 —— auto-memory(daily 入流)/ **auto-dream(本文档:digest 沉淀 + content link + G\*)** / auto-maintain(digest 组织端:split + 检测 + CAS)/ auto-link(背景图关系增强)。`structure.md` §3.5-3.6 的 L4 action 视角:dream 对应 `digest` action(`resource + daily → digest`),maintain 对应 `maintain` action(`digest → digest`)。auto-dream 承担"空闲整理"(报告 §5.2):把 daily / resource 的散点抽出共性、合并重复、生成总结,落到 `digest/<bucket>/<slug>.md`。
>
> **核心立场**:dream 定义**模型层**(节点 + 边 / 守恒规则)+ **生成侧**(G\* create_or_update)+ **召回**(SearchStep);组织端(M split / D 检测 / CAS / 时序)归 `auto_maintain_design.md`;后置增强(实体识别 / wikilink 写回)归 `auto_link_design.md`。三方共享 §1.5 节点 + 边模型 + §1.5.5 边守恒 + §1.5.4 F-invariants。
>
> **关键收敛**:digest 不分"逻辑层"。整个 digest = **物理布局(浅桶 + flat .md)** + **一张图(节点 + 边)**。没有 hub / topic / leaf 之分 —— 所有 .md 文件都是同一种节点,内容决定它扮演什么角色(主题概览 / 概念定义 / 方法描述 / 实体记录 ...)。"主题"是从图中涌现的,不是结构性宣告的。

---

## 0. 问题陈述

digest 是 agent 长期记忆的"组织化沉淀"层,与三层架构的另两层职责互补:

| 层 | 组织主轴 | 形态 |
|---|---|---|
| resource/ | 时间(`<date>/<name>`) | 外部原始资料,不可变 |
| daily/ | 时间 + 任务(`<date>/<slug>/`) | agent 任务过程,半可变 |
| **digest/** | **语义** | **跨任务知识,可重组** |

digest 的核心问题:
- **物理布局**:文件系统怎么组织?(目录 / 文件 / 路径)
- **节点与边**:概念粒度 / 链接形态 / 主题如何涌现?
- **演化机制**:从碎片到网络的过程谁负责?(digester 创建或更新节点 / maintainer 拆分过载节点 / D 信号写后 inline 检测)

---

## 1. 已对齐决策

### 1.1 节点粒度:Atomic 节点优先

| 项 | 决策 |
|---|---|
| **粒度** | 一个文件 = 一个原子单元(概念 / 方法 / 实体 / 案例 / 原则 / 主题概览) |
| **节点角色** | 由内容决定,不由 frontmatter 类型标记;一个节点扮演"主题概览"还是"具体方法",看它的 body 写了什么 |
| **风格参考** | Zettelkasten:atomic note + 高密度 wikilink 网络 |

**理由**:
1. **节点粒度 = retrieve 精度上限** —— semantic 检索召回 "一个原子单元" 远比召回 "一个 5000 字的主题文档" 信噪比高;agent 上下文窗口经不起粗粒度文档塞满
2. **wikilink 在 atomic 粒度才真有意义** —— `[[jwt-rotation]]` 指向"一个具体方法"比指向"auth 主题文档"精确一个数量级,这也是 I-4(wikilink 唯一跨层载体)能撑起来的前提
3. **物理目录纯做 navigate,图承担关系** —— 两套机制各司其职,不互相绑架

**Tradeoff**:
- 文件数量爆炸(一个领域几百节点)→ 浅桶物理归档 + 路径作 ID,wikilink 走完整路径写法
- 节点会频繁演化(新材料 update 已有节点 / 节点过载触发 split)→ G\* 去重质量、SearchStep 召回、写后 inline 检测都得到位

### 1.2 物理几何:浅桶(shallow bucket,**固定集合**)

| 项 | 决策 |
|---|---|
| **物理布局** | `digest/<bucket>/<slug>.md`,bucket 一层(顶多两层),桶内 flat |
| **bucket 角色** | **仅承担物理归档与 OS-level 浏览锚点**;不承担语义本体角色 —— 主题由图中的节点表达 |
| **bucket 内** | 不再分子目录,所有节点平铺 |
| **bucket 集合** | **固定预定义,不由 digester / maintainer 动态生成** |
| **bucket 主页节点** | **不强制存在**;若 split 在该 bucket 内累积出层级(parent 节点天然中心性高),parent 节点天然成为浏览主页(纯约定,非架构必需) |
| **新节点归属** | digester (G4) 只能从已有桶集合里**挑选**;LLM 不能造新桶 |
| **集合来源** | vault 配置(opinionated default + 消费层可改),与 schema/prompt 同属服务消费层 |

**理由**:
- 物理浏览有"主题轮廓"(打开 `digest/auth/` 能看到这一族节点),不像纯 flat 那样毫无锚点
- 节点不被深路径绑死("属 auth/jwt 还是 auth/session"这种归属焦虑被消解 —— 一个节点可以同时被多个主题通过 wikilink 引用)
- maintain 操作面坍缩到只剩 split:节点过载就拆,不做跨节点重组(详 §1.5 / §2)
- **固定集合的关键意义**:
  - LLM 在 G4 桶决定时只做"分类",不做"造类" —— 决策面坍缩,错率大幅下降
  - bucket 集合作为**预先约定的物理归档规则**,跨任务跨时间稳定;不会出现 "auth-stuff" / "auth" / "authentication" 三个语义重叠的桶共存
  - 与 reme 核心立场一致:bucket 集合是消费层契约,reme 不自主造桶
- **bucket 主页节点不强制的关键意义**:
  - 旧设计中 root hub 是架构必需(检测稳定锚点 / vault 总览结构性入口);新设计中这些角色都由图中心性自然承担,不需要为每个 bucket 强制创建一个空架子节点
  - 主页节点的"主页"地位是涌现的:某节点入度高 / 中心性高 → 它就是浏览入口
  - 若一个 bucket 完全没节点,它就只是个空目录;不需要先造一个 placeholder

**未归类节点**:digester 抽到一个原子单元但找不到合适的专属桶时,**不允许造新桶**;统一落入兜底桶 `digest/general/`。general 桶在固定集合内是一等公民,详见 §3.7。

### 1.3 节点身份:路径即 ID

| 项 | 决策 |
|---|---|
| **ID 载体** | **vault-relative 路径**(含 `.md`)即节点身份 —— `digest/auth/jwt-rotation.md` |
| **wikilink 写法** | `[[digest/auth/jwt-rotation.md]]`(literal,与 `wikilink_handler.py` 默认形态对齐;不隐含 `.md`,无 short-form 补全) |
| **`name` frontmatter** | 文件名 basename(不含扩展名),与文件名同步 —— 检索 hint / 人读标签,**不当 ID 用** |
| **同名冲突** | 同 bucket 内文件名冲突 → 文件系统层断言;**不需要独立 D6 信号** |
| **rename 成本** | 一次 `wikilink_handler.retarget_links(old_path, new_path)`,机制现成 |
| **跨桶移动** | F-1 已禁止;若必须做(人工介入修错桶),走一次 retarget |

**理由**:
- F-1(0 文件移动)+ 平铺 + 下层 immutable 后,slug abstraction 的核心价值(移动鲁棒性)蒸发;只剩下"wikilink 短形式"这一项收益,但代价是 file_graph slug 索引 + D6 冲突检测 + alias 表 + retrieve 透明展开,**净亏**
- `wikilink_handler.py` docstring 自己写的就是 *Recommended form: full path relative to the vault with extension* —— literal 匹配,无 short-link 补全;路径作 ID 与核心库默认完全对齐
- 完整路径前缀 `digest/auth/` 给 LLM 读写时提供语义 context(知道节点在哪个桶),不全是负担
- provenance wikilink 反指 daily/resource 本来就用路径,统一后整个 vault 一种 wikilink 形态,不必区分"slug 形态 vs 路径形态"

### 1.4 链接语义:基础 wikilink + 可选 Dataview 谓词

参考实现:`reme4/utils/wikilink_handler.py` + `reme4/schema/file_link.py`。

| 项 | 决策 |
|---|---|
| **link 基础形态** | `[[<vault-relative-path>.md]]` —— target 字面取(literal,不隐含 `.md`,不自动短链补全);路径即 ID(详 §1.3) |
| **alias / image** | `[[path.md\|alias]]`(显示文本)/ `![[image.png]]`(图片资源)—— rewrite 时 alias 保持 |
| **anchor 不引入** | digest 设计层**不使用** `[[path.md#section]]` —— atomic 节点 + child 边界已充当精度替代品(详 §1.3 / §3.13);`wikilink_handler` 仍可解析 anchor 字面(供其它消费层),但 digest 不生成、不依赖、不在 split 时迁移 anchor |
| **可选谓词(Dataview 风格)** | 行级: `predicate:: [[path.md]]` / 内联: `[predicate:: [[path.md]]]`;**谓词写在 `[[]]` 外**,不是 `[[predicate::path]]` |
| **谓词标识符** | `[A-Za-z][A-Za-z0-9_]*`(如 `is_a` / `extends` / `causes` / `references`);词表**开放**,任意标识符 |
| **未类型化合法** | 绝大多数 wikilink **不加** predicate;`predicate=None` 是默认 / 常态 |
| **edge 唯一性键** | `(target_path, predicate)`(二元组);同源同标但不同 predicate = 不同边。`FileLink.target_anchor` 字段在 schema 中保留(供其它消费层),digest 层永远写 `None` |
| **类型信息载体** | 节点 frontmatter `kind` + 边 `predicate`(双轨可选);二者都是**消费层 schema 提示**,reme 核心解析 / 存储 / 索引,但**不读它们做结构决策** |
| **plugin 层扩展** | transclusion / 引用图谱视图等留给消费层加,reme 核心不固化语义 |

**理由**:
- 与 I-4 "wikilink 是唯一跨层载体" 对齐 —— reme 核心永远只看机械拓扑
- 与 [[reme4_schema_layering]] 一致 —— 类型语义(无论是节点 `kind` 还是边 `predicate`)都是消费层契约,reme 核心不固化
- predicate 走 Dataview 而非内嵌:`[[]]` 内容保持"纯目标"(rewrite / retarget 不必感知 predicate);predicate 是文本上的**装饰位**,与 wikilink 解耦
- `kind` 与 predicate 严格只是内容标签:reme 核心**只有节点这一种结构类型 + 边这一种结构关系**,kind / predicate 永远不参与"hub / topic / leaf"这类结构角色判断

**reme 核心对 predicate 的"透明"边界**(关键):
- G7(横向 link)、retrieve 中心性 —— 都**聚合所有 predicate** 算,不分桶
- 只有 edge 唯一性 / 反向索引会用到 predicate(否则 `[[A]]` 和 `is_a:: [[A]]` 会被当作同一条边互相覆盖)
- 消费层若要按 predicate 做更精细的推理(如"taxonomic 路径只走 `is_a` 边"),自己读 `FileLink.predicate` 即可

### 1.5 图模型与节点演化

**核心模型**:digest = **物理布局(浅桶 + flat .md)** + **一张图(节点 + 边)**。

| 维度 | 形态 |
|---|---|
| **节点** | 每个 .md 文件 = 一个节点;无结构性 kind,角色由 body 内容决定(主题概览 / 概念定义 / 方法描述 / 实体记录 / 案例 ...) |
| **边** | 基础 `[[<vault-path>.md]]` wikilink(路径即 target,详 §1.3 / §1.4);可选 Dataview 谓词 `predicate:: [[path.md]]` / `[predicate:: [[path.md]]]` 写在 `[[]]` 外;边唯一性键 = `(target, predicate)`;**digest 层不引入 anchor**(详 §1.4 / §3.13);**reme 核心结构决策不读 predicate**(详 §1.4) |
| **多归属** | 一个节点可被多个其它节点引用,也可指向多个其它节点;**不存在"单父"约束** |

**演化只做两件事**:
1. **G\* create_or_update**:新材料进入,LLM 提取原子单元 → 命中已有节点就 update 该节点 body(语义守恒地融合新旧),否则新建节点
2. **M split**:节点累积过载(token / 主题离散度超阈值)→ LLM 把它拆成 parent overview + N 个 children,parent 文件原地保留作 overview,children 是新文件

"主题概览节点 / 摘要节点"不是一种 kind,也不是 maintainer 主动涌现的产物 —— 它是 split 的副产品(parent 节点天然成为该 cluster 的 overview)。

#### 1.5.1 单一节点 / 单一边

**节点 frontmatter** —— 只有保留字段:

| 字段 | 内容 | 用途 |
|---|---|---|
| `name` | 文件名 basename(不含扩展名),与文件名同步 | 检索 hint / 人读标签(I-4 不再用它做身份;路径才是 ID,详 §1.3) |
| `description` | 一句话 | 标题 / 检索 hint |
| (可选)`kind` | concept / method / case / entity / topic / ... | **消费层 schema 提示**,reme 核心透明,不读它做结构决策 |
| body | 任意内容 | 一句定义 / 一段方法 / 一篇主题概览 / 一份案例,皆可 |

`hub__` / `topic__` 前缀**不存在**;文件名自然命名(`auth-fundamentals.md` / `jwt-rotation.md` / `jwt-overview.md`)。"overview 节点"靠内容形态识别,不靠前缀。

**边的形态** —— 与 §1.4 一致,这里给最小汇总:

| 维度 | 形态 |
|---|---|
| 基础 | `[[<vault-path>.md]]`(无谓词;常态;`predicate=None`) |
| 可选谓词 | `predicate:: [[path.md]]`(行级)/ `[predicate:: [[path.md]]]`(内联);谓词在 `[[]]` 外 |
| alias | `[[path.md|display-text]]`(rewrite 时 alias 保持) |
| image | `![[image.png]]`(资源引用,不是知识边) |
| **不引入 anchor** | digest 层不使用 `#section`;详 §1.4 / §3.13 |
| **边唯一性键** | `(target_path, predicate)` —— 同源同标不同 predicate = 不同边 |
| **角色识别** | 默认无谓词时由端点内容形态推断;有谓词时谓词即角色标签(消费层语义,核心不读) |

> **关键收敛**:reme 核心**只有节点 + 边两种结构类型**;kind / predicate 都是内容标签,绝不参与 hub / topic / leaf 这类结构角色。

#### 1.5.2 节点演化:create_or_update + split

```
[入流]  material 进入(daily / resource)
              │
              ▼
        G1 scope:选哪些 daily/resource 进入本轮
              │
              ▼
        G2 提取原子单元(LLM,可能产 N 个候选)
              │
              ▼
        对每个候选:
              │
              ▼
        G* create_or_update(LLM 决策点)
        ├─ 语义相似查 → 拉相似候选节点(top-k)
        ├─ LLM 判:候选中有"同概念节点"吗?
        │   ├─ 有 → update 路径
        │   │       (a) 把新内容融入已有 body(语义守恒重写)
        │   │       (b) 加 provenance 反指
        │   │       (c) 必要时加 / 改 wikilink
        │   └─ 无 → create 路径
        │           G3 路径 / G4 bucket / G6 provenance / G7 横向 link / 写 body
        │
        ▼
        写入(机械)

[D3 检测] 写后立即:G\* / split 写完 body 顺手 inline 检测(token 阈值 → 超阈值则 LLM 判离散度;详 `auto_maintain_design.md` §4)
              │
              ▼
        D3 派发候选节点(F-4 一次一个)
              │
              ▼
        M split(LLM + 机械)
        ├─ LLM 把节点 body 拆成 N 个 cluster(每个是个原子单元)
        ├─ parent 文件原地保留 → body 重写为 overview + 列出 children wikilinks
        ├─ 每个 child 创建新文件(文件名 / bucket / body 由 LLM 给)
        ├─ children 各自加 [[<parent-path>.md]] 反向链接
        ├─ 边守恒机械校验:`(parent_new ∪ ∪children) ⊇ parent_old` 出边集合(F-11 / E-2)
        └─ inbound 链 `[[<parent-path>.md]]` 不动(F-10 / E-3) —— digest 层无 anchor 链,无需 retarget
```

**关键**:
- **G\* update 改 subject body(语义守恒重写)** —— 新材料融入已有节点正文,要求 LLM 守住"只增不删 / 不改原意",老内容不能丢;**写入前机械校验出边强守恒**(`new outbound ⊇ old outbound`,详 §1.5.5 E-1)
- **G\* 不改其它节点正文** —— 只动 subject;不像旧 M-E 会到邻居 body append wikilink
- **M split 不改其它节点正文** —— 只动 parent(重写为 overview)+ 新建 children
- **inbound 在 split 时一律不动** —— digest 设计不引入 anchor,inbound 全是裸链 `[[<parent-path>.md]]`,parent 路径未变即天然有效;后续 G\* 进入若 LLM 觉得 child 粒度更合适,直接加新边到 child(F-10)

#### 1.5.3 走一个具体例子

**场景**:`digest/auth/` 桶,初始只有几个零散 auth 节点,没有 jwt-rotation。

**第 1 轮 G\***:某 daily 提到 "JWT rotation:每 24 小时换密钥,旧密钥保留 1 小时窗口给未过期 token"。
- 语义查 → 没找到 jwt-rotation 节点
- 走 create 路径 → 新建 `jwt-rotation.md`,body = 一段 200 字的 rotation 描述

**第 2 轮 G\***:另一个 daily 提到 "JWT rotation 的 grace period 通常是 1-2 小时"。
- 语义查 → 命中 `jwt-rotation`(高相似)
- LLM 判:这是同概念,走 update 路径
- 把 grace period 信息**融入** `jwt-rotation.md` body(不只是 append):
  ```
  Before: "...旧密钥保留 1 小时窗口..."
  After:  "...旧密钥保留 1-2 小时 grace period(典型值,具体看 token 寿命)..."
  ```
- body 略增长,加一条 provenance 反指

**第 N 轮 G\***:经过几个月,各种 daily 持续 update `jwt-rotation` —— 加了密钥派生算法、加了 RS256/HS256 区别、加了 key rotation 失败处理、加了 with-leeway 实践、加了 monitoring 建议 ...

`jwt-rotation.md` body 现在 ~3500 token,涵盖:轮换策略 / 密钥派生 / 算法选择 / 失败处理 / 监控。

**触发**:D3 检测 token > 2000 阈值 → 派发候选。

**M split**:LLM 拉 `jwt-rotation.md` body + frontmatter,判断主题离散度(5 个相对独立的子主题),决定拆:
- parent: `jwt-rotation`(留下,body 重写为 overview)
- children: `jwt-key-derivation` / `jwt-algorithm-selection` / `jwt-rotation-failure-handling` / `jwt-rotation-monitoring`(4 个新文件)
- "with-leeway 实践"内容并入 parent overview(粒度太细不单独拆)

**执行后**:

```
digest/auth/
├── jwt-rotation.md                  ← 文件原地;body 重写为 overview
│   description: JWT 轮换策略总览
│   body: JWT 轮换的核心是 X,主要环节包括:
│         密钥派生 [[digest/auth/jwt-key-derivation.md]]
│         算法选择 [[digest/auth/jwt-algorithm-selection.md]]
│         失败处理 [[digest/auth/jwt-rotation-failure-handling.md]]
│         监控告警 [[digest/auth/jwt-rotation-monitoring.md]]
│         (参考:with-leeway 实践 ...)
├── jwt-key-derivation.md            ← 新建,body 来自原 jwt-rotation 拆出片段
│   → [[digest/auth/jwt-rotation.md]]               ← child 反指 parent
├── jwt-algorithm-selection.md       ← 同上
│   → [[digest/auth/jwt-rotation.md]]
├── jwt-rotation-failure-handling.md ← 同上
│   → [[digest/auth/jwt-rotation.md]]
├── jwt-rotation-monitoring.md       ← 同上
│   → [[digest/auth/jwt-rotation.md]]
└── ...(其它 auth 节点不变)
```

**inbound 不动**:之前指 `jwt-rotation.md` 的所有外部 wikilink(无论裸链还是 typed)都仍然指 `[[digest/auth/jwt-rotation.md]]`。如果后续某外部节点写新材料时 LLM 觉得 child 粒度更合适,直接 G\* 时新加 `[[digest/auth/jwt-key-derivation.md]]` 这种边即可 —— 不强求 split 时即时重定向。

**继续演化**:
- 若某个 child(如 `jwt-key-derivation`)被持续 update,某天也长到过载 → D3 又触发 → 它再次 split,自然涌现第三层
- 若某 child 长期空 / 0 入度 / 0 update —— 不主动删(没有 dissolve 操作);除非人工介入

#### 1.5.4 操作的核心约束

| # | 约束 | 含义 |
|---|---|---|
| **F-1** | **0 文件移动** | G\* / split 都不移动现有文件;split 创建的是**新文件**,parent 文件原地 |
| **F-2** | **改正文限定 subject** | G\* update 改 subject node body(语义守恒重写,不改其它节点);M split 改 parent body(重写为 overview)+ 创建 children body;**没有任何操作改"其它节点正文"** |
| **F-3** | **maintainer 只做 split** | 没有 summarize / merge / re-edge / link / unify / dissolve;过载 → 拆 |
| **F-4** | **一次一个候选** | M split 一次拆一个节点;G\* 一次处理一个原子单元(N 个候选 = N 次 G\*) |
| **F-5** | **不确定时不动** | G\* 拿不准是 create 还是 update → 倾向 create(不污染已有节点);split 拿不准 cluster 边界 → 不拆 |
| **F-7** | **多归属合法** | 一个节点可被多个其它节点引用,也可指向多个其它节点;**没有"单父"约束** |
| **F-10** | **inbound 目标节点不动** | 所有 inbound 都是裸链 `[[<parent-path>.md]]`(digest 不引入 anchor,详 §1.4 / §3.13);split 时全部保持不动,parent 路径未变即天然有效;后续 G\* 进入时 LLM 可自由选择更精细 target(直接加新边到 child) |
| **F-11** | **wikilink 是 body 的一部分** | 不存在"独立的边" —— 边的所有迁移都是 body 文本变化的副作用;reme 核心机械算子只感知字符层,语义责任在 LLM(G\* / split prompt)+ 守恒校验(outbound diff 机械验证;详 §1.5.5) |

#### 1.5.5 边的迁移规则

**前提**:wikilink 是 body 的一部分(F-11)。"边"不是独立抽象 —— body 一变,边就跟着变。reme 核心**没有"修边"算子**,边的所有变化都是 body 文本编辑的副作用。

但语义守恒不能放任 LLM:守恒责任在 prompt + 机械校验,不在算子。

**3 类边按"在哪类操作中变化"区分**:

| # | 类别 | 规则 | 谁负责 |
|---|---|---|---|
| **E-1** | **G\* update 节点出边**(subject 自身) | **强守恒**:新 body 出边集合 ⊇ 原 body 出边集合(`(target, predicate)` 二元组比对,**predicate 一并守住**);不满足 → LLM 重试或拒写 | LLM(prompt 强约束)+ 机械校验(outbound diff) |
| **E-2** | **split parent 出边**(parent body 拆解) | parent overview + N 个 children 各持一段,原 parent 出边按内容自然分配到 parent overview + children;**机械校验合计守恒**:`(parent_new ∪ ∪children_outbound) ⊇ parent_old` | LLM(split prompt)+ 机械校验 |
| **E-3** | **inbound wikilink** `[[<parent-path>.md]]` | split 时**不动** —— 仍指 parent;后续 G\* 进入若 LLM 觉得 child 粒度更合适,直接加新边到 child(F-10) | 不动 |

**provenance 不单列一类**:节点反指上游 daily/resource 的 wikilink 是 body 正文的一部分(§3.9),由 LLM 在 G\* / split prompt 中自然写出 —— 跟其它 body wikilink 走同一套规则:G\* update 走 E-1 强守恒(老 provenance 链不能丢,新材料追加新 provenance),split 走 E-2 合计守恒(parent 全量 provenance ⊆ parent_overview ∪ ∪children_outbound)。reme 核心**没有** provenance 专用算子。

**inbound anchor 这一类不存在**:digest 设计层不引入 anchor(详 §1.4 / §3.13),所有 inbound 都是裸链,走 E-3 即可,无需机械 retarget 子流程。

> **关键拆分**:
> - **守恒**(E-1 / E-2):LLM 写正文时不能丢边;靠 prompt + 写后 outbound diff 校验
> - **保守**(E-3):没有信号说一定要变;不变的代价 = 后续 G\* 自然纠正,变的代价 = 错信号大量假阳;选不变

**机械 outbound diff 校验**(E-1 / E-2)伪码:

```
write_subject_body(subject, new_body):
    old_outbound = extract_links(old_body)   # set of (target, predicate)
    new_outbound = extract_links(new_body)
    missing = old_outbound - new_outbound
    if missing:
        # LLM 漏了原边 —— 重试一次
        new_body = llm_retry_with_missing(missing)
        new_outbound = extract_links(new_body)
        if old_outbound - new_outbound:
            raise ConservationViolation(...)  # 拒写,记 audit,等人介入
    write(subject, new_body)
```

机械层只做集合比对,**不判断"为什么丢了"** —— 那是 LLM 的事。

**强守恒(集合包含)而非等价**:`new ⊇ old` 是"新材料融入,老知识保留"的最小契约 —— 允许加新边(新关联),不允许减边(老内容不能丢);等价(`new == old`)会拒绝任何新出边,update 失去意义。

**predicate 守住** —— `[[A]]` ↔ `is_a:: [[A]]` 视为不同 key,升降级走显式 audit 路径,不走默认。重排 / 改 alias / 加新边都不被拦下(集合相同或只增)。

#### 1.5.6 图模型 vs 建子目录

| 维度 | 建子目录(深树) | 图模型 + split 演化 |
|---|---|---|
| 物理变化 | 移动文件,改路径 | 0 文件移动(F-1);split 只创建新文件 |
| wikilink 影响 | 路径 ID 模型下要全图 retarget(代价大) | 0 影响(parent 路径未动);split 不触发任何 retarget |
| 主题归属 | 一个节点只能属一棵子树 | 一个节点可同时属多个主题(被多源 wikilink) |
| 撤销成本 | 移回文件 + 重建上下文 | 删 children + parent body 还原(手工) |
| 演化路径 | 子树重组痛苦 | parent 只增不减,children 是 parent 拆出的快照 |
| navigate | 浏览目录树 | 任意节点入手沿出边漫游;parent 节点是天然中心 |
| retrieve 精度 | 路径反映主题但与 link 无关 | 节点中心性 + 内容形态共同决定权重 |

#### 1.5.7 多级结构自然涌现

split 是节点的**局部操作**(只看一个过载节点),多级深度自然涌现:

```
digest/auth/
├── auth-fundamentals.md             ← 早期写下,~1500 token,稳定
├── jwt-rotation.md                  ← 第一次 split:body 从 3500 token 重写为 overview
├── jwt-key-derivation.md            ← 第一次 split 的 child
├── jwt-key-derivation-hkdf.md       ← 二次 split:jwt-key-derivation 累积 update 后过载,再拆
├── jwt-key-derivation-pbkdf2.md     ← 二次 split 的 child
├── ...
```

**物理仍是浅桶(1 层),"层级"由 split 链 + 节点中心性自然承载**。每一级的过载条件、决策机制、执行步骤完全相同 —— 没有"二级 split"特殊逻辑,只有"过载节点的 body 可以被 split 进一步拆"。

#### 1.5.8 retrieve 时节点怎么参与

| query | 期望返回 |
|---|---|
| "JWT 怎么轮换" | 优先 `jwt-rotation`(具体 overview)+ children(如 `jwt-key-derivation`) |
| "auth 体系" | 优先中心性高的节点(`auth-fundamentals` / `jwt-rotation` 等被多次 update / 是 split parent 的节点) |
| "auth 有什么子主题" | 沿高中心性节点的入/出邻居遍历;parent 节点优先返回 |
| "vault 里都有什么" | 各 bucket 中心性最高的节点(自然形成 vault 总览) |

**加权策略**(opinionated default,消费层可改):
- 节点权重 = base(=1.0) × intent 调节 × 中心性增益
- query 含"概览 / 主题 / 入门 / 全景"等**元意图**时,中心性高的节点加权(intent 调节 > 1)
- 中心性低 / body 短的具体节点权重稳定(默认 1.0,不被压低)

**topological traverse**:
- 沿 wikilink 自由走(不区分边类型 / predicate)
- 经过中心性高的节点默认**不强行展开**(否则一次 traverse 把整族 children 拉进来);agent 可显式深入

**中心性的天然来源 = split parent**:被拆过的节点是 parent,自然有 children 反向链接它,中心性自然高 —— 不需要单独维护 `kind: hub` 标记。

---

## 2. 完整能力集

### 2.0 设计目标

**让图的形状持续匹配实际知识的语义结构,在最小变更面 + 渐进演化的前提下,使任意尺度的知识访问都能命中合适粒度的节点。**

这个目标直接来自结构本身的设计意图 —— 浅桶 + 单一节点类型 + 单一边类型 + create_or_update + split 的组合,每一项都是为它服务。能力集的入选标准:**对至少一个验证维度有贡献**。

| 维度 | 含义 | 失败示例 |
|---|---|---|
| **形状匹配** | 节点中心性 / 边连接 / 节点邻域反映知识间的实际语义关系 | 一个节点 token 5000+ 长期不拆;同主题节点彼此 0 链接;同概念被建成多个独立节点 |
| **最小变更面** | 不重写其它节点正文,不大规模移文件,不破坏现有 wikilink | 任何"全图重组"或"批量改其它节点正文"的方案 |
| **任意尺度访问** | 具体方法节点 / overview 节点 / 节点邻居遍历都能命中 | 全 flat,主题级 query 命中不到东西 |

**显式排除**(不在目标内,避免能力集内卷):
- ❌ "完美归簇" —— F-5 留白,不确定就不动
- ❌ "实时一致" —— 异步 / eventual,节点写完不必立刻 split
- ❌ "零冲突 / 零违反" —— invariants 检测 + 事后修复,不追求永不发生
- ❌ 替消费层做检索 / 决策 —— digest 自治边界止于"维持图的形状"
- ❌ 跨节点重组(merge / re-edge / unify / dissolve)—— 简化模型不做这些;同概念二次进入由 G\* update 路径处理

**演化只有两件事**:G\* create_or_update(入流型,新材料融入)+ M split(后台,过载就拆)。detection 派生信号驱动这套循环。

### 2.1 生成侧:digester(入流型)

| # | 能力 | 服务 | 性质 | 何时发生 |
|---|---|---|---|---|
| **G1** | **scope 决定**:选哪组 daily/resource 进入本轮蒸馏 | 形状匹配(决定形状从哪生长) | LLM | digester 启动 |
| **G2** | **原子单元抽取**:从 scope 中识别值得沉淀的原子单元(N 个候选) | 形状匹配 + 任意尺度 | LLM | 核心环节 |
| **G\*** | **create_or_update**:对每个候选,**多路召回(SearchStep:vector + keyword + 邻接展开,RRF 融合,scope 限 `digest/`)** → LLM 看完整候选池 → 终判 create / update / drop;create 路径走 G3/G4/G6/G7 + 写 body;update 路径融入已有 body(语义守恒重写)+ 自然追加 provenance(详 §3.10) | 形状匹配(去重内置)+ 最小变更面 | LLM(决策)+ 机械(召回 + 守恒写入) | 每个候选 |
| **G3** | **路径命名**(create 路径):在 G4 选定 bucket 内,文件名同 bucket 唯一(fs 层断言);风格与同主题节点一致 | 任意尺度(可寻址) | LLM(命名)+ 机械(同 bucket 文件名冲突 → 拒写) | create 时 |
| **G4** | **bucket 落地**(create 路径):从固定集合中挑选;找不到合适专属桶 → 落 `general/`(§3.7) | 形状匹配(物理归档) | LLM(读 bucket 列表) | create 时 |
| **G6** | **provenance 写入**(create 与 update):新节点 body 内联反指上游 daily/resource 的 wikilink;update 时 LLM 在融入新材料时自然追加新 provenance 链,旧 provenance 链由 E-1 守恒校验保住(§3.9) | 任意尺度(跨层访问) | LLM(prompt 引导写出 `[[daily/...]]` / `[[resource/...]]`)+ 机械(outbound diff 校验) | 写入时 |
| **G7** | **横向 link**(create 时):新节点链到相关的已有节点(出边);update 时也可加新 link | 形状匹配 + 任意尺度 | LLM | 写入时 |

**关键边界**:
- **G\* 是入流唯一改 body 的操作**,且**只改 subject node** —— update 改的是同概念那个节点自己,不改其它节点
- **G\* update 必须语义守恒**:LLM 重写 body 时只能"融入"新内容,不能删除已有信息(只增不删 / 不改原意)
- **0 出边节点合法**(G7 没识别到合适邻居),后续 G\* 进入时其它节点可以反向链回来 —— 不强求 LLM 一次性给全
- **G\* 漏判去重**(把同概念建成新节点)→ 不主动兜底,接受重复;若 vault 累积明显的同概念重复,可由 auto-link 离线 audit 工具产报告(详 `auto_link_design.md` §1.3 L4)
- **digester 不做 split** —— split 是后台 M 操作

### 2.2 组织侧 / 检测 / 写入并发 → `auto_maintain_design.md`

M split / D 检测信号(D1 / D3 / D10)/ 阈值校准 / D3 写后触发模型 / G\* / split / auto-link L1 三方共用的 CAS 写入协议 / split provenance / 时序 / 后门 —— 全部归 `auto_maintain_design.md`。

dream 保留**模型层**(§1 节点 + 边 + 守恒规则)+ **生成侧**(§2.1 G\*)+ **召回**(§3.10 SearchStep);maintain 负责**组织 / 运行时**(split + D + CAS + 时序)。两份文档共享 §1.5 节点 + 边模型、§1.5.5 边守恒、§1.5.4 F-invariants。

| 在 maintain 文档中 | 内容 |
|---|---|
| §1 | M split 能力卡 + 关键边界 |
| §2 | 检测信号 D1 / D3 / D10 |
| §3 | 阈值校准 |
| §4 | D3 写后触发模型 |
| §5 | CAS 写入协议(三方共用) |
| §6 | split 时 provenance |
| §7 | G\* / split / auto-link L1 时序 |
| §8 | 后门(暂缓) |

### 2.4 边界协议(谁不能做什么)

| 边界 | 内容 | 来源 |
|---|---|---|
| digester ∩ maintainer | digester 不做 split;maintainer 不做原子单元抽取 / 新具体节点 create | 入流 vs 自维护职责分离 |
| digester → 其它节点 | G\* update 改 subject node body,**不改其它任何节点正文** | F-2 |
| digester → "摘要 / overview" | digester 不为做 overview 而创建节点;它产的节点都是具体原子单元;overview 是后续 split 的副产品 | F-3 |
| maintainer → 其它节点 | M split 改 parent body(重写为 overview)+ 创建 N 个 children body;**不改任何其它节点** | F-2 |
| maintainer → inbound 链 | split 时**全部不动** —— digest 不引入 anchor,inbound 一律是裸链 `[[<parent-path>.md]]`,parent 路径未变 | F-10 / E-3 |
| digester → 边守恒 | G\* update 写新 body 前,机械对比 old/new outbound:`new ⊇ old`((target, predicate) 二元组);失败 → LLM 重试一次,再失败拒写 | F-11 / E-1 |
| maintainer → 边守恒 | split 写新 parent body + N children body 前,机械对比:`(parent_new ∪ ∪children_outbound) ⊇ parent_old`;失败 → LLM 重试或拒写 | F-11 / E-2 |
| 全员 → typed link predicate | wikilink 的 predicate 是 edge identity 的一部分;G\* update / split 不能丢 predicate(`is_a:: [[A]]` 必须保持;否则被守恒校验当作 drop edge + add edge 拦下);predicate 升 / 降级走显式 audit 路径 | F-11 / §1.4 |
| 全员 → resource/daily | 都不能改 | I-2 / I-3 |
| 全员 → 节点 rename | rename = 一次 `wikilink_handler.retarget_links(old_path, new_path)`;无 alias 表,无透明展开 | §1.3 |
| 全员 → provenance link | 永远必须可达(I 不变量 + D10 检测) | I-1 / I-4 |
| 全员 → kind 字段 | reme 核心**透明**:不读取 frontmatter `kind` 做结构决策;`kind` 是消费层 schema 提示 | [[reme4_schema_layering]] |
| 全员 → predicate 谓词 | reme 核心**结构决策不读**:G7 / 中心性都聚合所有 predicate 算;edge 唯一性 / 反向索引会用到 predicate(防同源同标不同 predicate 互相覆盖);未类型化 link 是默认形态 | [[reme4_schema_layering]] / §1.4 |

---

## 3. 待对齐边界点(后续讨论清单)

### 3.1 G\* update 的语义守恒边界(已收敛)

**决策**:**LLM 重写整段**(prompt 强约束"语义守恒,只增不删 / 不改原意;冲突标注 `> 注:不同来源记载...`,不擅自仲裁")+ **机械守恒校验**(详 §1.5.5 E-1)。校验失败 LLM 重试一次,再失败拒写 + audit。

首版可先用 append 起步(出边集合天然 ⊇,守恒校验自动通过),prompt 工程量小;成熟后切到重写。

### 3.2 maintainer 的人 / agent 后门 → `auto_maintain_design.md` §8

### 3.3 G\* 与 split 的时序 → `auto_maintain_design.md` §7

### 3.4 节点 kind / 边 predicate(已收敛)

reme 核心**只有节点 + 边两种结构类型**:
- frontmatter `kind` 字段(若存在)= 消费层的**节点内容标签**(concept / method / case / entity / topic / ...),reme 不读它做结构决策
- 边 `predicate`(Dataview 风格,若存在)= 消费层的**边关系标签**(`is_a` / `extends` / `causes` / ...),reme 解析 / 存储 / 参与 edge 唯一性,但**结构决策不读**(G7 不区分 predicate;中心性不区分)
- "overview 节点"角色靠图位置(高中心性 / 是 split parent)+ body 内容形态识别,不靠 frontmatter 或 predicate 标记
- 未类型化 wikilink 是默认 / 常态形态

详见 §1.4 / §1.5.1 / §2.4。

### 3.5 retrieve 时的权重策略(部分收敛 → §1.5.8)

- 加权策略:节点权重 = base(=1.0) × intent 调节 × 中心性增益;query 含元意图("概览 / 主题 / 入门 / 全景"等)时,中心性高的节点加权
- traverse 默认不强行展开高中心性节点(防止整族 children 拉进来);agent 可显式深入
- 不按 frontmatter `kind` 加权;中心性天然来源 = split parent(详 §1.5.8)

剩余待定:**中心性算法选型**(eigenvector / PageRank / 简单入度,初期可用入度,后续校准)。

### 3.6 M split 时的 provenance 处理 → `auto_maintain_design.md` §6

### 3.7 bucket 集合管理

§1.2 已定:bucket 集合**固定预定义**,不由 digester / maintainer 动态生成。补足细节:

- **定义位置**:`vault.yaml` 顶层 `digest.buckets:` 是源 + 自动生成 `digest/_buckets.md` 作为人/LLM 可读视图;digester G4 时读后者作为 prompt context
- **初始化**:opinionated default(通用桶 `concept` / `method` / `pattern` / `tool` / `domain` 等 + 必带 `general`);消费层可改桶名,但 **`general` 不可删**(否则 G4 失去兜底)
- **扩展路径**:reme 不主动提议扩 bucket(对比旧设计的 maintainer 周期建议已 DROPPED);用户编辑 `vault.yaml` 后下次 G4 即生效

**未归类节点处理**(G4 找不到合适专属桶时):**统一落入 `digest/general/`**。

| 维度 | 内容 |
|---|---|
| **bucket 名** | `general`(固定集合一等公民,默认包含) |
| **语义** | "通用主题 / 暂无专属归属" —— 合法常态,非故障状态 |
| **路径** | `digest/general/<slug>.md`,与其它 bucket 完全等同 |
| **节点演化** | 与其它 bucket 一致 |
| **错桶后续** | 不主动跨桶 move(无 D9 / M-D);若严重,人工 mv + `retarget_links(old, new)` |

**为什么是 `general` 而不是 `_unclassified`**:`_unclassified` 暗示待处理状态,LLM/人都想清理掉;`general` 是合法常态,G4 选桶时是显式合法选项而非 fallback 故障路径。

**已排除**:拒绝写入(候选丢失)/ 强行选最近似专属桶(本体污染,general 反而更安全)。

### 3.8 检测阈值校准 → `auto_maintain_design.md` §3

### 3.9 provenance 载体形态(已收敛)

**决策**:**provenance wikilink 嵌在节点 body 正文中**(inline body prose),由 LLM 在 G\* / split prompt 里自然写出,跟其它 body wikilink 完全同形,**靠语义维护**。reme 核心没有 provenance 专用算子。

**写出形态**:
- 行文中自然带出处:"... 该模式最早出现在 [[daily/2026/05/15.md]] 的实践中"
- 或专门一段总结式段落,内含若干 wikilink 指向上游
- 可选 predicate(`derived_from:: [[daily/2026/05/15.md]]`),不强制

**机械保护**:
- E-1 守恒(G\* update):旧 provenance 不在新 outbound 集合 → 重试或拒写,机械兜底
- E-2 守恒(split):provenance 跟着对应内容段自然分配到 parent overview / children,合计守恒
- D10 检测:provenance 断裂 = D1 断链子集(target 命中 `daily/` / `resource/` 前缀);D10 严重程度高于普通 D1(I 不变量)

**E-5 / G6 等"provenance 专用机制"全部坍缩** —— 不再单列。Prompt 必须要求"出处用 `[[...]]` 形式表达"(纯散文会被守恒校验视为丢边)。

### 3.10 G\* 语义查后端(已收敛)

**决策**:**直接复用 `SearchStep`(`reme4/steps/index/search.py`)** —— 多路召回并发(vector + keyword)+ RRF 融合 + file_graph 邻接展开,把**完整候选池交给 LLM 终判**;G\* 入口不做 bucket 粗筛(LLM 拥有完整跨桶视野,可识别"概念错分到 general"或"跨桶同概念";三路信号 RRF 融合后噪声可控)。

**召回链路**:
1. 候选原子单元(摘要 / 关键词)→ `SearchStep`(`search_filter={"path_prefix": "digest/"}`,I-2/I-3 daily/resource 不入池)
2. `SearchStep` 内部:`vector_search` + `keyword_search` 并发 → RRF 融合 → `expand_links` 邻接展开 → 返回 top-`limit` FileChunks(含 path / 行号 / 邻接节点)
3. 候选池整体喂 LLM,按 path 自然聚合(同节点多 chunk 命中 = 强信号);终判输出节点路径
4. LLM 终判 create / update / drop;update 选定 subject node → 走 E-1 守恒重写

**provenance 不依赖召回** —— G\* / split prompt 让 LLM 直接写 `[[daily/...]]` / `[[resource/...]]`(§3.9)。

**索引维护**:沿用 `update_index` step,G\* / split 写 body 后调一次刷该节点索引;启动一次全建(`clear_and_scan` 已就绪),损坏走全建兜底。

**默认参数**(可按 dogfooding 调):`limit` 5~10 / `vector_weight` 0.7 / `expand_links` on / `min_score` 0(初版不过滤,LLM 兜底)。

### 3.11 G\* / split 写入并发 / 原子性 → `auto_maintain_design.md` §5

### 3.12 D3 触发模型 → `auto_maintain_design.md` §4

### 3.13 anchor 不引入 wikilink 设计(已收敛)

**决策**:**digest 设计层不使用 `[[path.md#section]]` 形态** —— wikilink 只有 `[[path.md]]`(可选 alias / 谓词),anchor 不进入 digest。当 LLM 想"指向某个具体子主题"时,正确做法是让那个子主题升级为独立节点(必要时通过 split),而不是在过载 parent 内部用 anchor 凑合。

**连锁简化**:
- E-4(inbound anchor 机械 retarget)整类**消失**;split 流程末尾不再扫 inbound anchor 子流程;`{anchor → child}` 映射输出从 split prompt 中移除
- 边唯一性键从三元组 `(target, predicate, anchor)` 简化为二元组 `(target, predicate)`
- §1.5.5 边迁移类别从 4 类(E-1..E-4)简化为 3 类(E-1..E-3)
- `FileLink.target_anchor` 字段在 schema 中保留(供其它消费层),digest 层永远写 `None`

**Prompt 约束**:G\* / split 的 prompt 必须明确告知 LLM 写 wikilink 时不带 `#section`。若 LLM 仍写出 `[[path.md#section]]`,wikilink_handler 仍能解析,守恒校验只看 `(target, predicate)`,不会形成"丢边"风险 —— 但 anchor 在 digest 层无语义。若引用方依赖某 anchor 锚定具体段落,表明该内容应升级为 child 节点。

---

## 4. 下一步

本文档覆盖 dream 模型 + 生成侧 + 召回(G\* / 节点+边模型 / SearchStep)。组织端实现清单(M split / D 检测 / CAS)见 `auto_maintain_design.md` §10。

1. **digester 流程图**(G1 / G2 / G\* 的实际编排;G\* 内 create / update 路径分流;**召回直接复用 `SearchStep`**(vector + keyword + 邻接展开,RRF 融合,scope `digest/`)—— 详 §3.10;**G\* update 写入前 outbound diff 守恒校验** — E-1)
2. **rename 路径设计**(`wikilink_handler.retarget_links(old_path, new_path)` 已就绪;封装为单步 step 入口,无 alias 表 / 无透明展开)
3. **bucket 集合配置**(`vault.yaml` schema / 默认桶模板 / `general` 兜底机制 / `_buckets.md` 视图生成)
4. **边守恒校验工具**(`extract_links` 已就绪;新增 outbound diff 比较器 + LLM 重试编排 + ConservationViolation audit 事件)
5. **provenance prompt 规范**(G\* / split 引导 LLM 写 `[[daily/...]]` / `[[resource/...]]` —— §3.9)

实现进入 `reme4/steps/jobs/` 与 `reme4/file_graph/` 时,本文档与 `auto_memory_design.md` / `auto_maintain_design.md` / `auto_link_design.md` 共同作为契约依据。
