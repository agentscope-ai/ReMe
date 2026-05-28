# auto-maintain 设计(digest 组织端:M split / 检测 / 写入并发)

> 本文档记录 reme4 中 **auto-maintain** 的设计讨论 —— digest 层的组织 / 重组 / 写入运行时,含 M split、D 检测信号、写后触发模型、CAS 写入协议。
>
> 配套阅读:
> - `structure.md` §3.6(maintain 动作语义)/ §7.3(maintainer 模块)
> - `auto_dream_design.md`:节点 + 边模型(§1.1-1.5)/ F-invariants(§1.5.4)/ 边守恒 E-1/E-2/E-3(§1.5.5)/ G\* 操作(§2.1)—— maintain 复用这套底层模型
> - `auto_link_design.md`:auto-link 写回也走本文档的 CAS 协议(§5)
> - `auto_memory_design.md`:auto-memory 不直接复用 maintain,但事件级"拆"与节点级 split 在概念上同构(都把过载粒度切小)
>
> **三层框架的位置**:报告 §5 三层为 auto-memory / auto-dream / auto-link。maintain 严格按 `structure.md` §3.5-3.6 的 L4 action 分类是独立 action(`maintain: digest → digest`),不属 `digest` action(`digest: resource + daily → digest`)。本文档作为四方分工的**第四份**,专门覆盖 dream 写完之后 digest 的组织 / 重组 / 写入运行时。
>
> **核心立场**:
> - **maintain 与 dream 同 pace**(idle background)、同模型(节点 + 边 / 守恒规则),但**语义边界不同**:dream 是 compose(资料 → digest),maintain 是 reorganize(digest → digest)
> - **maintain 只做 split**,不做 merge / dissolve / re-edge / unify;过载就拆,其它跨节点重组留给消费层 / 人工
> - **CAS 写入协议是基础设施**,被 dream G\* / maintain split / auto-link L1 共用,统一编排在本文档(§5)

---

## 0. 问题陈述

dream 模型(`auto_dream_design.md` §1.5)规定 digest 的演化只做两件事:G\* create_or_update(入流型,新材料融入)+ M split(后台,过载就拆)。dream 文档负责 G\* 与节点 / 边模型;**本文档负责 M split 与运行时机制**(D 检测 / 触发模型 / 写入并发协议)。

| 输入 | 输出 |
|---|---|
| dream 写入后的 digest 状态 + 触发信号(D3 过载,inline) | parent overview 重写 + N 个新 children 文件;边守恒 E-2 通过 |

**设计目标**:
1. **形状匹配** —— 让节点粒度持续与实际语义结构对齐(过载节点拆;不过载不动)
2. **最小变更面** —— split 改 parent + 创建 N children,不动其它节点(F-2)
3. **不引入新基础设施** —— 复用 dream 的节点 + 边模型 / 守恒规则;CAS 写协议自洽

**显式排除**:
- ❌ merge / dissolve / re-edge / unify —— 跨节点重组不做(简化模型;同概念二次进入靠 G\* update)
- ❌ 改其它节点正文 —— split 只改 parent body(重写为 overview)+ 创建 children body
- ❌ 重建 inbound —— split 时 inbound 一律不动(F-10)

---

## 1. M split(maintainer 唯一 op)

| # | 能力 | 服务 | 触发 | graph | file | body |
|---|---|---|---|---|---|---|
| **M split** | 节点过载 → LLM 拆成 parent overview + N 个 children;parent 文件原地保留,children 是新文件;children 加 `[[parent]]` 反向链接;inbound 边不动 | 形状匹配(粒度对齐)+ 任意尺度(涌现层级) | D3 过载 | parent 0 拓扑改;新 children 节点 + 各自加 `[[parent]]` 出边 | 创建 N 个 children 文件;parent 文件原地 | parent body 重写为 overview;children 各自有新 body |

**关键边界**:
- **M split 改两类 body**:parent body(重写为 overview)+ N 个新 children body;不改任何**其它**节点(`auto_dream_design.md` §1.5.4 F-2)
- **inbound 不重定向** —— 外部对 parent 的 wikilink 全部保留指 parent;后续 G\* 进入时若 LLM 觉得 child 粒度更合适,直接加新边到 child 即可(F-10)
- **没有 dissolve 操作** —— children 长期空也不主动删;消费层 / 人工显式介入
- **没有 merge / re-edge / unify** —— 跨节点重组不做;同概念二次进入靠 G\* update;错桶节点不主动 move(若严重,人工介入)
- **边守恒** —— split 写新 parent body + N children body 前,机械对比 outbound:`(parent_new ∪ ∪children_outbound) ⊇ parent_old`;失败 → LLM 重试或拒写(F-11 / E-2,详 `auto_dream_design.md` §1.5.5)

---

## 2. 检测信号 D1 / D3 / D10

| # | 信号 | 服务 | 服务能力 |
|---|---|---|---|
| **D1** | 断链(wikilink → 不存在的 path) | 任意尺度(可达性) | 告警 / 简单修复(就地删 wikilink 或保留 alias 文本) |
| **D3** | 过载节点(token 阈值 → LLM 判离散度) | 形状匹配(粒度) | maintainer(M split) |
| **D10** | provenance 断裂(digest 节点反指的 daily/resource 不可达) | 任意尺度(跨层不变量) | 严重告警(I 不变量违反) |

**触发模型**:**写后立即** —— G\* / split 写完 body inline 检测;无后台 watcher / 无周期 tick / 无 dirty 队列(详 §4)。D1 / D10 是 wikilink 断链的子集,跟 file_graph 链路一起在写时检测。

> **简化模型砍掉的信号**:
> - **D2 隔离 / D4 过疏 / D5 高入度 / D5b 低入度摘要 / D6 slug 冲突 / D7 相似未链 / D8 重复语义 / D9 邻居异质** —— 全部 DROPPED
> - 旧 D5 高入度涌现 → 由 split 副产品(parent + children)等价覆盖;触发源换成节点过载(D3)
> - 旧 D6 slug 冲突 → 路径即 ID 后,同 bucket 内文件名冲突由文件系统层断言(写入即拒),不需要独立信号(详 `auto_dream_design.md` §1.3)
> - 旧 D7 / D8 → 简化模型不做 link / merge 提议;若 vault 累积明显的同概念重复,由 `auto_link_design.md` §1.3 L4 离线 audit 工具产报告
> - 旧 D9 邻居异质 → 简化模型不做跨桶 move;桶选择只在 G4 一次性决定,后续不重排
>
> **D3 过载的判据**:token 阈值机械检查 + LLM 判离散度;**写后立即 inline**。阈值见 §3,触发模型见 §4。

---

## 3. 检测阈值校准

简化模型只剩 D3(过载)是核心阈值,其它都是 invariant 触发(无可调阈值)或 informational(无 maintenance 联动)。

| 信号 | 阈值类型 | 默认 | 备注 |
|---|---|---|---|
| **D3 过载** | token + 主题离散度 | token 2000 / 离散度由 LLM 写后 inline 判 | **唯一驱动 split 的阈值**(详 §4) |
| **D1 断链** | 0 容忍 | 任意 1 条断链 → 告警 | 修复策略简单(就地删 wikilink) |
| **D10 provenance 断裂** | 0 容忍 | 任意 1 条断裂 → 严重告警 | I 不变量 |

D3 阈值作为 `vault.yaml` 配置项(opinionated default,reme 核心提供机制不写死阈值),消费层可改;dogfooding 后调优。token 阈值起点 2000(对应"约 5 个独立子主题"的常见过载点),首版可调。

---

## 4. D3 触发模型(已收敛)

**决策**:**写后立即检测,无 watcher 抽象,无 batch 窗口** —— 每次 G\* update / split 写 body 成功后,**inline** 在同一 job 内跑 D3:token 阈值 + LLM 离散度判定 → 必要时 enqueue split job(异步,走 §5 CAS 队列)。

```
G* / split 写 body 成功(CAS 通过)
  └─ if len(body) > T:
        └─ LLM 判离散度
            └─ if is_overloaded:
                  └─ enqueue split job (FIFO, CAS-protected)
  └─ return
```

**协议**:
- token 阈值默认 `2000`(§3 已定,`vault.yaml` 可配)
- 离散度 prompt 输出 `{is_overloaded: bool, suggested_clusters: [...]}`(若 overloaded 直接供 split job 吃,不重判)
- 启动无全扫(避免长启动);新写入立即检测覆盖增长路径;历史遗留过载随下次 update 自然检出
- 无 dirty 标 / 无 dirty 集合 / 无后台 worker —— D3 是写路径的合成函数

**为什么 inline**:反应即时(不等下一次 ingest);实现最简(无批处理窗口 / dirty 状态 / 独立 worker);LLM 判定成本可接受(大多写入 < T 不触发,触发后 split 切小后续不再越界);不引入 watcher = 少一层部署/监控。

**已排除**:定时 cron tick(静止 vault 浪费扫描)/ ingest-after batch(引入 dirty 集合)/ 独立 L2 watcher worker(多余部署层)。

**演进路径(M1+)**:若 inline LLM 阻塞 G\* 时延成问题 → D3 改为 fire-and-forget enqueue;若同节点重复触发 LLM 成本高 → 加节点级 body hash 缓存。

---

## 5. CAS 写入协议(共享基础设施)

**位置说明**:CAS 是 G\* update(`auto_dream_design.md` §2.1)、M split(本文档 §1)、auto-link L1 写回(`auto_link_design.md` §1.2)**三方共用**的写入协议。归在本文档是因为 maintain 是 digest 的"组织 / 运行时"端,运行时机制(检测 / 触发 / 写入)集中在一处方便对照。

**决策**:**并行决策 + 乐观冲突重做(CAS)** —— 所有 G\* / split / auto-link L1 决策并发跑,写入前用 body 版本戳(hash / mtime)做 CAS 比对;变了就丢弃 planned body 重做。无锁,无 ingest 级互斥。冲突率低 + E-1 / E-2 守恒校验顺手承担 race 兜底,无需新基础设施。

**协议(单个写入调用)**:
1. **读 + 记戳**:读 subject body → `version_stamp = sha256(body) | mtime`
2. **决策**:LLM 看候选池 → 决定 create / update / drop / split / additive-link;产 planned new_body
3. **CAS 写入**:重读 body 比 version_stamp
   - **未变**:跑 E-1 / E-2 守恒校验 → 通过则 atomic write(write-temp + rename)→ done
   - **已变**:丢弃 planned new_body,带最新 body 重走 step 1
4. **守恒校验失败**:走 `auto_dream_design.md` §1.5.5 既有重试路径(LLM 重试一次,二次失败拒写 + audit)
5. **重做次数上限**:CAS-冲突重做最多 3 次;超出 → 跳过候选 + audit log(避免活锁)

**create 路径 race**:两个 G\* 都决定 `create digest/auth/jwt-rotation.md` → atomic create(`O_CREAT | O_EXCL`)只让一个赢;输者拿 EEXIST → 重走 step 1(此时大概率改判 update)。

**适用范围**(全部走同一套 CAS):同 ingest 内 N 个候选并发 / 跨 ingest job 并发 / 后台 split 与前台 G\* 命中同节点(split 同样走 CAS)/ auto-link 写回(`auto_link_design.md` §1.2)。

**不解决的**:高冲突 workload(同概念被反复 ingest)→ 重做上限触发后 audit;跨进程并发(多 reme 实例同 vault)→ 不在 M0,需 fs lock(M1+)。

---

## 6. split 时的 provenance 处理(已收敛)

**坍缩到 E-2 合计守恒** —— provenance 是 body 内联 wikilink(`auto_dream_design.md` §3.9),split 时跟其它 body 边完全同形:LLM 把 parent body 拆成 parent overview + N children,provenance wikilink 跟着对应内容段自然分配;机械层 outbound 合计守恒校验保证 `(parent_new ∪ ∪children_outbound) ⊇ parent_old`,旧 provenance 不可能丢。无需专门的 provenance 分配逻辑或"全部复制到 child / parent 保留全量"等特殊策略 —— LLM 按"哪个 child 谈到了哪段上游就带走哪条 provenance"自然处理。

---

## 7. G\* / split / auto-link L1 时序(已收敛)

时序由 §4 / §5 与 `auto_dream_design.md` §3.10 共同规定,这里给最小汇总:

- **G\* 调用本身同步** —— material 进来就走 G\* 决策(召回 + LLM 终判)+ CAS 写入(§5)
- **D3 检测 inline** —— G\* / split 写完 body 顺手跑 token 阈值 + LLM 判离散度(§4),无 tick / batch / watcher
- **split 异步** —— D3 触发后 enqueue split job 进 §5 CAS 队列,跟其它 ingest / split job FIFO 共享,异步消费;**不阻塞 G\* return**
- **auto-link L1 异步** —— 写入成功后 enqueue auto-link L1 job(`auto_link_design.md` §1.4),与 split job 同 CAS 队列、FIFO 共享;不阻塞 G\* return

检测延迟 ≈ 0(inline);split 执行延迟 ≈ 队列等待时间(typically 数秒~数十秒);新建 / update 节点不必等 split 完成,体验连续。

---

## 8. maintainer 的人 / agent 后门(暂缓 — 非底层)

消费层 / SDK 接口问题,不影响底层机制。底层只需保证 split / rename / delete 等 op 走同一套 §5 CAS + 守恒校验链路:F-3 仍成立(maintainer 自动路径只做 split);merge / dissolve / re-edge 在底层**不存在**(无对应机械算子)。后门接口形态推迟到 SDK 阶段再定。

---

## 9. 与 dream 模型的引用关系

本文档复用 dream 定义的底层模型,所有具体规则在 `auto_dream_design.md` 中:

| 引用 | 来源 |
|---|---|
| wikilink 基础语法(`[[path.md\|alias]]` / predicate) | `auto_dream_design.md` §1.4 |
| 节点 / 边模型 | `auto_dream_design.md` §1.5 / §1.5.1 |
| F-invariants(F-1..F-11) | `auto_dream_design.md` §1.5.4 |
| 边守恒 E-1 / E-2 / E-3 | `auto_dream_design.md` §1.5.5 |
| 路径即 ID / rename | `auto_dream_design.md` §1.3 |
| anchor 不引入 | `auto_dream_design.md` §1.4 / §3.13 |
| provenance 载体形态 | `auto_dream_design.md` §3.9 |
| G\* 行为 | `auto_dream_design.md` §2.1 |

---

## 10. 下一步

1. **M split step 实现** —— D3 触发 → 候选 → split prompt → 写入 + E-2 守恒(§1 / §4 / §5)
2. **D 检测信号实现清单**(D1 断链 / D3 写后 inline / D10 provenance 哪些已就绪 / 缺哪些)—— §2
3. **D3 阈值配置**(`vault.yaml` 中 D3 token / 离散度阈值)—— §3
4. **CAS 写入框架** —— per-path body version_stamp + CAS 写入 + EEXIST create race + 重做上限 + audit;对外暴露给 dream G\* / auto-link L1 复用 —— §5
5. **后门 SDK 接口形态**(暂缓 M1+)—— §8

实现进入 `reme4/steps/jobs/` 与 `reme4/file_graph/` 时,本文档与 `auto_dream_design.md` / `auto_link_design.md` 共同作为契约依据。
