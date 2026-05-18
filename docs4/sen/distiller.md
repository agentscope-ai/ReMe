# Distiller 设计方案 (v3)

## 1. 角色定位

**Distiller = 知识蒸馏器**:从 Synchronizer 写下的 `daily/<date>/<slug>/`
工作区,提取实体/概念/论断/方法,沉淀到 `knowledge/<slug>/`,
让主 Agent 通过 Retriever (search / graph:traverse) 反哺 Hot Context。

冷写侧:与 Synchronizer (热写,daily 工作区) 互补,与 Retriever (读)
形成"对话 → 工作区 → 知识库 → 召回"闭环。

| 维度 | Synchronizer (热) | **Distiller (冷)** | Retriever (读) |
|---|---|---|---|
| 触发 | 每次对话/会话 | 显式调用 | 主 Agent 查询时 |
| 输入 | 对话 messages | `daily_paths` 列表 | query string |
| 输出 | `daily/<date>/<slug>/` | `knowledge/<slug>/` | 检索结果 |
| 改 daily? | 写 daily 本身 | 只翻 `status: distilled` | - |

---

## 2. 与 reme4 现状的对齐

- 路径用 reme4 已配置的 `application_config.knowledge_dir = "knowledge"` (单数)
- Frontmatter 用 `lint:schema` 已要求的 4 key (`title/lifecycle/scope/source/role`) + 现有 `_VALID_STATUS = active/distilled/archived`
- **不新增** `schema/memory_axes.py` 或 preset Python 类 — 协议在 `protocol.md` 用文字承诺
- **不内嵌** schema validator — 让 LLM 写完后主动调 `lint:schema` 自查

---

## 3. 物理形态

```
<vault>/
  daily/<YYYY-MM-DD>/<slug>/<slug>.md       Synchronizer 写
  daily/<YYYY-MM-DD>/<slug>/<material>.md   sibling materials
                                            ↓ Distiller 读
  knowledge/<slug>/<slug>.md                Distiller 写 (folder note)
  knowledge/<slug>/<material>.md            (可选,蒸馏过程拆出的子事实)
```

`knowledge/<slug>/<slug>.md` 是 folder note,符合 reme4 path_resolver
的 folder-note 规则,主 Agent 写 `[[<slug>]]` 即命中。

---

## 4. Frontmatter Schema (4 轴,文字约定)

详见 `protocol.md`。要点:

| key | 取值 |
|---|---|
| `lifecycle` | `streaming` (会衰减) / `evolving` (持续编辑) / `frozen` (不可改) |
| `scope` | `instance` (具体) / `class` (抽象) |
| `source` | `auto` (机器) / `curated` (人/LLM) / `derived` (推算) |
| `role` | `profile` / `concept` / `claim` / `method` / `reference` / `observation` / `question` / `fundamentals` |

knowledge/ 常见组合:

| 用途 | role | lifecycle | scope | source |
|---|---|---|---|---|
| 人/项目/工具 | profile | evolving | class | curated |
| 抽象概念 | concept | evolving | class | curated |
| 论断 | claim | evolving | class | curated |
| 做法 | method | evolving | class | curated |
| 资料 | reference | frozen | instance | auto |

---

## 5. R-M-W 三段

### Read
1. 取 `daily_paths` (必须显式传入,**不自动扫描**全 vault)
2. 对每个 daily,只 read `<slug>.md` (summary note);material 文件由 LLM 按需 read
3. 不预扫现有 knowledge — LLM 在 M 阶段按需 lookup (`list`/`graph:traverse`/`tags:list`)

### Modify (LLM Agent + Toolkit)
LLM 接到打包好的 daily 内容 + protocol.md + caller hint,执行决策树:

```
for each candidate (entity / concept / claim / method) in daily:
    lookup [[candidate]]   # via list / graph:traverse / tags:list
    ├ no hit            → CREATE knowledge/<slug>/<slug>.md
    ├ exact role match  → UPDATE: read → merge → write overwrite=True
    ├ role mismatch     → CREATE new knowledge with correct role
    └ only relation     → LINK: 追加 wikilink 到现有 knowledge body
```

特别约束:
- 写完后调一次 `lint:schema` 拿违规清单,用 `property:update` 修
- 所有写成功的 daily 翻 `property:update status=distilled` (若 `flip_status=True`)

### Write
- LLM 直接调 `write` / `property:update`
- **Distiller 不内嵌校验**,`lint:schema` 兜底
- IO 失败进 `DistillResult.failed[]`,不阻断后续 op

---

## 6. Toolkit (LLM 可见)

| 类 | 工具 |
|---|---|
| 读 | `list` `read` `stat` `property:read` |
| 搜 | `tags:list` `tags:stat` (search_step 暂无 tool method 绑定) |
| 图 | `graph:traverse` |
| 质检 | `lint:dangling` `lint:collisions` `lint:schema` |
| 写 | `write` `property:update` `property:delete` |

`upload` / `download` / `lint:orphans` / `search_step` 不暴露
(`search_step` 是 step 但没暴露 tool method,等需要时再加 wrapper)。

---

## 7. 输入/输出契约

### Input (RuntimeContext)

```python
daily_paths:  list[str]                  # 必填; 无默认扫描
hint:         str = ""                   # caller 偏好 (e.g. "重点关注 auth")
flip_status:  bool = True                # 处理完是否翻 daily 为 distilled
```

无默认扫描 = 成本可控;批量调度由外层 wrapper step (如果需要) 负责。

### Output (DistillResult)

```python
class DistillResult(BaseModel):
    used_llm:          bool
    skipped:           bool       = False  # daily_paths 为空 / 无 LLM / 全失败

    daily_read:        list[str]  = []     # 实际读到的 daily

    knowledge_written: list[dict] = []     # [{path, size}] — 写在 knowledge/ 下,CREATE+UPDATE 合并
    links_added:       list[dict] = []     # [{src, dst, predicate}] — 暂留空,等结构化 audit
    status_flipped:    list[str]  = []     # daily paths flipped to distilled

    failed:            list[dict] = []     # tool call 失败
    summary:           str        = ""     # LLM 一段话总结
```

字段命名用单数 `knowledge_*` 对齐 `knowledge_dir`。
`knowledge_written` 一栏合并 CREATE+UPDATE — audit 不携带"是否覆盖"信号,
caller 需要可借文件系统二次区分。

---

## 8. 闭环

```
对话 ──► Synchronizer ──► daily/<date>/<slug>/  (status=active)
                            │
                            ▼
                    Distiller.execute(daily_paths=[...])
                       Read → Modify (LLM) → Write
                            │
              ┌─────────────┼─────────────────────────┐
              ▼             ▼                         ▼
        CREATE knowledge  UPDATE knowledge       LINK (追加 wikilink)
              │             │                         │
              └─────────────┼─────────────────────────┘
                            ▼
                       lint:schema (LLM 自查)
                            │
                            ▼
              property:update status=distilled (闭环, flip_status=True 时)
                            │
                            ▼
              主 Agent ──► search / graph:traverse ──► 反哺 Hot Context
```

---

## 9. 降级路径

**无**。蒸馏强依赖 LLM (state-align / merge / 决策都不可降级)。
无 `as_llm` 时:
- `DistillResult.used_llm=False, skipped=True`
- `failed[]` 写一条 `{op: "distill", error: "no as_llm configured..."}` 提示 caller

旧 v1 的 `_degraded` (无 LLM 时按 `target_path` mechanical 落盘) 删除 —
单点写入是 `crud:write` 的职责,不是 distiller 的。

---

## 10. 与 v1 (旧 ingestor) 的差异

| 维度 | v1 (Ingestor) | v3 (Distiller) |
|---|---|---|
| 命名 | 数据采集语义 | 知识蒸馏语义 |
| 角色 | SSOT 写入入口 | 知识蒸馏器 |
| 输入 | `content` + `target_path` | `daily_paths: list[str]` |
| 输出域 | 任意单文件 | `knowledge/` 域 + flip daily status |
| schema 校验 | 无 | LLM 自查 `lint:schema` |
| 降级路径 | mechanical 写 target_path | 无 (强依赖 LLM) |
| Toolkit | crud + property + graph + tags + lint | 同 + `lint:schema` 暴露 |
| 闭环 | 无 | flip daily `status=distilled` |

---

## 11. 不在本次范围

- **Maintainer** (merge / split / decay) — 独立服务,按 cron / 阈值触发
- **`schema/memory_axes.py`** preset Python 类 — 待 vault 域足够稳定后再固化
- **Synchronizer** — 不动,只读它的输出
- **自动调度** — distiller 不感知阈值,caller 决定何时/对哪些 daily 调用

---

## 12. 改动文件

```
reme4/steps/jobs/
├── distiller.py    R-M-W 三段, DistillResult
├── distiller.yaml  system_prompt 角色 + {protocol}/{daily_blob}/{hint}
├── distiller.md    本文档 (设计 SSOT)
└── protocol.md     4 key + 决策树 + wikilink + flip 协议
```

tests:
```
tests4/unittest/test_jobs_steps.py
  - test_distiller_skipped_when_no_dailies
  - test_distiller_skipped_when_no_llm
  - test_distill_result_default_fields
  - test_distill_result_success_property
  - test_classify_audit_entry_buckets
  - test_pack_daily_*
```
