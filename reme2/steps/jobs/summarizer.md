# Summarizer 设计方案 (v4)

## 1. 定位

**Summarizer = Agent 的"任务工作区"周期同步器**

- 触发方:宿主 agent;触发时机:上下文/对话轮次达阈值(agent 自决)
- 双重产出:
  1. **持久化**:把 in-progress 任务沉到 vault 的 workspace folder(folder + summary note + 任意类型 materials)
  2. **回灌 context**:把完整的 summary note 内容随结果一起返回,agent 可在 compact / new-session 时直接注入新 context
- 不做:不压缩 agent context、不 distill、不动 events/topics、不感知阈值

对照 `structure.md`:summarizer 写的是 **Warm Summary**,且**主动把 Warm Summary 反哺 Hot Context**。

---

## 2. 工作区物理形态

```
<vault>/daily/<YYYY-MM-DD>/<slug>/
├── <slug>.md                # workspace summary note (即 folder note)
├── <material1>.md           # 文本材料
├── <material2>.pdf          # 任意类型 (pdf / doc / png / json / csv ...)
└── ...
```

- `<slug>` = task title 的 kebab-case ≤ 60 字符;同一任务在同一天 slug 必须稳定
- `<slug>.md` 与父目录同名 — 即是 **summary note**,也是 reme L1 视角的 **folder note**(同一份文件,两个名字)
- Materials 不限文件类型

> **术语统一**:本方案中 "summary note" 与 "folder note" 指同一份 `<slug>.md` 文件。前者是 summarizer 视角的语义名(内容是 workspace summary),后者是 reme L1 视角的结构名(同名约定下的目录索引)。

---

## 3. Summary note 结构

```markdown
---
title: <一行任务标题>
description: <2-3 句:任务是什么、为什么>
status: active | completed
created: <YYYY-MM-DD>
updated: <YYYY-MM-DD>
inherits: [[daily/<earlier-date>/<earlier-slug>/<earlier-slug>]]   # 可选,仅 INHERIT 时
---

## Objective
<长期稳定目标 — 创建时定一次,scope 变化才改>

## Plan
<整体计划/方法/分阶段路线 — 滚动维护,wholesale 重写>

## Progress
- <YYYY-MM-DD HH:MM> <一条进度>
<append-only>

## Findings
- <关键发现/事实/数据点 — next-session 必须知道的"已确认结论">
<append-only>

## Decisions
- <关键决策与原因>
<append-only>

## Next
- [ ] <todo>
- [x] <done todo>
<wholesale 替换>

## Materials
- [[<material1>]] — <一句话说明>           # md 用 wikilink (folder note 短链)
- [<material2.pdf>](material2.pdf) — <说明>  # 非 md 用 markdown 链接 + 显式扩展名
<union 去重>
```

字段哲学:
- **Append-only**:Progress / Findings / Decisions
- **Wholesale-replace**:Plan / Next
- **Set-once**:Objective(scope shift 才改)
- **Union**:Materials

---

## 4. Materials 来源(双轨)

| 来源 | 谁写 | 落盘方式 | 进 Materials list |
|---|---|---|---|
| **Agent 主动落盘** | Agent 在工作中通过 `write` / `upload` 把片段、文档、产物存到 `daily/<date>/<slug>/` | 任意工具 | summarizer 触发时 `list daily/<date>/<slug>/`,把所有非 index 文件加进 Materials |
| **Summarizer 自动抽** | Summarizer 从对话里抽关键文本片段(报错堆栈、用户原话、产出代码块、调研结论)| `write` 成 `daily/<date>/<slug>/<auto-name>.md` | 写完 append 到 Materials |

**UPDATE 路径必须先 `list`** workspace folder,把 agent 已放进去的新文件补进 Materials,避免丢失。
非 md 材料只能由 agent 通过 `upload` 放,summarizer 不生成非 md 文件。

---

## 5. 决策树

```
[1] messages 非空? ─否─▶ SKIP, used_llm=false, return
        │是
        ▼
[2] LLM 读 messages: 是不是 in-progress 多步任务?
   否 ─▶ "[SKIP]", no tool calls
        │是
        ▼
[3] LLM 推断稳定 task title → slug
        │
        ▼
[4] list daily/<today>/
    ┌── 今天已有 <slug>/ ─▶ UPDATE
    │       (a) read daily/<today>/<slug>/<slug>.md
    │       (b) list daily/<today>/<slug>/        (含非 md)
    │       (c) merge 各 section:
    │             Objective:  保留
    │             Plan:       wholesale 重写
    │             Progress:   append (旧 + 新)
    │             Findings:   append
    │             Decisions:  append
    │             Next:       wholesale 替换
    │             Materials:  union(旧 list, 新文件 list, 自动抽片段)
    │       (d) [可选] 写关键片段成新 md material (防冲突)
    │       (e) write 新 index (overwrite=True 覆盖旧 index 是预期)
    │
    ├── 今天没有, 扫近 N (默认 7) 天 daily/ 找 status=active 同 title ─▶ INHERIT
    │       (a) property:read 候选 index 看 title/status
    │       (b) read predecessor index → 拷 Objective + Plan
    │       (c) [可选] 写关键片段 (防冲突)
    │       (d) write 新 index:
    │             Inherits:    [[daily/<earlier-date>/<earlier-slug>/<earlier-slug>]]
    │             Objective:   拷
    │             Plan:        拷
    │             其他 sections: 全新
    │             Materials:   仅自己生成 (旧 materials 通过 Inherits 链回去)
    │       (e) predecessor 的 status 不动
    │
    └── 没找到 ─▶ CREATE
            (a) [可选] 写关键片段 (防冲突)
            (b) write 新 index (各 section 全新)

        ▼
[5] 完成态判定:
    - 默认 status=active
    - 仅当对话有非常明确的"任务结束/已交付"信号 → completed
    - 翻转能力保留, prompt 写死保守判定

        ▼
[6] LLM 输出最终回复 (一行 summary):
    "<created|inherited|updated> daily/<date>/<slug>/<slug>.md (+M materials)"
    或 "[SKIP]"

        ▼
[7] Step 在 LLM 完成后, 用 file_store 直接 read 出
    刚写的 daily/<date>/<slug>/<slug>.md 完整内容,
    填进 SummarizerResult.summary 字段
```

---

## 6. Toolkit (LLM 可用)

复用现有 6 个 crud,无需新增:

| 工具 | 用途 |
|---|---|
| `list(path, recursive)` | 扫今天目录、扫 workspace folder(含非 md)、扫 inherit 候选 |
| `stat(path)` | **写前防冲突检查** |
| `read(path)` | UPDATE 读旧 index;INHERIT 读 predecessor |
| `write(path, content, overwrite)` | 写 index;写自动抽 material(默认 `overwrite=False`)|
| `property:read(path)` | 扫 inherit 候选只看 frontmatter |
| `property:update(path, **fields)` | 翻 status / 改 updated 不重写正文 |

`upload` 不在 toolkit 里 — summarizer 不拷非 md 材料。

---

## 7. Step 参数 & 输出契约

### 7.1 `__init__` 参数

```python
class Summarizer(BaseStep):
    def __init__(
        self,
        toolkit: Toolkit | None = None,
        console_enabled: bool = False,
        timezone: str | None = None,
        inherit_window_days: int = 7,        # INHERIT 扫描窗口
        **kwargs,
    ):
        ...
```

### 7.2 输出契约(扩展)

```python
class SummarizerResult(BaseModel):
    used_llm:  bool
    applied:   list[dict]          # 成功的 tool calls
    failed:    list[dict]          # 失败的 tool calls
    skipped:   bool                # True = LLM 判定非任务
    actions:   str                 # LLM 一行 action 陈述

    # —— 服务 agent 上下文管理的核心字段 ——
    workspace: str | None = None   # daily/<date>/<slug>/  (folder 相对路径,带尾 /)
    summary:   str | None = None   # summary note **完整 markdown 内容**
                                   # (含 frontmatter + 全部 sections)
                                   # SKIP 或 失败时为 None

success = len(failed) == 0
```

**`summary` 字段是关键**:agent 在 compact context / 新 session 启动时,可直接把 `result.summary` 作为 system context 注入,不必再发 read 请求。

填充时机:LLM 完成所有 tool call 后,Step 在 `execute()` 末尾用 `file_store` **直接** read 出 note 文件(从 `actions` 行正则提取 path),确保拿到的是 agent 刚刚写完的最新版本。

> **Input/Output 字段对齐**:`workspace` 一词在两边语义统一 — 输入时是 caller 给的 hint(任务标题或 `daily/.../` 路径),输出时是确定的 folder 相对路径。

---

## 8. 冲突管理(LLM 自管)

`write` 默认 `overwrite=False`(已支持,见 `reme2/steps/crud/write.py:55`)。LLM 协议:

| 场景 | LLM 应做 |
|---|---|
| **写 index `<slug>.md`** | UPDATE 路径:已 `read` 确认是同一任务 → 显式传 `overwrite=True`<br>CREATE / INHERIT:`stat` 不存在再写;意外存在 → 加后缀消歧 |
| **写自动抽的 material** | 名字 LLM 自取(描述性);`stat` 检查;若存在 → 改名(如 `auth-error-2.md`)再写;不允许 silent overwrite |
| **agent 已写过同名** | 同上 — `stat` 拦下,LLM 改名 |

---

## 9. 关键设计决策汇总

| # | 决策 | 选定 |
|---|---|---|
| D1 | workspace 形态 | folder + summary note(= folder note) + 同目录 materials |
| D2 | 术语 | "summary note" / "folder note" = 同一份 `<slug>.md`,本方案统一 |
| D3 | summary note 字段 | Objective / Plan / Progress / Findings / Decisions / Next + Materials + (Inherits) |
| D4 | append-only | Progress / Findings / Decisions |
| D5 | wholesale-replace | Plan / Next |
| D6 | Materials 来源 | 双轨 — agent 主动 + summarizer 自动抽(仅 md) |
| D7 | Materials 文件类型 | 任意 — md 用 wikilink,非 md 用 markdown 链接 + 扩展名 |
| D8 | INHERIT 扫描窗口 | 参数化 `inherit_window_days`,默认 7 |
| D9 | completed 翻转 | 保留能力,prompt 写死保守判定 |
| D10 | Predecessor status | INHERIT 时不动 |
| D11 | Material 自动文件名 | LLM 自取(描述性) |
| D12 | 冲突管理 | `write` 默认 `overwrite=False`,LLM 自己 `stat` 检查并改名 |
| D13 | 触发器 | agent 自决,summarizer 不感知阈值 |
| D14 | 与 ingestor 关系 | summarizer 写 daily/ workspace(warm);ingestor 写 events/topics(cold)— 解耦 |
| D15 | 输出含完整 note | `SummarizerResult.summary` 携带刚写完的 summary note 全文,服务 agent 上下文管理 |
| D16 | summary 字段填充 | Step 在 LLM 完成后用 `file_store` 直读,不依赖 LLM 自己回传 |
| D17 | input/output 字段对齐 | `workspace` 一词在 input(hint) 与 output(folder path) 共用,`actions` 字段承载 LLM 一行陈述 |

---

## 10. 改动清单

### `reme2/steps/jobs/summarizer.py`
1. `__init__` 加 `inherit_window_days: int = 7`
2. `execute()`:
   - 输入字段:`task_hint` → `workspace`(语义统一)
   - `prompt_format("user_message", ...)` 多传 `inherit_window_days` + `workspace`
   - LLM 完成后,从 `actions`(原 `agent_summary`)解析出 note path(正则匹配)
   - 用 `path_resolver.to_absolute(file_store, note_path).read_text()` 拿完整内容
   - 填进 `SummarizerResult.summary` / `workspace`
   - SKIP / 失败 → 两个字段 None
3. `SummarizerResult` 字段重命名:`agent_summary`→`actions`,`workspace_path`→`workspace`,`note`→`summary`,删除 `note_path`

### `reme2/steps/jobs/summarizer.yaml`
完全重写 `system_prompt` + `user_message`:
- 路径模板改 folder 形态 `daily/<date>/<slug>/<slug>.md`
- 字段扩展 Objective / Plan / Progress / Findings / Decisions / Next + Materials
- Materials 引用规则:md wikilink,非 md markdown link + 扩展名
- 决策树明确 UPDATE / INHERIT / CREATE
- INHERIT 窗口用 `{inherit_window_days}` 占位
- Workspace hint 用 `{workspace}` 占位 (替代旧的 `{task_hint}`)
- 冲突管理协议写明 `stat` 先行 + 改名
- 完成态保守判定
- 输出格式 `<action> daily/<date>/<slug>/<slug>.md (+M materials)` 或 `[SKIP]`
- 强调:LLM 输出的 path **必须用相对 vault 的形式**,Step 后续会据此 read 完整 note

### note path 提取协议(实施细节)
让 LLM 在 actions 行严格输出 `<action> <relative-path> (+M materials)`,Step 用正则 `r"daily/\d{4}-\d{2}-\d{2}/[^/\s]+/[^/\s]+\.md"` 提取 path。SKIP 行不提取。提取失败 → `summary=None` 但不算 step 失败(只是回灌 context 失败,持久化已完成)。
