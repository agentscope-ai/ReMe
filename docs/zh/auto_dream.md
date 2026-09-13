# Auto Dream

`auto_dream` 是 ReMe 的 daily 到 digest 的长期记忆沉淀流程。它默认扫描目标日期及前一天的 daily 输入，只处理相对上次 dream
发生变化的文件，从整个扫描窗口中抽取少量高价值 memory units，并整合进 `digest/`。

它消费的 daily 输入通常来自 [Auto Memory](./auto_memory.md) 和 [Auto Resource](./auto_resource.md)。`digest/`、Sources 章节
和 wikilink 的文件语义见 [Memory as File](./memory_as_file.md)；Integrate 阶段的链接策略详见 [Auto Link](./auto_link.md)。
主动发现是独立流程，见 [Proactive](./proactive.md)。

## 配置入口

默认配置在 `reme/config/default.yaml`：

```yaml
auto_dream:
  backend: base
  parameters:
    date:
      type: string
      default: ""
    hint:
      type: string
      default: ""
    scan_days:
      type: integer
      default: 2
    max_units:
      type: integer
      default: 5
  steps:
    - backend: dream_extract_step
      file_catalog: dream
      scan_days: 2
      max_units: 5
    - backend: dream_integrate_step
    - backend: dream_finish_step
      file_catalog: dream
    - backend: auto_tag_step
```

参数含义：

| 参数                   | 作用                                                              |
|------------------------|-------------------------------------------------------------------|
| `date`                 | 要处理的日期，格式为 `YYYY-MM-DD`。为空时使用应用时区中的今天。   |
| `hint`                 | 调用方给抽取和整合阶段的额外指导。                                |
| `scan_days`            | 以 `date` 结尾的最近日期窗口；默认扫描 2 天，最小为 1。           |
| `max_units`            | 一次最多抽取多少个可复用 unit；默认 5。                           |

## 输入和输出

输入来自以指定日期结尾的最近 `scan_days` 天 daily markdown。例如 `date=2026-06-20`、`scan_days=2` 时会扫描：

```text
daily/2026-06-19.md
daily/2026-06-19/**/*.md
daily/2026-06-20.md
daily/2026-06-20/**/*.md
```

Auto Dream 只扫描 Markdown 日期索引和笔记，不读取 proactive 状态或 `interests.yaml`。

主要输出有三类：

| 输出                           | 说明                                              |
|--------------------------------|---------------------------------------------------|
| `digest/procedure/*.md`        | 方法、流程、runbook、可执行经验。                 |
| `digest/personal/*.md`         | 用户、团队、项目相关的偏好、事实、长期上下文。    |
| `digest/wiki/*.md`             | 通用知识、概念、观察、决策先例。                  |
| `metadata/file_catalog/dream*` | dream 专用 catalog，用于判断 daily 输入是否变化。 |

## 四个阶段

### 1. Extract

`dream_extract_step` 做三件事：

1. 刷新扫描窗口内每天的索引页 `daily/<date>.md`。
2. 扫描这些日期的索引页和 `daily/<date>/**/*.md`，与 `file_catalog: dream` 中记录的 mtime 对比。
3. 只把 changed files 一起交给 LLM，全局抽取结构化 memory `units`。

`units` 是准备沉淀进 digest 的长期记忆单元，包含 `name`、`bucket`、`summary`、`paths`。一次最多返回 `max_units`
个，抽取器会优先合并指向同一抽象的跨文件证据，并丢弃短暂提及、逐文件摘要和缺少复用价值的弱候选。`bucket` 只允许
`procedure`、`personal`、`wiki`；未知值会路由到 `wiki`。

如果没有 changed files，Extract 会成功返回空 units；Integrate 随后没有 unit 可处理，Finish 仍会正常汇总 catalog。
如果有变化但没有配置 LLM，Extract 会失败，因为抽取依赖 LLM。

### 2. Integrate

`dream_integrate_step` 对每个 unit 独立调用 Agent，将一个 unit 整合成一个 digest 节点。它会给 Agent 暴露这些工具：

```text
node_search, read, frontmatter_read, write, edit, frontmatter_update
```

这一阶段承担 `auto_link` 的核心职责：先用 `node_search` 在 digest 节点级召回相似或相关节点，再判断是新建还是更新，最后把来源和相关
digest 节点写成 wikilink。具体召回、去重和写边规则见 [Auto Link](./auto_link.md)。

Extract 已经承担“是否值得长期记住”的过滤，因此 Integrate 不提供 `SKIP` 动作：每个进入本阶段的 unit 都应落到且只落到一个
digest 节点。新增与更新都必须保留来源，并把相关 digest 链接写进有上下文的句子；不能只写裸 Wikilink 或独立的关系字段。

整合动作只有四种：

| 动作          | 含义                                           |
|---------------|------------------------------------------------|
| `CREATE`      | 没有相同抽象，创建新的 digest 节点。           |
| `CORROBORATE` | 同一记忆再次出现，追加来源或强化表述。         |
| `REFINE`      | 新材料补充了边界、步骤、前提、适用范围或细节。 |
| `CORRECT`     | 新材料修正了旧节点的错误、遗漏或冲突。         |

Integrate 成功的 unit 会记录到 `integrate_results`；失败的 unit 会进入 `failed_units`，其来源路径会进入 `failed_paths`。
Finish 阶段不会 checkpoint 失败路径，保证下次还能重试。

### 3. Finish

`dream_finish_step` 负责收尾：

1. 将成功处理的 changed paths 写入 `file_catalog: dream`。
2. 将扫描窗口内每个已刷新的 day-index 页也写入 catalog。
3. 如果有 upsert 或 delete，持久化 dream catalog。
4. 返回包含 scanned、changed、integrated、checkpoint 等计数的摘要。

Auto Dream 不读取或写入 proactive 状态和 `interests.yaml`。这些文件由 proactive refresh writer 链路负责，
见 [Proactive](./proactive.md)。

失败路径不会被 checkpoint。这样下一次 `auto_dream` 仍会把它们视作 changed input，直到整合成功。

### 4. Auto Tag

Finish 后，`auto_dream` 和 `dream_cron` 都会通过 `auto_tag_step` 为本轮整合实际新增或修改的 Markdown digest 文件打标，
包括 Agent 异常后恢复的落盘结果。同一文件被多次写入时只打标一次。该 Step 复用 [Auto Memory](./auto_memory.md) 的请求级
`changes` 协议，将实体标签写入配置的 frontmatter 字段，默认为 `memory_tags`。Dream 不为未变化文件或 daily 来源笔记打标。

打标诊断记录在 `metadata.auto_tag`。单文件打标失败保留 dream 原有的摘要、成功状态和 checkpoint 决策；后续没有文件变化的
调用不会自动重试失败的打标。标签索引通过现有文件 watcher 异步更新。

## 运行方式

CLI：

```bash
reme auto_dream date=2026-06-20
```

带调用提示：

```bash
reme auto_dream date=2026-06-20 hint="优先沉淀工程决策和长期偏好"
```

覆盖默认扫描窗口和 unit 上限：

```bash
reme auto_dream date=2026-06-20 scan_days=3 max_units=8
```

也可以在配置中把同一组 step 放进 `cron` job，例如每天凌晨运行：

```yaml
jobs:
  daily_auto_dream:
    backend: cron
    cron: "30 3 * * *"
    steps:
      - backend: dream_extract_step
        file_catalog: dream
      - backend: dream_integrate_step
      - backend: dream_finish_step
        file_catalog: dream
      - backend: auto_tag_step
```

## 关键边界

`auto_dream` 只消费 daily 输入，不改写 daily 正文。daily 是事实和现场记录，digest 才是抽象后的长期记忆层。

`digest` 不是原文复制。正文应保留可复用抽象，Sources 章节用带上下文的完整句子指回来源，例如
`该决策记录在 [[daily/<date>/decision.md]] 中。`链接写法遵循
[Memory as File](./memory_as_file.md) 中的 workspace-relative wikilink 语义。

`auto_dream` 不凭空生成总览。只有 daily 输入中确实出现、并被抽取为 memory unit 的内容，才会进入 digest。

完整流程依赖 LLM 完成 Extract、Integrate 和 Auto Tag。
