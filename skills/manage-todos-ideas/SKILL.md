---
name: manage-todos-ideas
description: 通过 ReMe 的 todo、idea 工具记录项目待办和灵感，追加进展、完成或作废任务，归档或恢复想法，并建立 Idea 与 Todo 的关联。适用于随手记下事情、卸载临时思路和继续上次的工作。
---

# 管理待办和想法

使用 `todo` 和 `idea`，参数直接放在顶层。`proactive` 配置已提供两个工具；复用当前服务，CLI 对应
`reme todo action=...` 和 `reme idea action=...`。返回示例展示业务结果 `answer`；ID 和时间均为示例，
实际调用使用工具返回的 ID。

## 对话中的判断

- 用户的灵感、备选方案和待讨论问题记为 Idea；明确要推进的事情记为 Todo。每条保留一句清晰的项目事项。
- 用户指定的项目直接使用；未指定时继承来源 Idea，否则使用当前明确的项目；没有项目上下文时省略，
  由工具默认归入“随笔”。
  期限没提到就省略，不要为了补齐字段反问。只有无法确定要更新哪个条目时才澄清。
- 修改已有条目前，按项目或关键词 `list`，必要时 `get` 核对。相同事情已有记录时优先更新，不重复创建。
- 根据用户明确陈述或 Agent 实际完成且可验证的结果更新进展。部分完成保留原目标，追加已做和剩余事项，
  设为 `partial`；只有需要分别推进时才拆成独立 Todo。讨论或计划本身不代表已经完成。
- 一次只展开当前工作相关的条目；保存成功后简短确认内容与状态，不把全部历史重新塞回对话。

## 工具约定

`todo` 支持 `create/list/get/update`；`idea` 支持 `create/list/get/update/archive/restore`。

- 创建时系统生成 ID。Todo 默认 `open`，进展默认为空列表。`due_date` 可选。
- Todo 和 Idea 的项目默认“随笔”，空白也归入“随笔”；自定义项目名保留原文。默认项目内部始终使用
  `随笔`，中文显示“随笔”、英文显示“Notes”；英文对话查询默认项目也传 `project="随笔"`。
  `update` 省略项目保持原值，传空白改为“随笔”；`list` 省略项目查询所有项目。
- Todo 状态为 `open`（未完成）、`partial`（部分完成）、`done`（完成）、`cancelled`（作废）。
  `done/cancelled` 自动记录关闭时间并归档；设回 `open/partial` 自动恢复并清除关闭时间。
- `progress_text` 追加一条进展，时间默认由系统生成；用户给了明确实际时间时可传带时区的 `progress_time`。
  进展与状态同一次保存。不要通过重写历史来追加进展。
- `update` 只传要修改的字段。省略 `due_date/source_idea` 保持原值，传 `null` 才清除。`text` 不能为 null。
- `list` 可传 `project`、`query`、`scope`（`active/archive/all`）、`offset`、`limit`；Todo 还可按
  `status`、`source_idea` 筛选。默认当前记录、每页 20 条，最多 100 条；根据 `total` 继续翻页。
  返回最近一条进展及 `progress_count`；完整进展用 `get`。
- `get` 按 ID 同时查找当前与归档。归档不改变 ID，也不删除历史；两个 Job 不提供永久删除。
- 每个 Todo 最多关联一个 Idea，一个 Idea 可以关联多个 Todo。`todo.create/update` 传 `source_idea`；
  创建时可继承 Idea 的项目和内容，更新关联不会修改 Todo 的项目、内容或双方状态。
  从 Idea 创建任务、完成或作废任务，都不自动关闭原 Idea。Idea 不再需要时显式 `archive`。

## 调用示例

### 没有项目上下文时随手记

用户：“记一下，周末整理一下最近冒出来的点子。”调用 `todo`：

```json
{"action":"create","text":"周末整理最近的点子"}
```

```json
{"id":"t000","project":"随笔","text":"周末整理最近的点子","status":"open","progress":[]}
```

英文用户说“Remember to sort my recent ideas this weekend”，仍省略项目，返回的项目仍是 `随笔`，
在英文回复和界面中称为“Notes”。Idea 使用同一默认规则。

### 记录想法，再形成行动

用户：“ReMe 可以通过对话管理一句话待办，先记下来。”

```json
{"action":"create","project":"ReMe","text":"通过对话管理一句话待办"}
```

调用 `idea`，返回：

```json
{"id":"i001","project":"ReMe","text":"通过对话管理一句话待办"}
```

用户：“这个开始做，先把后端工具实现。”调用 `todo`：

```json
{"action":"create","source_idea":"i001","text":"实现待办管理的后端工具"}
```

```json
{"id":"t001","project":"ReMe","text":"实现待办管理的后端工具","status":"open","progress":[],"source_idea":"i001"}
```

### 记录期限和部分进展

用户：“后端工具周五前完成。”根据当前日期解析出明确日期后调用 `todo`：

```json
{"action":"update","id":"t001","due_date":"2026-11-20"}
```

```json
{"id":"t001","project":"ReMe","text":"实现待办管理的后端工具","status":"open","progress":[],"due_date":"2026-11-20","source_idea":"i001"}
```

用户：“接口完成了，测试还没写。”调用 `todo`：

```json
{"action":"update","id":"t001","status":"partial","progress_text":"接口已完成，测试待补"}
```

```json
{"id":"t001","project":"ReMe","text":"实现待办管理的后端工具","status":"partial","progress":[{"time":"2026-11-18T10:30:00Z","text":"接口已完成，测试待补"}],"due_date":"2026-11-20","source_idea":"i001"}
```

### 完成、作废和重新打开

用户：“测试也通过了，这个完成。”调用 `todo`：

```json
{"action":"update","id":"t001","status":"done","progress_text":"接口和测试均已完成"}
```

```json
{"id":"t001","project":"ReMe","text":"实现待办管理的后端工具","status":"done","progress":[{"time":"2026-11-18T10:30:00Z","text":"接口已完成，测试待补"},{"time":"2026-11-18T11:00:00Z","text":"接口和测试均已完成"}],"due_date":"2026-11-20","closed_at":"2026-11-18T11:00:00Z","source_idea":"i001"}
```

作废使用 `{"action":"update","id":"t001","status":"cancelled"}`，返回完整条目，状态改为
`cancelled` 并保留关闭时间。重新打开使用 `{"action":"update","id":"t001","status":"open"}`，
返回完整条目，清除 `closed_at`，保留历史进展。

### 已有记录关联、解除关联

调用 `todo`：

```json
{"action":"update","id":"t001","source_idea":"i001"}
```

返回完整 Todo，`source_idea` 为 `i001`，其余内容和状态不变。解除关联：

```json
{"action":"update","id":"t001","source_idea":null}
```

返回完整 Todo，不再包含 `source_idea`。

### 找回工作与完整进展

调用 `todo` 查看项目当前待办：

```json
{"action":"list","project":"ReMe","limit":20}
```

```json
{"items":[{"id":"t001","project":"ReMe","text":"实现待办管理的后端工具","status":"partial","progress":[{"time":"2026-11-18T10:30:00Z","text":"接口已完成，测试待补"}],"progress_count":1,"due_date":"2026-11-20","source_idea":"i001"}],"total":1,"offset":0,"limit":20}
```

`{"action":"get","id":"t001"}` 返回完整 Todo，包括全部进展。查询某个想法产生的全部任务：

```json
{"action":"list","source_idea":"i001","scope":"all"}
```

返回相同分页结构，包含当前及已归档的关联任务。Idea 的 `list/get` 用法相同，Idea 不支持状态或来源筛选。

### 补充、归档和恢复想法

调用 `idea` 补充一句话：

```json
{"action":"update","id":"i001","text":"通过对话管理一句话待办，并持续记录进展"}
```

返回更新后的完整 Idea。用户说“不再考虑这个想法”时调用：

```json
{"action":"archive","id":"i001"}
```

```json
{"id":"i001","project":"ReMe","text":"通过对话管理一句话待办，并持续记录进展","closed_at":"2026-11-18T12:00:00Z"}
```

`{"action":"restore","id":"i001"}` 返回恢复后的完整 Idea，不再包含 `closed_at`，已有 Todo 关联不变。

## 保存边界

后端保存当前 `todos.json`、`ideas.json` 和 `archive/` 下的同名 JSON 数组；归档每份最多 1000 条，
满后新增编号分片，保留历史。复用一个服务写入工作区；使用两个业务工具修改记录。
工具报错时不要声称保存成功；先读取当前条目核对结果，再决定是否重试，避免重复创建或重复追加进展。
