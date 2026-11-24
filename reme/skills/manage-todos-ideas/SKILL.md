---
name: manage-todos-ideas
description: 用 ReMe 的 todo 和 idea 工具查找项目、管理待办和想法、维护进展并关联条目。
---

# 管理待办和想法

用 `todo` 记录明确要做的事，用 `idea` 记录想法和待讨论的方案。先查有没有已有条目，再决定更新或新增。
工具参数和 `action` 放在同一层。示例中的 `t001` 和 `i001` 用工具返回的实际 ID 替换。
从命令行调用时，先运行 `reme start config=proactive`，再在另一个终端调用 `reme todo` 或 `reme idea`，用 `key=value` 传入参数。

## 共用参数

| 参数 | 用法 |
| --- | --- |
| `action` | 必填。取下文对应工具列出的操作名。 |
| `id` | `get`、`update` 和 Idea 的 `archive`、`restore` 使用已有条目的 ID。新建时由工具生成。归档后仍可用原 ID 查询。 |
| `project` | `create` 和 `update` 写项目名，`list` 按项目名精确筛选。使用查询返回的原名。单独新建时省略或留空默认归入「随笔」（英文界面显示 Notes）。`list` 省略时查所有项目，`update` 省略时保留原项目。 |
| `text` | `create` 和 `update` 使用一句非空文字。新建 Idea 时必填，新建 Todo 时可由 `source_idea` 提供。 |
| `scope` | `list` 和 `todo list_projects` 使用。`active` 查未归档，`archive` 查已归档，`all` 查全部。`list` 默认 `active`，`list_projects` 默认 `all`。 |
| `query` | `list` 使用。不区分大小写，按子串匹配项目和正文。Todo 还会匹配进展和关闭备注。 |
| `offset` | `list` 使用。跳过前几条，默认 `0`。 |
| `limit` | `list` 使用。每页返回 `1`–`100` 条，默认 `20`。根据 `total` 和 `offset` 判断是否需要继续翻页。 |

## Todo 专用参数

| 参数 | 用法 |
| --- | --- |
| `status` | `create` 和 `update` 设置状态，`list` 精确筛选。默认 `pending`（待开始），开工后用 `in_progress`，完成或取消时用 `closed`。重新设为 `pending` 或 `in_progress` 会恢复条目并清除关闭时间和备注。 |
| `due_date` | `create` 和 `update` 使用，格式为 `YYYY-MM-DD`。传 `null` 清除，省略则保留已有期限。 |
| `source_idea` | `create` 和 `update` 传已有 Idea 的 ID 建立关联，`list` 用它查找关联 Todo。新建 Todo 时，省略的 `project` 和 `text` 继承 Idea 的值。更新时传 `null` 解除关联。 |
| `progress_text` | `create` 或 `update` 追加一条非空进展。`update` 同时传 `progress_match` 时，替换匹配条目的文字。写入前先读完整进展，只记录新增事实。 |
| `progress_time` | 与 `progress_text` 一起使用，时间带时区。省略时，新增进展用当前时间，替换进展保留原时间。 |
| `progress_match` | 仅用于 `update`。传 `get` 返回的原进展对象，包含完整的 `time` 和 `text`，工具会精确匹配一条进展。省略 `progress_text` 会删除匹配的进展。 |
| `close_note` | `create` 或 `update` 传 `status=closed` 时，可用一句话说明结果或关闭原因。传 `null` 清除备注。 |

## `todo` 操作

### `list_projects`

从 Todo 和 Idea 汇总项目。每项返回 `project`、`todo_count` 和 `idea_count`。

```json
{"action":"list_projects","scope":"all"}
```

### `list`

查询 Todo。列表只带最近一条进展和 `progress_count`。要查看全部进展，用 `get`。

```json
{"action":"list","project":"ReMe","status":"in_progress","source_idea":"i001","scope":"all","query":"发布包","offset":0,"limit":20}
```

### `get`

读取单条 Todo 及其完整进展。

```json
{"action":"get","id":"t001"}
```

### `create`

从已有 Idea 新建 Todo，沿用 Idea 的项目，并写入已发生的进展。

```json
{
  "action": "create",
  "source_idea": "i001",
  "text": "验证 Windows 发布包",
  "status": "in_progress",
  "due_date": "2026-11-25",
  "progress_text": "Windows 安装通过，待验证启动。",
  "progress_time": "2026-11-24T10:15:00+08:00"
}
```

### `update`

传 ID 和要改的字段。下面把新结果合并到原进展，并关闭已经验证完毕的 Todo。

```json
{
  "action": "update",
  "id": "t001",
  "progress_match": {
    "time": "2026-11-24T10:15:00+08:00",
    "text": "Windows 安装通过，待验证启动。"
  },
  "progress_text": "Windows 安装和启动均通过。",
  "status": "closed",
  "close_note": "发布包通过 Windows 安装和启动验证。"
}
```

## `idea` 操作

### `list`

按项目、关键词和归档状态查找 Idea。

```json
{"action":"list","project":"ReMe","scope":"all","query":"发布包","offset":0,"limit":20}
```

### `get`

读取单条 Idea。

```json
{"action":"get","id":"i001"}
```

### `create`

用一句话记录想法。

```json
{"action":"create","project":"ReMe","text":"发版前在干净环境验证发布包能安装和启动。"}
```

### `update`

传 ID 和要修改的项目或文字。

```json
{"action":"update","id":"i001","text":"发版前分别在干净的 macOS 和 Windows 环境验证发布包。"}
```

### `archive`

归档 Idea，保留原 ID 和 Todo 关联。关联 Todo 创建或关闭后，Idea 仍可单独决定是否归档。

```json
{"action":"archive","id":"i001"}
```

### `restore`

恢复已归档的 Idea。

```json
{"action":"restore","id":"i001"}
```

## 随记后台更新

收到事件后，查找所有相关 Todo 和 Idea，读完命中 Todo 的全部进展。只更新未关闭 Todo 的增量进展，确认开工时可设为 `in_progress`。
需要用户决定的新增、修改、关闭和关联，逐项交给 Remy，写清已有条目、新依据和建议处理方式。
