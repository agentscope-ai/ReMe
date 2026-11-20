---
name: manage-todos-ideas
description: 通过 ReMe 的 todo 和 idea 工具记录项目待办与灵感、追加进展、查找未完成工作，并关联 Todo 与 Idea。
---

# 管理待办和想法

在 `proactive` 服务中使用 `todo` 和 `idea`，参数放在顶层。CLI 可调用 `reme todo action=...` 和
`reme idea action=...`。

- 明确要推进的事项记为 Todo，灵感、备选方案和待讨论的问题记为 Idea。每条用一句话概括。
- 修改前用项目或关键词 `list`，用返回的 ID 执行 `get` 或 `update`。优先更新已有条目。
- 创建时使用明确指定的项目。由 Idea 创建 Todo 时，省略 `project` 和 `text` 可继承 Idea 的值。
  没有项目上下文时省略 `project`，工具默认归入 `随笔`，英文界面显示为 Notes。查询这个项目时传 `project="随笔"`。
- 根据用户确认或实际验证的结果追加 `progress_text`，保留已有进展。有明确发生时间时传带时区的 `progress_time`。
  进展和状态可以在同一次 `update` 中保存。期限可选，`due_date` 使用 YYYY-MM-DD。

## 找回记录

两个工具均支持 `create/list/get/update`。列表可按 `project`、`query` 和 `scope` 筛选，
默认 `scope="active"`，也可选 `archive` 或 `all`。省略 `project` 查询所有项目。使用 `offset`、`limit` 和
返回的 `total` 翻页。Todo 可另按 `status` 或 `source_idea` 筛选。列表仅展示最近一条进展及
`progress_count`，用 `get` 查看完整进展。已归档记录仍可用原 ID 查询。

## 记录想法并转成任务

先用 `idea` 记录想法：

```json
{"action":"create","project":"ReMe","text":"通过对话管理一句话待办"}
```

用户决定推进时，用 `todo` 创建关联任务，将示例 ID 替换成 Idea 实际返回的 ID：

```json
{"action":"create","source_idea":"i001","text":"实现待办管理的后端工具"}
```

一个 Idea 可以关联多个 Todo。用 `todo list` 的 `source_idea` 和 `scope="all"` 查找全部关联任务。
在 Todo 的 `update` 中传 `source_idea` 可更换关联，传 `null` 可解除关联。Idea 通过自己的
`archive` 和 `restore` 按 ID 归档与恢复。

## 更新任务进展

```json
{"action":"update","id":"t001","status":"partial","progress_text":"接口已完成，测试待补"}
```

Todo 默认 `open`。部分完成用 `partial`，完成用 `done`，作废用 `cancelled`。
`done` 和 `cancelled` 自动记录关闭时间并归档。设回 `open` 或 `partial` 恢复任务并清除关闭时间。
更新时仅发送需要修改的字段，传 `due_date=null` 可清除期限。
