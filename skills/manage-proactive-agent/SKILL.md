---
name: manage-proactive-agent
description: 创建并落实主人交办的定时任务和事件监控，包括编写、部署采集代码，通过 HTTP 投递观察，以及查询、修改、暂停和删除任务。适用于实验日志监控、异常或结束提醒、每日科技早报。
---

# 管理主动任务

使用 `manage_proactive_agent` 管理任务。事件监控须完成创建、编写并启动采集代码、HTTP 投递和验证，接入成功后再告知主人已开始监控。

## 创建任务

- 先用 `action="list"` 查找已有任务，按需用 `action="status"` 了解相关任务，优先复用或更新；新任务使用简短、含义明确的英文名称，忽略大小写不能重名，`config.name` 即不可修改的 `agent_id`。
- 根据主人的最新要求和相关上下文编写 `config.description`，写清背景、目标、工作内容、提醒条件、时间和打扰偏好；只把已确认的信息写成要求。时间或输入来源不明时，先查上下文，再询问必要信息。
- 用 `action="create"` 创建后核对返回的 `memory_content`；`memory_truncated=true` 时读取完整 `memory_path`，确认记忆符合主人的要求。

## 选择触发方式

| 需求 | 配置 | 说明 |
| --- | --- | --- |
| 固定时间执行，如早报、定期检查日志 | `trigger_type="cron"`，设置 `cron` | 框架定时触发，可联网查资料，无需外部投递事件。 |
| 收到新资料后判断，如实验异常或结束 | `trigger_type="event"` | 采集器投递观察，Agent 根据资料和本地文件分析。 |
| 短期跟进某次实验或临时事项 | 额外设置 `temporary=true` | 与触发方式独立，可用于 cron 或 event。 |

`cron` 按应用时区决定执行时间，`notify_window` 决定允许提醒的时段。定期读取文件时，在任务描述中写清工作区内的文件路径。

## 事件监控：创建之后完成接入

1. 复用当前 HTTP 服务，调用 `POST /app_config`，请求体 `{}`，从 `answer` 确认 `workspace_dir` 和 `resource_dir`；确认输入来源、采集频率及已有运行宿主。
2. 在返回的 `memory_path` 所在目录下编写采集代码和运行配置，或复用现有采集器。代码持续取得新资料，按规模选择快照或增量采集，先将完整资料原子保存到服务工作区，再调用 `POST /append_event`；是否异常或结束由 Agent 判断。
3. 文件使用规范的工作区相对 `paths`，不含绝对路径或 `..`；远程资料先同步到服务工作区。保存独立快照或分段文件，保留原始日志；增量采集只在服务确认接收后推进游标，轮转日志须保留归档或接入训练回调。
4. 编写并启动对应宿主配置，让采集代码持续运行。记录命令、运行标识、输出日志、重启策略和停止方式，并把入口写入任务记忆；ReMe 内部采集器交给 BackgroundJob/Component 生命周期管理。
5. 用真实资料完成一次采集和投递，再验证新增资料能持续进入事件流。通过 `status` 查看 `pending_events`、`in_flight_events`、`last_completed_at` 和 `error`，用 `query` 或关联记录确认已分析；提醒条件发生时再核对实际提醒，交付时区分已验证和尚未发生的环节。

### HTTP 投递示例

以下请求由采集代码发送，地址、时间、ID 和路径替换为实际值；请求根层级直接使用 `agent_id` 和 `events`：

```bash
curl --fail-with-body http://127.0.0.1:2333/append_event \
  -H 'Content-Type: application/json' \
  --data-binary '{
    "agent_id": "experiment_monitor",
    "events": [{
      "id": "experiment-log-snapshot-001",
      "event_ts": "2026-11-15T10:30:00+08:00",
      "source": "experiment_log",
      "description": "实验日志已更新，请读取快照判断运行状态。",
      "paths": ["resource/experiments/experiment_monitor/snapshot-001.log"],
      "metadata": {"experiment": "本次实验"}
    }]
  }'
```

采集代码检查 HTTP 状态、`success=true`、`answer.agent_id` 和返回的 `answer.event_ids`；HTTP 200 且 `success=false` 也算失败。
临时错误有限重试，同一批重试复用相同的 `id` 和原始事件内容，新资料使用新 ID；持续失败须报错并让宿主处理。
事件简述和文件引用留在请求中，完整资料留在文件中；`source="cron"` 为框架保留值，不直接写内部事件流文件。

## 查询、修改与删除

- 查看配置和进展用 `action="status"`；询问具体工作情况用 `action="query"`，传入 `agent_id` 和 `query`，需要详情时读取关联记录。
- 修改或删除前先查看 `status` 和当前记忆，截断时读完整文件。任务要求和背景使用 `agent_memory(action="read", agent_id="...")` 读取，通过 `action="write"` 携带 `content` 与读取返回的 `expected_revision`（值取 `revision`）保存，替换失效要求、保留相关知识，并读回确认；发生版本冲突时重新读取合并，不用命令直接覆盖 MEMORY.md。
- 运行配置用 `action="update"`，传入 `agent_id` 和 `config`，如 `cron`、`min_trigger_interval_seconds`、`notify_window`、`enabled`；它不接受 `description`。
- 暂停用 `update` 设置 `enabled=false`，删除用 `action="delete"` 和 `agent_id`。暂停、结束或删除监控时，同步停止该任务的采集作业，保留原始资料和历史，不影响共用采集器。

## 示例：实验日志监控

主人说：“帮我看看实验日志，有问题或者结束通知我。”确认实验、日志位置和判断条件，检查已有任务后，使用以下 `manage_proactive_agent` 参数：

```json
{
  "action": "create",
  "config": {
    "name": "experiment_monitor",
    "trigger_type": "event",
    "temporary": true,
    "description": "跟进本次实验，读取持续投递的实验日志；仅在确认运行异常或实验结束时提醒主人，说明证据与影响。正常进展只记录，没有新日志不能单独作为结束依据。"
  }
}
```

按实际背景补充 `description`，创建后继续编写、启动并验证日志采集器，按上述 HTTP 格式持续投递。正常进展只记录，确认异常或结束时提醒。

## 示例：每日科技早报

主人说：“每天早上给我一份简洁的科技早报。”先检查已有任务；具体时间未约定时确认，以下假设已确认每天 08:00：

```json
{
  "action": "create",
  "config": {
    "name": "tech_morning_brief",
    "trigger_type": "cron",
    "cron": "0 8 * * *",
    "description": "每天早上为主人整理简洁的科技早报，关注最新的重要科技进展，合并重复消息，用少量要点说明事实和影响，并保留原始来源 URL。"
  }
}
```

创建后核对 `next_trigger_at`，由框架定时执行。改到九点时先查看 `status`，再 `update` 为 `cron="0 9 * * *"`，同步记忆中的时间；“只看 AI 新闻”则修改任务记忆。
