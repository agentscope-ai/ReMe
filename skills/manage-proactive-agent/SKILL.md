---
name: manage-proactive-agent
description: 通过 ReMe 管理定时任务和事件监控，包括任务要求、运行设置、进展问答和采集接入。适用于实验监控、异常或结束提醒及每日早报。
---

# 管理主动任务

使用 `manage_proactive_agent`，参数放在顶层，通过 `agent_name` 指定任务。先 `list` 查看是否已有相关任务，
按需用 `status` 查状态。名称建议用下划线连接简短的小写英文单词，如 `experiment_watch`。名称忽略大小写必须唯一，
`Remy` 保留给主对话。

## 任务要求与运行配置

- 创建时用 `task_description` 写清已确认的目标、背景、提醒条件和时间。传 `cron` 创建定时任务，省略则创建事件任务。
- 需求以 `task_path` 指向的 `TASK.md` 为准。编辑前读取当前内容，保留有效背景，保存后重新读取核对。重建同名已删除任务会复用原文件和历史。
- `update` 修改运行设置。调整约定执行时间时同步修改 `cron` 与 `TASK.md`。`cron` 使用应用时区。
  `notify_window` 使用 HH:MM 格式的 `{start,end}`，传 `null` 取消限制。事件任务的
  `min_trigger_interval_seconds` 从上一轮成功完成时计时，采集频率由采集器配置。

## 六个 action 示例

`create`、`status`、`update` 的 `answer` 是同一 Agent 视图，包含 `task_path`、运行设置、事件计数和时间。
`list` 返回 `{items,total}`，其中 `items` 为任务视图。`status=active` 表示执行中。

### list：查找任务

```json
{"action":"list"}
```

### create：创建定时任务

```json
{"action":"create","agent_name":"tech_news","task_description":"每天早上整理重要科技新闻，简洁说明事实和影响，保留来源；每天08:00提醒。","cron":"0 8 * * *"}
```

创建成功后核对 `task_path` 和 `next_trigger_at`。框架按日程触发定时任务。

### status：查看配置和执行情况

```json
{"action":"status","agent_name":"tech_news"}
```

`last_completed_at=null` 表示没有完成记录；发生错误时会返回 `error`。任务要求和完整背景从 `task_path` 阅读，返回中只提供文件路径。

### update：修改运行配置

```json
{"action":"update","agent_name":"tech_news","cron":"0 9 * * *"}
```

以返回视图的 `cron` 和 `next_trigger_at` 核对新安排。暂停设置 `enabled=false`，恢复设置 `enabled=true`。
切换到定时时设置 `cron`，事件间隔归零。切回事件任务时设置 `cron=null`。

### query：询问任务进展

```json
{"action":"query","agent_name":"tech_news","question":"今天的早报完成了吗？有什么需要继续跟进？"}
```

此操作从最新任务会话创建临时只读分支，直接返回回答。尚无会话时，通过 `status` 和任务说明了解情况。

### delete：删除任务

```json
{"action":"delete","agent_name":"tech_news"}
```

删除停止框架内的调度和执行，并保留历史文件。暂停或结束监控时，按需停止该任务独占的外部采集作业。

## 事件监控：创建后完成采集接入

主人交办“跟进实验日志，异常或结束时通知我”时，先确认实验、日志位置、判断条件和采集频率，再创建事件任务：

```json
{"action":"create","agent_name":"experiment_watch","task_description":"跟进本次实验，分析持续投递的实验日志；确认异常或结束时提醒，说明依据与影响。正常进展只记录，没有新日志不能单独作为结束依据。"}
```

创建事件任务后接入来源：

1. 复用当前 HTTP 服务，通过 `POST /get_client_info`（请求体 `{}`）确认 `workspace_dir`、`resource_dir`，确认输入来源和已有采集宿主。
2. 在返回的 `task_path` 所在目录下编写采集代码和运行配置，或复用现有采集器。先将完整资料原子保存到服务工作区，再通过 `POST /append_agent_events` 投递观察；是否异常或结束由任务根据资料判断。
3. `paths` 使用规范的工作区相对路径，不含绝对路径或 `..`；远程资料先同步到工作区。按规模选择快照或增量采集，保留原始日志；增量游标只在服务确认接收后推进，日志轮转须保留归档或接入训练回调。
4. 启动对应宿主配置，记录入口、命令、运行标识、输出日志、重启策略和停止方式，并写入任务说明；ReMe 内部采集器由 BackgroundJob/Component 管理生命周期。
5. 用真实资料验证采集和投递，并确认新资料能持续进入。通过 `status` 查看待处理、处理中数量、完成时间和错误，再用 `query` 或日记核对分析结果；提醒条件出现时核对实际提醒，区分已经验证和尚未发生的环节。

采集代码与管理工具统一使用 `agent_name` 指定任务：

```bash
curl --fail-with-body http://127.0.0.1:2333/append_agent_events \
  -H 'Content-Type: application/json' \
  --data-binary '{
    "agent_name": "experiment_watch",
    "events": [{
      "id": "experiment-log-snapshot-001",
      "event_ts": "2026-11-17T10:30:00+08:00",
      "source": "experiment_log",
      "description": "实验日志已更新，请读取快照判断运行状态。",
      "paths": ["resource/experiments/snapshot-001.log"]
    }]
  }'
```

地址、时间和路径替换为实际值。检查 HTTP 状态、`success=true`、返回的 `answer.agent_name` 和 `answer.event_ids`；HTTP 200 但 `success=false` 也算失败。临时错误有限重试，同一批重试复用相同事件 ID 和原始内容，新资料使用新 ID；持续失败须报错并交给宿主处理。

完整资料留在文件中，事件携带简述和路径；`source="cron"` 为框架保留值，不直接写内部事件流文件。
