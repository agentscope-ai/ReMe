---
name: manage-proactive-agent
description: 创建和管理主人交办的定时任务与事件监控，包括接入采集代码、投递观察、查询进展、修改、暂停和删除。适用于实验监控、异常或结束提醒、每日早报。
---

# 管理主动任务

使用 `manage_proactive_agent`，参数直接放在顶层，通过 `agent_name` 指定任务。名称使用 1–3 个简单、常见的小写英文单词，多个单词用下划线 `_` 连接，简洁表达任务主题或职责，能用一个单词说明时优先使用一个，例如 `finance`、`work`、`tech_news`、`experiment_watch`。忽略大小写不能重名；`Remy`、`tmp` 是保留名。先 `list`，再按需 `status`，优先复用相关任务。

## 任务要求与运行配置

- 创建时提供任务名称和说明；不传 `cron` 为事件任务，传 `cron` 为定时任务。系统内置的 `Remy` 负责主对话，不能通过此工具创建。
- 创建时用 `task_description` 写清背景、目标、工作内容、提醒条件、时间和打扰偏好，只把已确认的信息写成要求。
- `create`、`list`、`status` 返回工作区相对的 `task_path`。修改任务说明前，先读取该路径指向的当前文件；根据用户最新要求更新内容，保留仍有效的背景、要求和长期信息，保存后重新读取核对。任务内容以 `TASK.md` 为准。
- 创建后读取任务说明，确认符合主人要求。恢复同名任务时沿用原任务说明和历史。
- `update` 只传需要修改的运行参数，不能修改 `task_description`或名称。修改任务要求时直接编辑 `TASK.md`。改执行时间时同步修改 `cron` 和任务说明中的约定时间。
- `cron` 使用应用时区；`notify_window` 使用 `{start,end}`，格式为 HH:MM，设为 null 可取消时段限制。事件任务可设置 `min_trigger_interval_seconds` 控制处理间隔，它不等于采集频率。
- 任务长期保留，删除操作保留源文件与历史，不自动回收任务目录。暂停、删除或结束事件监控时，也要停止该任务的采集作业，保留原始资料，不影响共用采集器。

## 六个 action 示例

以下给出六种请求。`create`、`status`、`update` 返回统一的 Agent 视图；`list` 返回 `{items,total}`。视图包含任务设置、`task_path`、`status`、事件计数、执行时间和错误；`status=active` 表示执行中。

### list：查找任务

```json
{"action":"list"}
```

### create：创建定时任务

```json
{"action":"create","agent_name":"tech_news","task_description":"每天早上整理重要科技新闻，简洁说明事实和影响，保留来源；每天08:00提醒。","cron":"0 8 * * *"}
```

创建成功后核对任务说明和执行时间，定时任务由框架触发，无需外部投递事件。

### status：查看配置和执行情况

```json
{"action":"status","agent_name":"tech_news"}
```

`last_completed_at=null` 表示没有完成记录；发生错误时会返回 `error`。任务要求和完整背景从 `task_path` 阅读，返回中只提供文件路径。

### update：修改运行配置

```json
{"action":"update","agent_name":"tech_news","cron":"0 9 * * *"}
```

```json
{"agent_name":"tech_news","updated":{"cron":"0 9 * * *"},"next_trigger_at":"2026-11-18T09:00:00+08:00"}
```

暂停设置 `enabled=false`，恢复设置 `enabled=true`。切换到定时任务时设置 `cron`，系统清除旧事件触发间隔；显式传入非零间隔会报错；切换到事件任务时设置 `cron=null`。

### query：询问任务进展

```json
{"action":"query","agent_name":"tech_news","question":"今天的早报完成了吗？有什么需要继续跟进？"}
```

此操作从最新任务会话创建临时只读分支，沿用原有上下文直接回答。运行中使用当前会话，等待下次触发时使用最近一次会话；查询不替换任务主会话，不执行任务或修改任务说明。尚无可查询会话的任务暂不能使用 `query`，先通过 `status` 和任务说明了解情况。

### delete：删除任务

```json
{"action":"delete","agent_name":"tech_news"}
```

删除停止框架内的调度和执行，保留历史文件；外部采集器仍需按其停止方式关闭。

## 事件监控：创建后完成采集接入

主人交办“跟进实验日志，异常或结束时通知我”时，先确认实验、日志位置、判断条件和采集频率，再创建事件任务：

```json
{"action":"create","agent_name":"experiment_watch","task_description":"跟进本次实验，分析持续投递的实验日志；确认异常或结束时提醒，说明依据与影响。正常进展只记录，没有新日志不能单独作为结束依据。"}
```

创建任务不等于已经开始监控，还须完成以下工作：

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
