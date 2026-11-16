---
name: manage-proactive-agent
description: 创建和管理主人交办的定时任务与事件监控，包括接入采集代码、投递观察、查询进展、修改、暂停和删除。适用于实验监控、异常或结束提醒、每日早报。
---

# 管理主动任务

使用 `manage_proactive_agent`，参数直接放在顶层，通过 `name` 指定任务。名称可以是简短的中文或英文，不含空格或路径分隔符，忽略大小写不能重名；`chat`、`tmp` 是保留名。先 `list`，再按需 `status`，优先复用相关任务。

## 任务要求与运行配置

- 创建时用 `instructions` 写清背景、目标、工作内容、提醒条件、时间和打扰偏好，只把已确认的信息写成要求。
- `create`、`status` 返回工作区相对的 `memory_path`。使用原生文件工具阅读和修改该文件，保留有效背景，替换失效要求，保存后核对；不需要专用记忆工具。
- 创建后阅读实际记忆，确认符合主人要求。`restored=true` 表示恢复了同名旧任务，原记忆和历史会保留，新传入的 `instructions` 不覆盖旧记忆。
- `update` 只传需要修改的运行参数，不能修改 `instructions`、名称或 `temporary`。修改任务要求时直接编辑记忆；改执行时间时同步修改 `cron` 和记忆中的约定时间。
- `cron` 使用应用时区；`notify_window` 使用 `{start,end}`，格式为 HH:MM，设为 null 可取消时段限制。事件任务可设置 `min_trigger_interval_seconds` 控制处理间隔，它不等于采集频率。
- `temporary` 仅在创建时设置，与触发方式独立；临时任务连续七天没有新输入后由框架清理。暂停、删除或结束事件监控时，也要停止该任务的采集作业，保留原始资料，不影响共用采集器。

## 六个 action 示例

以下分别给出请求和返回的业务结果（`answer`）；时间和工作结果均为示例。

### list：查找任务

```json
{"action":"list"}
```

```json
{"items":[{"name":"科技早报","trigger_type":"cron","enabled":true,"cron":"0 8 * * *"}]}
```

### create：创建定时任务

```json
{"action":"create","name":"科技早报","instructions":"每天早上整理重要科技新闻，简洁说明事实和影响，保留来源；每天08:00提醒。","trigger_type":"cron","cron":"0 8 * * *"}
```

```json
{"name":"科技早报","memory_path":"reme_agents/科技早报/MEMORY.md","next_trigger_at":"2026-11-18T08:00:00+08:00"}
```

创建成功后核对记忆和执行时间，定时任务由框架触发，无需外部投递事件。

### status：查看配置和执行情况

```json
{"action":"status","name":"科技早报"}
```

```json
{"name":"科技早报","trigger_type":"cron","enabled":true,"temporary":false,"cron":"0 8 * * *","memory_path":"reme_agents/科技早报/MEMORY.md","running":false,"pending_events":0,"in_flight_events":0,"last_completed_at":"2026-11-17T08:00:05+08:00","next_trigger_at":"2026-11-18T08:00:00+08:00"}
```

`last_completed_at=null` 表示没有完成记录；发生错误时会返回 `error`。任务要求和完整背景从 `memory_path` 阅读，返回中不重复附带记忆正文。

### update：修改运行配置

```json
{"action":"update","name":"科技早报","cron":"0 9 * * *"}
```

```json
{"name":"科技早报","updated":{"cron":"0 9 * * *"},"next_trigger_at":"2026-11-18T09:00:00+08:00"}
```

暂停设置 `enabled=false`，恢复设置 `enabled=true`。切换到定时任务时传 `trigger_type="cron"` 和 `cron`；切换到事件任务时传 `trigger_type="event"`，代码会清理不适用的旧配置。

### query：询问已完成的工作

```json
{"action":"query","name":"科技早报","question":"今天的早报完成了吗？有什么需要继续跟进？"}
```

```json
"今天08:00的早报已完成，已整理三条主要消息，没有待处理资料。"
```

此操作在临时只读会话中回答，不会执行任务或修改记忆；尚未完成过执行的任务暂不能使用 `query`，先通过 `status` 和记忆文件了解情况。

### delete：删除任务

```json
{"action":"delete","name":"科技早报"}
```

```json
"Deleted task '科技早报'; history files retained."
```

删除停止框架内的调度和执行，保留历史文件；外部采集器仍需按其停止方式关闭。

## 事件监控：创建后完成采集接入

主人交办“跟进实验日志，异常或结束时通知我”时，先确认实验、日志位置、判断条件和采集频率，再创建事件任务：

```json
{"action":"create","name":"实验监控","instructions":"跟进本次实验，分析持续投递的实验日志；确认异常或结束时提醒，说明依据与影响。正常进展只记录，没有新日志不能单独作为结束依据。","trigger_type":"event","temporary":true}
```

```json
{"name":"实验监控","memory_path":"reme_agents/tmp/实验监控/MEMORY.md"}
```

创建任务不等于已经开始监控，还须完成以下工作：

1. 复用当前 HTTP 服务，通过 `POST /app_config`（请求体 `{}`）确认 `workspace_dir`、`resource_dir`，确认输入来源和已有采集宿主。
2. 在返回的 `memory_path` 所在目录下编写采集代码和运行配置，或复用现有采集器。先将完整资料原子保存到服务工作区，再通过 `POST /append_event` 投递观察；是否异常或结束由任务根据资料判断。
3. `paths` 使用规范的工作区相对路径，不含绝对路径或 `..`；远程资料先同步到工作区。按规模选择快照或增量采集，保留原始日志；增量游标只在服务确认接收后推进，日志轮转须保留归档或接入训练回调。
4. 启动对应宿主配置，记录入口、命令、运行标识、输出日志、重启策略和停止方式，并写入任务记忆；ReMe 内部采集器由 BackgroundJob/Component 管理生命周期。
5. 用真实资料验证采集和投递，并确认新资料能持续进入。通过 `status` 查看待处理、处理中数量、完成时间和错误，再用 `query` 或日记核对分析结果；提醒条件出现时核对实际提醒，区分已经验证和尚未发生的环节。

采集代码调用的 HTTP 接口使用 `agent_id`，值就是任务名称，不需要生成另一套 ID：

```bash
curl --fail-with-body http://127.0.0.1:2333/append_event \
  -H 'Content-Type: application/json' \
  --data-binary '{
    "agent_id": "实验监控",
    "events": [{
      "id": "experiment-log-snapshot-001",
      "event_ts": "2026-11-17T10:30:00+08:00",
      "source": "experiment_log",
      "description": "实验日志已更新，请读取快照判断运行状态。",
      "paths": ["resource/experiments/snapshot-001.log"]
    }]
  }'
```

地址、时间和路径替换为实际值。检查 HTTP 状态、`success=true`、返回的 `answer.agent_id` 和 `answer.event_ids`；HTTP 200 但 `success=false` 也算失败。临时错误有限重试，同一批重试复用相同事件 ID 和原始内容，新资料使用新 ID；持续失败须报错并交给宿主处理。

完整资料留在文件中，事件携带简述和路径；`source="cron"` 为框架保留值，不直接写内部事件流文件。
