---
title: 服务与部署
description: 使用 ReMe 的 HTTP、SSE、MCP 和 Studio 服务，并理解默认安全边界。
---

# 服务与部署

ReMe 可以作为本地 HTTP 服务、独立 MCP Server 或一次性 CLI Job 运行。默认 HTTP 服务监听
`127.0.0.1:2333`；同一进程提供 JSON API、SSE、MCP 与可选的 ReMe Studio。

## HTTP 服务

在一个终端启动服务并保持运行，以下启动命令任选一条：

```bash
reme start
reme start service.host=127.0.0.1 service.port=8181
```

在另一个终端通过 CLI client 调用 Job：

```bash
reme search query="用户偏好" limit=5
```

普通 Job 也暴露为 `POST /<job-name>`；以下 HTTP 示例使用默认端口 `2333`：

```bash
curl -s http://127.0.0.1:2333/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"用户偏好","limit":5}'
```

请求体允许 Job 参数位于顶层。响应遵循 `Response`：

```json
{
  "success": true,
  "answer": "...",
  "metadata": {}
}
```

未被 Step 捕获的错误会转换成 `success: false` 的响应。

## Streaming 与 SSE

`backend: stream` 的 Job 仍使用 `POST /<job-name>`，但响应类型是 `text/event-stream`。每个 chunk 使用统一的 streaming schema；失败时会发出错误 chunk，并以终止事件结束。

默认 `chat` 是 Stream Job：

```bash
curl -N http://127.0.0.1:2333/chat \
  -H 'Content-Type: application/json' \
  -d '{"query":"总结我的长期偏好"}'
```

Studio 的普通对话和 **Proactive Chat** 共用一套界面。默认配置只提供 `chat`，接收 `query` 和可选的 `session_id`，
各对话独立。使用 `config=proactive` 启动后，提供 `pa_*` Job，不再提供 `chat`。前端按服务实际开放的 Job 启用对应入口。

Proactive Chat 使用由 Application 管理当前会话的 `pa_chat`，发送时只传 `query`。`pa_session` 无参数，`answer` 返回
`{"session_id": "...", "status": "idle"}`；尚未创建会话时两个字段均为 null。`pa_new_session` 无参数，创建并切换空会话，
在 `answer` 返回新 ID，不触发模型生成。

先在一个终端运行 `reme start config=proactive`，再从另一个终端调用：

```bash
reme pa_session
reme pa_chat query="有哪些事项值得继续？"
reme pa_new_session
```

当前会话在同一 Application 内共享，重启后清空指针，会话文件保留。生成期间拒绝并发发送和新建，状态查询无需等待生成。
PA Job 目前只支持 `CodexAgentWrapper`，其他 wrapper 抛出 `NotImplementedError`。
当前原生会话不可用时，`pa_session` 返回 `status: "error"`，可通过新建会话恢复。
空会话创建一旦开始，会先完成当前会话的切换，再传播取消信号。

`proactive` 还提供只读的 `POST /pa_events`，供后台客户端订阅。服务启动后，在另一个终端调用：

```bash
curl -N http://127.0.0.1:2333/pa_events \
  -H 'Content-Type: application/json' \
  -d '{}'
```

每个 SSE `data:` 帧仍是 `StreamChunk`，其中 `chunk` 为有类型约束的
[`PaEvent`](../../reme/schema/pa_event.py)。首次连接收到 `snapshot`，包含当前会话 ID、合并后的 PA 活动状态和
可选的 `SessionSnapshot` 历史，不会创建会话。后续事件如下：

| 类型 | 客户端处理 |
| --- | --- |
| `session_changed` | 切换当前会话，清空上一会话的展示历史。 |
| `message_upsert` | 按 `(session_id, message.id)` 插入或替换完整的 `message`。 |
| `status_changed` | 更新活动状态；PA 操作释放串行锁之前持续显示忙碌。 |
| `reset` | 丢弃旧游标；同一连接随后发送新的 `snapshot`。 |
| `heartbeat` | 保持连接，不携带游标。 |

应用事件后保存其不透明的 `cursor`。重连时传 `{"cursor":"<已保存的游标>"}`，先重放缓冲中的遗漏事件，再接收实时变化。
进程或 runtime 重启、上游断开、客户端落后超过缓冲范围时，必须重新同步快照。快照读取与增量交接通过版本复核，
覆盖并发切换和消息提交。历史读取失败时，快照包含 `error`，不含历史和游标；客户端保留已有展示，服务在心跳间隔后重试，
成功取得快照后恢复增量投递。
客户端需要容忍重复 upsert；只有已完成的主动消息才进入提醒气泡队列，逐字回复继续使用 `pa_chat`。

消息事件在主动消息成功发布后，或聊天 turn 结束且原生历史可读后产生，也覆盖取消后的历史。恢复依据仍是会话文件，
事件缓冲和游标仅保存在内存中，不代表送达或已读回执。通过 `components.pa.default.max_events`、`max_buffer_bytes` 和
`heartbeat_seconds` 配置缓冲容量和心跳。`jobs.pa_events.stream_queue_size: 1` 限制 HTTP 输出队列，慢客户端通过历史
重新同步，避免无限积压。其他 Stream Job 默认行为保持不变，`stream_queue_size: 0` 表示无界队列。
自定义配置启用 `pa_events` 时，需要一并配置 `proactive.yaml` 中的 `pa` 组件；订阅跟随该组件配置的 Codex wrapper。

`pa` 组件从 `{metadata_dir}/pa/session_ids.json` 的最后一条恢复当前对话。选择会话时，先原子保存会话顺序，
再发布当前选择，同时保留读取游标。启动时将已有 `{session_dir}/pa.json` 的选择迁入列表，保存成功后移除旧文件。
原生对话历史仍归 Codex 所有，不删除历史；元数据无效或原生历史不可用时明确报错，不静默创建替代会话。
事件游标和活动状态在重启后重新创建，客户端通过快照恢复。

`proactive` 配置设置 `service.timeout_graceful_shutdown: 5`。HTTP 服务退出时，先给正在处理的请求五秒完成，
随后取消仍未结束的请求，再关闭应用组件，避免长订阅无限阻塞退出。自定义 HTTP 配置启用长订阅时，也应设置有限的
优雅退出超时。

三个 Agent 统一使用 `{workspace_dir}/{daily_dir}/personal/MEMORY.md`（默认是工作区内的 `daily/personal/MEMORY.md`）。
PA 的 Prompt 提供实际配置的工作区和该记忆路径；对话 Prompt 要求回复前先读记忆，
更新前先读取最新完整文件，再合并修改，不再将记忆正文注入会话 Prompt。新建会话或执行主动任务时，公共逻辑仅在文件
不存在时，用 `reme/steps/proactive/base_pa_step.yaml` 的 `default_memory` 初始化，通过逐路径锁和原子写入保护并发。
默认内容覆盖财经新闻、AI 前沿和 ReMe 近况，并要求只提醒有价值的内容，
非重要或重复信息不要打扰主人；已有文件（包括空文件）保持原样，
不再进行首次信息询问或根据空正文跳过任务。

`pa_chat.py` 提供对话 Step（`pa_chat_step`）、新建会话和会话状态 Step；Remy 的人设、记忆维护与遗忘说明由
`PromptHandler` 从 `pa_chat.yaml` 加载，提供英文默认版本和中文版本。对外 Job 仍为 `pa_chat`、`pa_session`、`pa_new_session`。
proactive 默认 `language=zh`，可用 `language=en` 切换英文；Step 的 `language` 优先于 Application。
三个角色使用简洁的中英文提示词与少量通用例子，模型为 `kimi-k3.1-pro`，推理强度为 `xhigh`。
Codex 客户端采用 65536 token 的上下文预算和 56000 token 的自动压缩阈值，这不是模型厂商的最大上下文声明；
原生长期记忆保持启用。`CODEX_MODEL_NAME` 可覆盖模型，切换型号时也应检查 model catalog 的能力描述。
配置文件变更在下次启动时加载，恢复后的会话在后续聊天中采用当前配置。
通过 Step 的 `prompt_dict` 覆盖对应语言的键（如 `system_prompt_zh`、`default_memory_zh`）；默认记忆覆盖仅影响缺失文件。
主人要求遗忘时，提示 Agent 删除对应内容，不转存，也不从旧会话或摘要中恢复；维护时清理确认过时的状态，保留有效的长期事实。
创建会话会初始化缺失的记忆，但不生成回复。每次聊天从当前配置生成 Prompt 和工具绑定，不在指针文件或内存会话状态中
缓存这些参数；记忆路径始终指向最新文件。
路径无效或初始化失败时，新建失败并保留原来的当前会话。

`proactive` 配置每次唤醒创建一个主动 Agent 会话，在同一会话中依次执行六个 query：长期记忆及上轮交接、
对话反馈和近期对话、到期任务及价值分析、提醒决策、可选的唤醒配置调整、生成下轮交接。整轮工具绑定保持一致。
只有最后一问成功且摘要非空时才原子覆盖交接文件。消费反馈和推进游标前，先将输入保存到 `proactive_context.jsonl`，
并追加本轮阶段结果；失败保留恢复上下文和上轮摘要，成功完成交接后清空恢复文件。

PA 对话、会话和事件 Step 明确依赖配置中的 `PaRuntime`，不再兜底到 metadata 字典。
Runtime 管理会话生命周期及单个有类型的当前轮次，内部任务与提醒对象各自持有自己的锁；轮次结束不影响后台投递。
提醒提交必须绑定有效轮次。反馈工具合并在 `feedback.py`，提醒工具合并在 `reminders.py`。
任务工具仅由 YAML 配置，默认是截图与财经采集。会话选择只从 `session_ids.json` 恢复，旧 `pa.json` 启动迁移已移除。

PA 组件在 Job 启动前创建 `{metadata_dir}/pa/` 下缺失的文件：

| 文件 | 初始内容 |
| --- | --- |
| `session_ids.json` | `[]`，最后一条表示当前对话 |
| `proactive.json` | `components.pa.default.default_proactive` |
| `tasks.json` | `components.pa.default.default_tasks`，补齐 ID 和登记时间 |
| `proactive_inbox.txt` | 空文件 |
| `handoff.txt` | 空文件 |
| `proactive_context.jsonl` | 空的恢复记录 |
| `reminders.jsonl` | 空文件 |
| `task_runs.jsonl` | 空文件 |

默认值在 `reme/config/proactive.yaml` 中配置：start_time 为字符串 "0900"、end_time 为字符串 "2300"、
interval_minutes 为 5。已有内容保留，包括空文本和空任务列表；已有 JSON 错误时明确报错，不用默认值覆盖。
每个 Agent 调用模型前通过 `base_pa_step.yaml` 的本地化默认内容初始化缺失的个人记忆。

唤醒使用应用时区，开始时间包含、结束时间不包含；支持跨午夜区间，开始结束相同表示全天。
活动时间内一轮完成后再等待指定间隔，活动时间外等待下次开始。等待期间最多每30秒检查 JSON 修改，无需重启；
配置错误时报告错误并暂停模型工作。旧 YAML 等待参数、内存临时覆盖和 set_next_wakeup 工具已移除。
资源清理仍使用独立 CronJob。

可选交流 hint 默认开启：在 `proactive.json` 设置 `hints_enabled: false` 可关闭抽样和注入；
`hint_probabilities` 分别配置 `morning`、`midday`、`evening`，默认依次为1.0、0.5、0.5，范围均为0到1。
初始值来自 YAML 的 `default_proactive`；已有 JSON 缺失这些字段或只配置部分时段时，其余使用默认值。
此前使用单值 `hint_probability` 的配置需要移除旧字段，改用 `hint_probabilities` 对象。
在活动时间内，早间06:00–12:00、午间12:00–14:00，以及傍晚结束前30分钟分别最多抽一次；
结束前窗口最早从18:00开始，跨午夜或全天活动则使用21:00后的晚间窗口，不把晚间 hint 投到早上。
未抽中也占用当天该时段的机会；有当前对话时才抽样，并在 `hint_state.json` 中保存结果，重启不重抽，尽量避开上次同类变体。
抽中的双语方向来自 `proactive_agent.yaml`，仅在提醒决策前注入；Agent 可以忽略，正常的新信息提醒不受抽样影响。
共18个方向，围绕生活关心、问候和小彩蛋，只有一个晚间方向涉及已确认 TODO 的总结。
是否实际发送仍以 `reminders.jsonl` 为准，随机方向不写入个人记忆，也不应作为后续轮次指令。

`read_conversation(max_turns=20)` 跨最近100个登记会话消费新增的已完成对话轮次，每次主人输入与其回复作为一轮，
排除工具调用和主动提醒。每条默认2000字符、总量默认20000字符，返回从旧到新的大字符串；窗口外的旧轮次省略。
时间缺失明确标注未知，读取或恢复记录保存失败时保留游标。主动 Agent 在 q2 中只读取一次。

`send_to_proactive(message)` 追加带时间的反馈；`get_proactive_messages()` 在组件锁内获取全部文字并清空。
`get_reminder(limit=10, offset=0)` 按创建时间从新到旧返回每条提醒的最新状态，只读、不废弃历史。
JSONL 用相同 ID 追加 `pending`、`sent`、`discarded`、`failed` 状态，包含创建、更新、过期、发送时间和目标会话，
替代原来的 `is_draft` 字段。

`send_reminder(content)` 每轮接受一次决策，最多200字符。空内容撤销绑定会话的待发送提醒并返回 `skipped`；
新内容成功落盘后替换旧待发送提醒。当前空闲时尝试直接注入并返回 `sent`；忙碌时返回 `pending`，不等待对话结束。
原生 `reminder_delivery` 后台 Job 等待对话结束并连续空闲10秒后投递，期间再次对话则重新计时。
投递前检查当前会话、最新原生轮次、活动时间和有效期；有效期为提交时的一个唤醒间隔。
无会话、会话切换、离开活动时间或过期时废弃。重启从 JSONL 重建待发送状态，并重新检查和等待。
重试复用原生消息 ID；`sent` 表示已注入，不代表主人已读。注入与 JSONL 状态落盘仍是两个操作。

主动 Agent 只装备反馈获取、对话读取、到期任务、提醒历史、提醒提交和 JSON 校验六个 Job 工具。
会话状态查询归通用会话操作，不再放在主动工具目录。相关性、语义去重和提醒价值由 Agent 判断。
长期个人记忆由对话 Agent 维护，近期关注和未完成工作保留在交接摘要；旧对话和摘要不能恢复已遗忘信息。

### 定时任务 query

私有工具 `run_due_tasks()` 每次重新读取 `{metadata_dir}/pa/tasks.json`，依次通过 `task_agent(query=...)` 执行到期任务。
文件缺失时从 `components.pa.default.default_tasks` 初始化，已有空列表保留，错误文件报错而不覆盖。任务包含稳定 id、
五段 cron、query、enabled 和代码补齐的登记时间。默认每五分钟观察屏幕活动，每十五分钟寻找主人可能喜欢的财经新闻。
通过原生文件工具编辑，使用 `check_json` 检查语法，加载器校验业务字段；修改 query 或 cron 保留任务 ID 与历史。

新任务从登记后的下一个 cron 时间开始，错过多个周期时只执行最近一期。启动任务 Agent 前，PA 组件先向
`{metadata_dir}/pa/task_runs.jsonl` 追加 running 记录；完成后用相同 run_id 追加结束记录，包含计划、开始及结束时间、
状态、最终解读和任务会话 ID。单个失败不阻塞其他任务，取消时记录中断。后续调用将遗留 running 标记为中断、结果未知，
同一计划时刻不会自动重跑。组件锁串行化工具调用。

返回本轮完成的任务记录列表，没有到期任务返回 `[]`。主动 Agent 每次醒来调用，实际执行延迟取决于唤醒频率和任务耗时。
这替代固定的截图 CronJob；资源 TTL 清理仍为独立 CronJob。

## MCP

ReMe 0.4.2.2 使用 FastMCP 4 与 MCP Python SDK 2。升级时请同步更新 ReMe 和宿主侧 MCP bridge；
独立 bridge 环境需要 `fastmcp>=4.1.4,<5`。CLI 工具列表统一使用 SDK 2 的 snake_case 字段，
包括 `input_schema`、`output_schema` 和 annotation hints；读取旧驼峰字段的调用方需要同步更新。

默认 HTTP 服务会在 `/mcp` 挂载 streamable HTTP MCP：

```yaml
service:
  backend: http
  mcp_enabled: true
  mcp_path: /mcp
  mcp_stateless_http: false
```

也可以改用独立 MCP Service：

```bash
reme start service.backend=mcp service.transport=stdio
reme start service.backend=mcp service.transport=sse service.port=2333
reme start service.backend=mcp service.transport=streamable-http service.port=2333
```

MCP tool 来自 `enable_serve: true` 的非流式 Job。`service.jobs` 可以设置允许列表；`injected_job_kwargs` 可以注入服务端管理、调用方不能覆盖的参数。

```yaml
service:
  backend: http
  jobs: [search, read, traverse, auto_memory]
  injected_job_kwargs:
    tenant_id: local-user
  tool_error_on_failure: true
```

## ReMe Studio

安装 `reme-ai[web]` 或 `reme-ai[core]` 后，默认 HTTP 地址同时提供 ReMe Studio：

```text
http://127.0.0.1:2333/
```

可通过配置关闭或指定自定义构建：

```yaml
service:
  web_enabled: false
  # web_static_dir: /absolute/path/to/static
```

静态资源缺失不会阻止 Job API 启动。

## 一次性 Job

需要脚本式执行而不常驻服务时：

```bash
reme start job=search query="用户偏好" limit=5
```

它会切换到一次性 CLI Service，但仍经过正常的 Application、Component 和 Job 生命周期。

## 服务发现

```bash
reme find_reme
```

ReMe 会记录本机运行服务的启动参数。普通 `reme <action>` 优先使用实际运行服务的 backend、host、port 和 transport；找不到运行服务时才回退到本地配置。

## 安全边界

ReMe 默认定位为本地服务：

- 默认绑定 `127.0.0.1`；
- HTTP CORS 配置允许任意 origin；
- Job 可进行文件写入、移动和删除；
- 当前服务层不提供通用用户认证。

远程访问必须通过 `reme start service.host=0.0.0.0` 显式启用。不要直接把服务暴露到公网。请在受控网络或带
身份认证、TLS、访问控制和请求大小限制的反向代理后部署，并通过 `service.jobs` 只开放必要 Job。

## OpenAPI

HTTP 服务使用 FastAPI。启用 HTTP 服务时，可以通过 `/docs`、`/redoc` 和 `/openapi.json` 查看当前配置实际注册的端点；Studio 的 SPA fallback 不会覆盖这些保留路径。
