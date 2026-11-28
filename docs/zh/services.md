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

主机地址也可以使用 `::1` 等 IPv6 字面量；HTTP 和 MCP client 会在连接时正确格式化该地址。
如果服务绑定到 IPv6 通配地址 `::`，ReMe 会向进程内 client 提供可连接的回环地址 `::1`。

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

`proactive` 的工作区接口为 `list_workspace`、`search_workspace`、`load_file`、`save_file` 和 `get_client_info`。
Studio 保存时提交完整原文，并将 `load_file` 返回的 `mtime` 作为 `expected_mtime`。文件已变化时返回 `file_conflict`。
默认配置提供 `list`、`search`、`load`、`save` 和 `app_config`。

`proactive` 配置提供 `ask_agent(query, agent_name=Remy)`，使用固定 Chat Agent 的当前原生会话，
`create_chat_session` 创建并选中新会话。`demo` 继承这两个 Job，并提供 `list_chat_sessions`、
`session_read` 和 `session_status` 用于查看历史与原生会话。MCP 不暴露 Stream Job。

## 主动智能运行时 {#proactive-runtime}

`proactive` 初始化一个 Remy 和四个后台 Runtime：项目记忆、随记跟进、金融、科技早报。
它们位于 `components.agent_runtime`；`components.agent_manager.default` 只管理后台任务。
Runtime 持有会话引用、执行锁、独立 Inbox 和 Outbox，Job 负责单轮执行和调度，Step 每次调用重新创建。
共享的 AgentWrapper 和 App Server 仍由 Application 管理。

使用 Claude Code 时，启动 `reme start config=proactive_cc`。该配置使用 Claude Code 原生默认提示词，
通过追加方式注入 Runtime 的角色指令，包括 Remy 的 Chat 提示词。关闭提示词快照，使续聊使用当前追加的指令。
`proactive` 和 `proactive_cc` 共用 `LLM_API_KEY`、`LLM_BASE_URL`、`REME_MODEL_NAME`。
Claude Code 的地址需要支持 Anthropic API，预设默认使用 `https://dashscope.aliyuncs.com/apps/anthropic`；
Codex 使用 Responses API。环境变量用于首次导入，之后使用服务端保存的配置。
这两种预设可在 Studio「设置 → 模型」选择 Codex 或 Claude Code，测试连接后保存并重启。
Chat 和主动任务共用所选模型。切换执行后端会开始新的会话，原有会话文件保留。

```bash
# 终端一：启动服务
reme start config=proactive
# 终端二：查询或创建任务
reme manage_proactive_runtime action=list
reme manage_proactive_runtime action=create agent_type=event agent_name=experiment_watch \
  task_description='跟进实验日志；异常和结束时提醒，正常进展只记录。'
```

`manage_proactive_runtime` 支持 `list/create/status/update/delete/query`。所有后台任务共用私有 Job `run_proactive_agent`。
预设 Runtime 的 `initial_task`、`skills`、`job_tools` 和会话选项统一配置在 `components.agent_runtime` 下。
Runtime 只在 TASK.md 不存在时写入初始文本，每轮执行读取当前文档，保留用户修改。
动态创建支持 `task_description`、`skills`、`job_tools`，由 AgentManager 应用 `runtime_defaults` 并检查
`allowed_job_tools`；工具的可信注入约束由部署配置指定。触发类型和会话绑定创建后固定，同名恢复时保留。
动态运行配置仍保存在工作区的每任务 config.json 中。旧领域 Job 名称和旧配置结构不再支持，升级时更新任务配置，保留 TASK.md 和历史文件。
创建、状态和更新返回原始运行信息：运行设置在 `config`，`running` 表示正在执行；列表返回 `items/total`。
客户端 Agent 视图仍由 `list_agents` 和会话接口提供。

创建时必须传 `agent_type=event` 或 `agent_type=time`。时间任务必须设置 `schedule` Cron 表达式，按应用时区执行；事件任务没有日程。
创建后触发类型、skills 和工具绑定固定。`update` 仅修改 enabled、schedule、
min_trigger_interval_seconds、notify_windows。暂停让当前轮完成，删除取消执行并保留工作区文件；同名恢复沿用工具绑定和历史。
读取或修改任务要求使用 `task_document action=read/save agent_name=...`，保存必须提交读取返回的 `mtime` 作为 expected_mtime。

`notify_windows` 使用应用时区的 `{start,end}` 时段列表，格式 HH:MM，默认
`[{"start":"09:00","end":"12:00"},{"start":"14:00","end":"23:00"}]`。命中任一时段即可提醒，`[]` 表示不限时段。
含起点、不含终点，可跨午夜，起止相同表示全天。只控制提醒处理和发送，不改变任务执行时间。
已有任务的 config.json 需要使用此列表字段。

Remy 的个人记忆通过 `components.agent_runtime.Remy.initial_memory` 创建到 `{reme_agent_dir}/Remy/MEMORY.md`，
已有内容保留。`components.agent_runtime.Remy.notification_config` 控制 P0 延迟（默认 0 秒）、P1 空闲（10 秒）、P2 空闲（1800 秒）。
P0 可以进入正在执行的前台原生轮次；P1/P2 等待前台空闲。发送方 enabled 和 notify_windows 仍限制提醒资格。

### 资料接入与采集控制

先启动 `reme start config=proactive`，再通过 HTTP 或 MCP 调用 `manage_connector`。
预设采集器和自定义 SSE 订阅使用同一来源视图；预设 ID 为 `preset:<job-name>`，订阅 ID 为 `source_<uuid>`。

| 动作 | 输入和行为 |
| --- | --- |
| `list` | 只传 action；通过 items/total 返回统一来源视图。 |
| `create` | name、非空 runtime_names，以及 `binding: {url: "https://example.com/events"}`；固定 SSE/CloudEvents，不需要 format、trigger_type。 |
| `update` | source_id，以及 name、runtime_names、enabled 中至少一项；未传字段保持原值，不接受 null 修改。 |
| `send` | runtime_names 和非空 events；按目标返回 deliveries、rejected、errors。 |

runtime_names 完整替换目标；[] 表示继续采集、不投递新事件。目标必须已存在、为事件型且忽略大小写不重复。
`update enabled=false` 暂停，`enabled=true` 恢复；改名不改变身份。已保存批次保留原目标和来源名称，
来源开关不修改 Runtime 执行开关。旧的 enable/pause 动作和 connector_job 请求参数已移除。

```json
{"action":"update","source_id":"preset:screen_capture_cron","name":"项目观察","enabled":false}
```

采集器先发布完整工作区资料包，再投递 Event。SSE GET 接口返回 text/event-stream，每条 data 是一个
CloudEvents 1.0 JSON 对象。原文和首次接收时间保存在 `{resource_dir}/cloudevents`，订阅定义和投递检查点
保存在 `{metadata_dir}/proactive/sources`。事件身份由 CloudEvents source + id 决定，同身份内容变化会报错。
配置成功不等于已连接、投递成功或已处理；统一来源视图区分 enabled、available、connected、待投递数和错误。
手工修改订阅定义后需重启加载，更新接口会拒绝覆盖尚未加载的改动。

重启携带已提交的 Last-Event-ID 恢复，历史补发依赖上游。错误或拒绝保留待投递事件并阻止游标推进。
HTTP 204 停止重连，直到显式恢复或重启。SSE 单行及事件缓冲上限为 1 MiB；不配置自定义鉴权头或平台特有握手。
自建采集脚本继续通过 action=send 投递工作区相对路径的 Event，source=time 保留给框架。
时间来源直接从 Runtime 计划生成；图标和 emoji 属于前端配置。

### 提醒提交与恢复

后台任务使用 EventInbox，输出经 NotificationOutbox 交给 Remy 的 NotificationInbox；Remy 用 ChatOutbox 向原生用户会话追加回复。
任务成功完成且记录校验通过后才确认输入。文件头只包含提醒决策 reminder_priority（P0/P1/P2/skip）及 reminder_message；
模型失败或超时保留输入和草稿。格式错误可在同一会话修复一次；输入回执落盘后，通知交接可在重启后恢复，无需重新推理。

通知队列持久化源记录引用，正文及等级始终来自 daily Markdown。接收方 InboxReceipt 保存 send/skip 处理事实，
发送方 Outbox 只保存交付事实；ChatOutbox 关联原生消息。`read_at` 表示 Remy 已处理，不表示用户已读。
提醒分支只读源记录，通过 `submit_chat_reply(action="send", content=...)` 或 `action="skip"` 提交；skip 不传 content。
提交前重新检查源文件、前台轮次、原生会话版本及发送方设置，过期决定不能发送或确认。原生写入响应丢失时先查固定消息 ID，再恢复确认。

`agent_manager_loop` 调度后台任务，`chat_notifications_loop` 调度提醒，二者均为 BackgroundJob。
`run_chat_agent`、`run_proactive_agent`、`process_chat_notifications` 和 submit_chat_reply 均不对服务公开。
前台 ask_agent 负责外层流的 DONE；后台任务问答使用最新任务会话的临时只读 fork，不消费队列，也不修改任务主会话。

Runtime 组件从 `components/proactive/runtime/skills/` 加载内置技能模板，通过 PromptHandler 选择语言；也可按名称引用项目技能。
Chat 与 Proactive Runtime 分别配置自己的 skills 和工具。生成的 SKILL.md 限于本次调用使用，结束后清理。
Wrapper 的 skills 使用 SkillSpec 列表：已有技能用 name，生成技能使用 name、description、content；原生会话协议保持不变。
生成技能的名称由操作系统文件锁保护；进程异常退出后，下次调用会恢复遗留链接。
执行 Job 由工作流固定。Runtime 定义不再包含 `run_job`，提醒循环不再接受 `process_job`；迁移旧配置时删除这些字段。
公共提示词加载器位于 `reme/utils/prompt.py`，Runtime 初始化不再反查执行 Step。
本轮 query 和输入模板位于 `reme/steps/proactive/agent.yaml`，通过 `jobs.run_proactive_agent.steps[0].prompt_dict`
定制，使用 `query`、`record_instruction`、`event_input`、`time_input` 及对应的 `_zh` 键。
Runtime 的 `prompt_dict` 用于角色和初始化文本。
Notebook 自动写入限制由可信 Runtime 配置注入，在 manage_notebook 工具中检查。

源码见 [组件](../../reme/components/proactive/)、[工作流](../../reme/steps/proactive/) 与 [配置](../../reme/config/proactive.yaml)。

### 主动智能总览订阅

先启动 `reme start config=proactive`，再通过 HTTP `POST /watch_proactive_overview` 订阅，
请求为 `{}` 或 `{"agent_names":["proactive_memory"]}`。可选名单筛选后台 Runtime，共享 Remy 自动包含。
该只读 StreamJob 不执行模型、不创建原生会话、不领取输入，也不确认提醒；现有会话和详情 Job 继续保留。

每个 `data` 类型的 StreamChunk 携带以下一种事件：

- `snapshot {epoch, revision, as_of, overview}`：整体替换 `sources`、`agents`、`event_inbox`、
  `remy_inbox`、`reminders`、`workspace` 六部分；`errors` 单独报告无法读取的事实。
- `update {epoch, previous_revision, revision, as_of, changes, transitions}`：按集合和业务身份替换、移除对象，
  change 的 value 为 null 时移除。动作带稳定 ID，关联成功入队、领取批次、提交记录、投递提醒和形成最终消息。
- `heartbeat`：通常每 15 秒发送一次存活信号。错误和结束沿用 StreamChunk 的 `error`、`done` 契约。

每次连接都先返回权威快照；慢客户端落后于有界动作缓冲时也重新接收快照。
版本水位取读取开始时的值，快照读取期间发生的变化仍由后续增量补齐，不要求生产者空闲。
增量的 epoch 和 previous_revision 必须与本地匹配，否则重连；快照历史不播放动画，实时动作按 ID 去重。
Studio 共享客户端提供 `watchProactiveOverview` 和 `ProactiveOverviewModel`，包含重连、版本检查和减少动态效果支持。
订阅共享有界读取投影，内部持久变更和执行活动会及时刷新，外部文件修改最长约五秒校验一次。
Agent 的 `waiting_reason / waiting_until / retry_at` 与调度器共用判断；当前执行提供工具和 turn_id，
执行记录保留本轮输入和产物，等待信息不重复存入每份执行记录。

收件箱待处理数和处理中数互不重叠。入队时间和来源快照由 Inbox 入箱记录保存，独立于原始 Event。
提交回执保留输入、来源、执行和文件关联，包括有记录但没有提醒的情况。
提醒只包括已发送的原生消息，投递、Remy 处理和用户阅读分别表示。
提醒查询区分创建时间和真实投递时间，未确认投递时 delivered_at 为空；delivery 与 remy_processing
分别表示发送方和接收方状态，尚未入队时接收状态为空。
工作区返回原始相对路径，正文和编辑仍使用已有文件 Job。契约见[公共模型](../../reme/schema/proactive_overview.py)。
已登记来源的目标以当前配置为准，历史事件保留原来源快照，不恢复已经移除的连线。
工作区保留配置入口、每个任务最近三条投递的记录、当前/最近执行产物，以及最近列表所引用的文件；
历史投递不会全部加入工作区。完整历史继续通过原有详情和分页 Job 查询。


## MCP

ReMe 使用 FastMCP 4 与 MCP Python SDK 2，宿主侧独立 MCP bridge 环境需要 `fastmcp>=4.1.4,<5`。
CLI 工具列表使用 SDK 2 的 snake_case 字段，包括 `input_schema`、`output_schema` 和 annotation hints。

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

安装 `reme-ai[core]` 后，运行 `reme start config=proactive` 启用 ReMe Studio：

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

### 后端设置

HTTP 模式下，`reme start` 管理 Python 服务进程。重启期间，Studio 仍可通过原地址打开。在「设置 → 后端」可以直接更换工作区。Codex 预设还支持设置 LLM URL、API Key、模型和推理级别。点击「保存并重启」应用配置。模型连接测试在临时工作区运行。新配置启动失败时，守护进程会尝试恢复上一次生效配置。

启用了默认 DashScope 语音组件或 DashScope 搜索步骤的实例还可以设置 DashScope API Key。这项凭据独立于 LLM API Key，保存并重启后供已配置的语音识别、语音合成和搜索使用。设置页只显示凭据是否已配置。

后端设置分为工作区、Harness 与模型、语音三组。缺少核心配置时，Studio 显示首次配置引导，补齐后才能进入。语音可以稍后启用。语音地址使用 DashScope 原生协议，默认是 `https://dashscope.aliyuncs.com/api/v1`，模型和音色填写服务支持的 ID，点击「恢复语音默认值」可重新填入默认值。

首次从启动配置、系统环境变量及找到的最近一份 `.env` 导入用户设置。`.env` 从启动目录向上查找，最多检查五级父目录，导入时系统环境变量优先。之后统一读取实例目录的 `settings.json`。修改模型和语音参数通过设置页保存。Key 字段使用 Fernet 加密，主密钥保存在后端主机的系统凭据库。服务器或容器可以设置独立的 `REME_ENCRYPTION_KEY`，使用 `Fernet.generate_key()` 生成并在后续启动中保持一致。

守护进程在 `~/.config/reme/instances/` 下保存各实例的配置、凭据和重启状态。默认实例由启动目录、配置、监听地址和端口确定。通过 `REME_DAEMON_DIR` 可指定工作区之外的私有目录。Docker Compose 分别挂载 `/state` 和 `/data`。本机回环监听可直接管理。启用远程管理时，设置随机的 `REME_ADMIN_TOKEN` 并通过 HTTPS 连接，在 Studio 提示时输入令牌。单独部署 Web 静态页面时，通过同源反向代理访问 `/admin/`。管理接口独立于 Job 和 MCP。

Studio 和所有管理客户端统一使用 `X-ReMe-Admin-Token: <token>`；不再接受 `Authorization: Bearer ...`，客户端须与后端一起升级。

由代理终止 HTTPS 时，保留公网 `Host` 并设置 `X-Forwarded-Proto: https`。启用 `service.proxy_headers=true`，将 `service.forwarded_allow_ips` 限定为可信代理的 IP 或 CIDR；仅在监听端口只能由可信代理访问时使用 `*`。

[魔搭 Docker 创空间](https://github.com/modelscope/modelscope-skills/blob/main/skills/ms-studio-deploy/SKILL.md#docker-studio-reference)保留 `Authorization` 请求头并要求使用 `7860` 端口。设置 `REME_PORT=7860`、`REME_CONFIG=proactive` 及上述代理配置，将 `REME_ADMIN_TOKEN` 和持久使用的 `REME_ENCRYPTION_KEY` 存入平台 Secrets。

## 随记项目

先用 `reme start config=proactive` 启动服务，再在另一个终端查询项目和明细。

```bash
reme manage_notebook action=list_projects
reme manage_notebook kind=todo action=list project=ReMe scope=all
reme manage_notebook kind=idea action=list project=ReMe scope=all
```

`manage_notebook action=list_projects` 从 Todo 和 Idea 记录汇总项目，返回项目总数 `total` 和列表 `items`，
每项包含 `project`、`todo_count` 和 `idea_count`。默认 `scope=all`，包含活跃和归档记录。传入
`scope=active` 或 `scope=archive` 可按范围统计。使用返回的项目名查询该项目的 Todo 或 Idea 明细。

Todo 默认 `pending`（待开始），开始后可设为 `in_progress`（进行中）。完成或取消时设为 `closed` 并自动归档，
可同时用 `close_note` 记录结果或原因。重新打开会清除本次关闭时间和备注。

活跃待办默认按计划日期排序。用 `reme manage_notebook kind=todo action=list order=created` 可按创建时间倒序查看，旧待办缺少创建时间时排在末尾。
Studio 的待办视图也可以切换这两种顺序。

`proactive_note` 由事件触发，与 `proactive_memory` 接收同一份新增 Codex 和 Claude Code 对话。随记使用 `manage-notebook` 查找相关
Todo 和 Idea，先读完 Todo 的进展，再按新依据追加、更正或撤回进展，也可以把已开工的 Todo 改为 `in_progress`。
条目的新增、关闭、Idea 修改和关联变更交给 Remy。需要用户决定的事项逐条写入 Markdown，说明条目、新依据和下一步，再按统一流程投递。

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
