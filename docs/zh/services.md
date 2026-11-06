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

`proactive` 配置设置 `service.timeout_graceful_shutdown: 5`。HTTP 服务退出时，先给正在处理的请求五秒完成，
随后取消仍未结束的请求，再关闭应用组件，避免长订阅无限阻塞退出。自定义 HTTP 配置启用长订阅时，也应设置有限的
优雅退出超时。

PA 的 Prompt 提供实际配置的工作区和 `{digest_dir}/personal/REME_MEMORY.md` 路径；Agent 按需读取当前记忆，
更新前先读取最新完整文件，再合并修改，不再将记忆正文注入会话 Prompt。新建会话或执行主动任务时，公共逻辑仅在文件
不存在时，用 `reme/steps/proactive/pa.yaml` 的 `default_memory` 初始化，通过逐路径锁和原子写入保护并发。
默认内容覆盖财经新闻、AI 前沿和 ReMe 近况，区分已知关注和可探索方向；已有文件（包括空文件）保持原样，
不再进行首次信息询问或根据空正文跳过任务。

Remy 的人设、记忆维护与遗忘说明同样由 `PromptHandler` 从该 YAML 加载，提供英文默认版本和中文版本。
proactive 默认 `language=zh`，可用 `language=en` 切换英文；Step 的 `language` 优先于 Application。
通过 Step 的 `prompt_dict` 覆盖对应语言的键（如 `system_prompt_zh`、`default_memory_zh`）；默认记忆覆盖仅影响缺失文件。
主人要求遗忘时，提示 Agent 删除对应内容，不转存，也不从旧会话或摘要中恢复；维护时清理确认过时的状态，保留有效的长期事实。
创建会话会初始化缺失的记忆，但不生成回复。同一会话复用初始化时的 Prompt 和工具，记忆路径始终指向最新文件。
路径无效或初始化失败时，新建失败并保留原来的当前会话。

`proactive` 配置启动串行后台循环 `pa_loop`。启动和失败后的等待间隔读取
`jobs.pa_loop.steps[0].default_wakeup_minutes`，默认五分钟。记忆初始化失败时，本轮在处理资源前失败。
每轮使用两个独立的 Agent 会话：

- **收集：**Remy 阅读已确认的记忆、可选的 `{digest_dir}/personal/INTERESTS.md` 和新增资源包。
  处理资源前先阅读当前 PA 会话文件中的近期任务、要求和反馈；会话文件缺失或不可读时暂缓，不确认资源。
  尚无当前 PA 会话时跳过收集。
  Prompt 决定相关性、去重、网页搜索和兴趣探索；有用的新发现每轮最多写入一篇
  `{daily_dir}/interest_HHMMSS.md`，重名追加数字后缀。代码补齐 `unsent` 状态和带时区的创建、更新时间。
  完整判断后才通过 `resource_processed` 确认资源，有价值的资料必须先保存成功。
- **提醒：**Remy 阅读兴趣文档和原生 PA 会话文件，调用 `session_status`，判断时机和话术。
  本轮没有新发现时仍评估历史未推送内容。私有 Job `send_reminder(message, files)` 接受一段话术和一个或多个
  带 `.md` 后缀的兴趣文件名，目标会话由代码绑定。Agent 对这些文件只读；Job 校验所选文件和对话未改变，
  通过会话 owner 恢复未加载的会话，确认空闲后注入 assistant 消息，成功后将未被修改的对应文件标记为 `sent`，
  不代表主人已读。

主人明确表达的信息保存在 `{digest_dir}/personal/REME_MEMORY.md`，由前台对话维护；收集 Agent 将兴趣推测
保存在 `INTERESTS.md`，按主题记录日期、来自主人的依据和未确定的范围。自身搜索、推荐、重复记录和主人的沉默
都不能确认兴趣。两个角色的 Prompt 均按角色、上下文信息、工作流和注意事项组织，位于 `reme/steps/proactive/`。
Agent 工具集中在 `proactive/tools/`，一个工具一个模块；资源采集实现及其公共基类统一放在
`proactive/data_connectors/`。工具注册名和 Job 参数不依赖这些模块路径。

投递直接调用会话 append 接口，同一轮重试复用同一个消息 ID。文件锁和版本校验保护并发修改，单个文件被修改或
无法回写时，不阻塞其他文件和后续轮次。不保留跨轮投递日志；若 append 成功后、文件状态保存前进程崩溃，可能重复
提醒。运行时仅支持 `unsent`、`sent`，移除原来的 `ready` 流程和按日期分目录的兴趣文档布局，不自动移动或重写
工作区已有数据。

Remy 可调用私有 Job `set_next_wakeup(minutes=...)`，仅调整下一次等待；未设置、本轮失败或服务重启时恢复配置默认值。
`pa_proactive`、`send_reminder` 和 `set_next_wakeup` 均为私有 Job，通过后台 Agent 的工具绑定使用。
关闭后台提醒可配置 `jobs.pa_loop.enabled=false`。

MCP 不暴露 Stream Job。

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
