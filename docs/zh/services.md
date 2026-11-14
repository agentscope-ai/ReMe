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

`proactive` 配置提供 `chat(query)`，使用固定 `main` Agent 的当前原生会话。
`new_chat_session` 创建并选中新会话，历史由 `list_chat_sessions` 查询。MCP 不暴露 Stream Job，Studio 另行适配。

## Agent 运行时

`agent_runtime` 组件管理会话、事件流、收件箱和调度；`reme/steps/proactive/agent/` 下的 `ChatAgentStep` 与
`ProactiveAgentStep` 分别定义两种 Agent 行为。组件配置使用 `components.agent_runtime.default`，
工作区相对配置 `reme_agent_dir` 默认是 `reme_agents`。每个 Agent 拥有 `config.json`、`MEMORY.md` 和
`sessions.jsonl`；事件与定时 Agent 另有 `events/` 和 `reminder/YYYY-MM-DD/<agent_id>_HHMMSS.md`，临时 Agent 位于
`reme_agents/tmp/<agent_id>/`。不再维护全局定义索引或检查点文件，启动时从源文件恢复内存。
会话登记保留追加中断前的完整记录；读取时忽略不完整尾行，下次追加时清理尾行，完整记录损坏时明确报错。
先保存校验通过的提醒，再提交事件消费回执，Outbox 独立重试投递；未提交就中断的轮次允许重新执行。
消费回执关联提醒引用，在接收方完成交接前保留对应确认记录。

在一个终端配置 `LLM_API_KEY`、`LLM_BASE_URL` 和可选的 `CODEX_MODEL_NAME`，启动服务：

```bash
reme start config=proactive
```

另一个终端调用：

```bash
reme manage_proactive_agent action=list
reme manage_proactive_agent action=create \
  'config={"name":"experiment_monitor","trigger_type":"event","description":"分析实验日志，失败或结束时提醒我。"}'
reme manage_inbox action=read agent_id=main
reme manage_inbox action=mark_read agent_id=main 'msg_ids=["消息ID"]'
```

`manage_proactive_agent` 支持 `list/create/status/update/delete/query`。创建名称即不可变的 `agent_id`，
`main` 和 `tmp` 是保留名。删除停止调度和执行，保留文件、事件和原生会话历史；再次创建同名 Agent 会恢复它。
`query` 使用临时只读 fork 回答问题，不改变后续执行的主会话。

`config.description` 只用于创建时初始化 Agent 的 `MEMORY.md`，其中分为当前任务、已确认背景和长期结论；
任务文本不再保存在 `config.json` 中，也不再每轮单独注入。`list/update` 返回配置、状态和工作区相对 `memory_path`；
`create/status` 还返回 `memory_content`（最多 2000 字符）和 `memory_truncated`，截断时继续读取完整文件，
不返回配置文件路径。修改或删除前先查看 `status`；任务要求直接修改 `memory_path`，替换失效要求并读回确认，
运行配置使用 `update`，停止并删除任务使用 `delete`。修改对后续轮次生效，正在执行的一轮可能已读取旧要求。
同名恢复保留已有记忆。旧 Agent 迁移时，将配置中的 `description` 移入 MEMORY 的当前任务部分，保留积累的知识，
再从 `config.json` 删除 `description`。
`reme_agent_dir` 默认值已改为 `reme_agents`；已有目录可显式设置 `reme_agent_dir=agents` 继续使用，
不会自动移动工作区文件或改写其中的路径引用。

Chat 的初始记忆通过 `components.agent_runtime.default.main.description` 配置，Runtime 启动时与任务 Agent
使用同一机制初始化 MEMORY；重启和切换会话保留已有记忆。Chat 的 `job_tools` 统一在 Step 配置中选择。

可在 `proactive.yaml` 的 `components.agent_runtime.default.agents` 中声明初始任务，字段与创建接口相同，
同样用 `description` 初始化 MEMORY。启动只创建缺失身份，重启不覆盖已修改的记忆或配置，也不恢复已删除任务；
默认启用两个任务：`finance` 每小时关注 AI 相关的大新闻；`work` 根据新观察资料跟踪主人正在做的项目，
整理各项目的进度、进展、难点和待讨论问题。截图采集开启后默认投递给 `work`，采集器仍默认关闭。

任务配置使用 `trigger_type`（`cron` 或 `event`）。已有 Agent `config.json` 和 API 调用中的旧字段
`type` 需改为 `trigger_type`，旧字段不再接受。

`event` 通过 `append_event(agent_id, events)` 接收观察，每条包含 `event_ts`、`source`、`description`，
可附工作区相对 `paths`、`id` 和 `metadata`。`cron` 要求 cron 表达式，到点执行完整任务，不接收外部观察。
两类使用应用时区，每个 Agent 串行执行；`source=cron` 保留给框架。事件任务可设置 `min_trigger_interval_seconds`。

Chat 和 cron Agent 开启原生 `web_search`、`web_extract` 和命令网络访问，event Agent 保持离线。
已有任务配置和创建参数中的旧字段 `enable_web` 需要删除。
`web_extract` 将完整文本或 PDF 保存到 `resource_dir`，返回路径和有限文本预览，不执行 JavaScript，也不提取 PDF 文本。
临时 `query` fork 继续保持只读、离线。`notify_window={start,end}` 限制自动唤醒时段，
时段外仍可分析并保存消息。`temporary=true` 与触发类型独立，连续七天没有新接收输入后允许清理；
cron 触发也算输入，重复事件和重试不续期。有执行、待处理事件、未投递消息或保留的 Inbox 引用时暂缓删除，
日记和原始资源保持不变。重启后从下一次未来时间恢复 cron，已入队事件继续处理，不补跑停机期间未入队的触发；
Chat 空闲计时和失败重试计时在内存中重新开始。
日记 `daily/YYYY-MM-DD/<agent_id>.md` 保存累计交接，`<agent_id>_HHMMSS.md` 保存本轮明细，
Agent 自己的 `reminder/YYYY-MM-DD/<agent_id>_HHMMSS.md` 以 Markdown 正文保存唯一的提醒内容，以 frontmatter 保存投递、唤醒和已读时间，
由框架维护；同秒日记明细重名增加数字后缀。

只有 `main` 接收消息。它的 `inbox_unread.jsonl` 和 `inbox_read.jsonl` 只保存 `{file,id}` 引用，
已读视图保留最近 1000 条；不复制正文或状态时间。共用一个生命周期内的文件监听器并定期补扫，恢复修改和中断交接。
向已读视图追加有效引用会更新原提醒的已读回执，丢失的视图从提醒重建。
旧的日级提醒 JSONL 需要先迁移到 Markdown frontmatter，再启动服务。
`manage_inbox(action="read")` 不改变已读状态，默认 `unread_only=true`，
不传 `priority` 时不筛选级别；`unread_only=false` 包含已读历史。结果按优先级、时间排序。
处理后通过 `action="mark_read", msg_ids=[...]` 标记，包括决定不提醒的消息；重复标记保留首次已读时间。
Agent 已读不代表主人已读，不使用 claim token。Agent 内部调用自动绑定身份，外部调用必须指定 `agent_id`。

P0 入箱后立即请求 Chat 处理，可通过 `components.agent_runtime.default.inbox.p0_delay_seconds` 调整延迟，默认 0。
P1 等待 Chat 连续空闲 10 秒，P2 可主动读取或在连续空闲 1800 秒后自动唤醒，分别由
`inbox.p1_idle_seconds`、`inbox.p2_idle_seconds` 配置。用户对话与自动轮次共用锁，P0 在当前轮结束后处理。
每个 Chat 轮次完成、失败或取消后重新计时，单纯入箱不重置 idle。
P0 成功唤醒与已读分别记录，避免同批反复唤醒。Chat 对 P0 优先提醒，对 P1/P2 结合上下文判断，无需提醒时静默结束。

`proactive_run`、`proactive_query`、`inbox_chat` 为服务私有 Job。收件箱后台任务启动真实 Chat 推理，
替换原来的正文直接追加和发送方延后复核。采集器默认关闭，配置 `jobs.screen_capture_cron.agent_id`
并启用任务后，完整资源包才会转成观察事件；本流程不使用资源 watcher 或自动 TTL 删除。

本次重构不保留旧收件箱和 PA 接口，工作区历史文件保留，不自动转换。当前后端使用 CodexAgentWrapper；
其他 Wrapper 需要提供对应的会话能力。

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
