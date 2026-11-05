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
各对话独立。使用 `config=proactive` 启动后，只提供三个 `pa_*` Job，不再提供 `chat`。前端按服务实际开放的 Job 启用对应入口。

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

每个新 PA 会话都会将工作区的 `personal/REME_MEMORY.md` 加入配置的 Prompt。文件不存在或正文为空时，提示
Agent 在首次回复中自然了解用户的偏好、当前状态、关注事项和项目，并根据确认的信息创建文件；后续出现新增或变化时，
要求先读取最新文件，再合并更新，保持内容简洁、完整、有效。PA 人设、记忆、首次询问和遗忘提示词统一由 `PromptHandler`
从 `reme/steps/proactive/pa.yaml` 加载，提供英文默认版本和中文版本。proactive 默认 `language=zh`，可用 `language=en`
切换英文；Step 的 `language` 优先于 Application。定制时使用 Step 的 `prompt_dict` 覆盖对应语言的键（如 `system_prompt_zh`）。
加载的记忆正文保留原文。用户明确要求遗忘时，提示 Agent 删除对应内容，不得转存，也不得从旧会话、摘要或记忆副本恢复。
维护时还应清理确认失效的状态、被替代的偏好、不再关注的事项及没有长期价值的已结束项目细节，保留仍有效的长期事实。
创建会话本身不生成回复，也不写入记忆。
同一会话复用初始化时的 Prompt，新会话重新读取文件。文件必须完整读取且不超过 50 KiB；读取失败或路径逃出工作区时，
新建失败并保留原来的当前会话。

`proactive` 配置还启动 `pa_loop` 后台任务：启动及重启后先等五分钟，每轮先发现用户可能感兴趣的资源，再结合当前
PA 会话决定是否提醒。两个角色的实现与双语提示词位于 `reme/steps/proactive/`。后台要求存在非空的
`personal/REME_MEMORY.md`；缺失或读取失败时本轮失败，五分钟后重试，不调用 agent 或确认消费资源。
发现 agent 自行调用 `resource_changes` 和 `resource_processed`：无关资源完整判断后确认，相关资源成功保存后确认，失败项留待后续。
文档保存为 `{daily_dir}/YYYY-MM-DD/interest_HHMMSS.md`，同秒重名追加数字后缀。frontmatter 的 `status` 为
`unsent`、`ready`、`sent` 或 `discarded`；代码补齐 ID、创建和更新时间，也会恢复缺失元数据的文档。
发现角色只能创建指定输出路径，必须保留已有文档；推送角色只能修改已有候选的 `status` 和 `message`，越界修改会使本轮失败。
推送 agent 每轮评估所有日期的历史候选，获得当前会话 ID、原生会话文件路径、状态与近期消息；没有新文档也会继续评估旧候选。
每轮最多一篇 `ready`，提醒正文写入 frontmatter 的 `message`。代码投递前复查目标会话和对话轮次，通过稳定消息 ID
恢复中断的投递，拒绝覆盖并发修改过的文档，并记录 `target_session_id`、`sent_at` 等元数据；`sent` 表示写入会话成功，不表示用户已读。
推送 agent 可调用私有 Job `set_next_wakeup(minutes=...)`，通过 `ApplicationContext.metadata` 调整本轮结束后的下一次等待。
该设置消费一次，未设置、执行失败或服务重启时恢复五分钟。`pa_proactive` 和 `set_next_wakeup` 不暴露为 HTTP/MCP 服务接口；
后者只通过后台 agent 的工具列表使用。关闭后台提醒可配置 `jobs.pa_loop.enabled=false`。

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
