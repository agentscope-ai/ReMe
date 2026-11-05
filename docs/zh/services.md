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
