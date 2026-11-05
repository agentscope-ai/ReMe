---
title: Services and Deployment
description: Run ReMe through HTTP, SSE, MCP, the CLI, and ReMe Studio while respecting its local security boundary.
---

# Services and Deployment

ReMe can run as a local HTTP service, a standalone MCP server, or a one-shot CLI Job. By default HTTP binds to
`127.0.0.1:2333`; one process serves JSON, SSE, streamable HTTP MCP, and optional ReMe Studio.

## HTTP API

Start the service in one terminal and leave it running. Choose one of these startup commands:

```bash
reme start
reme start service.host=127.0.0.1 service.port=8181
```

From another terminal, call a Job through the CLI client:

```bash
reme search query="user preferences" limit=5
```

Regular Jobs also become `POST /<job-name>`; this HTTP example uses the default port `2333`:

```bash
curl -s http://127.0.0.1:2333/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"user preferences","limit":5}'
```

Job arguments live at the request body's top level. A regular response follows the `Response` schema:

```json
{"success": true, "answer": "...", "metadata": {}}
```

Unhandled Step failures become unsuccessful responses.

## Streaming and SSE

Jobs with `backend: stream` also use `POST /<job-name>`, returning `text/event-stream`. Error paths emit an error chunk and always terminate the stream. The default `chat` Job is streaming:

```bash
curl -N http://127.0.0.1:2333/chat \
  -H 'Content-Type: application/json' \
  -d '{"query":"Summarize my long-term preferences"}'
```

Studio shares one conversation UI between **Chat** and **Proactive Chat**. The default configuration exposes only
`chat`, with `query` and an optional `session_id` for independent conversations. Start with `config=proactive` to expose
the three `pa_*` Jobs instead of `chat`. Entries are enabled according to the service's exposed Jobs.

Proactive Chat uses the application-owned `pa_chat` stream with only `query`. `pa_session` takes no parameters and returns
`{"session_id": "...", "status": "idle"}` in `answer` (both values are null before a session exists). `pa_new_session` takes
no parameters, creates and selects an empty session, and returns its ID in `answer` without model generation.

Start the service with `reme start config=proactive` in one terminal, then use another terminal:

```bash
reme pa_session
reme pa_chat query="What should I follow up on?"
reme pa_new_session
```

The current session is shared within one Application and resets on restart; session files remain on disk. Chat and
session creation reject concurrent operations. Status queries do not wait for generation. Empty-session creation is
supported by AgentScope and Codex; other wrappers fail explicitly.
If the current native session is unavailable, `pa_session` returns `status: "error"`; create a new session to recover.
Once empty-session creation starts, selecting that session completes before cancellation is propagated.

MCP does not expose Stream Jobs.

## MCP

ReMe 0.4.2.2 uses FastMCP 4 and MCP Python SDK 2. Upgrade ReMe and host-side MCP bridges together;
standalone bridge environments require `fastmcp>=4.1.4,<5`. CLI tool catalogs use SDK 2 snake_case fields,
including `input_schema`, `output_schema`, and annotation hints. Update consumers that read the old camelCase fields.

The default HTTP service mounts streamable HTTP MCP at `/mcp`:

```yaml
service:
  backend: http
  mcp_enabled: true
  mcp_path: /mcp
  mcp_stateless_http: false
```

For a standalone MCP service:

```bash
reme start service.backend=mcp service.transport=stdio
reme start service.backend=mcp service.transport=sse service.port=2333
reme start service.backend=mcp service.transport=streamable-http service.port=2333
```

MCP tools come from non-stream Jobs with `enable_serve: true`. Use `service.jobs` as an allowlist. `injected_job_kwargs` adds server-managed values that callers cannot override.

```yaml
service:
  backend: http
  jobs: [search, read, traverse, auto_memory]
  injected_job_kwargs:
    tenant_id: local-user
  tool_error_on_failure: true
```

## ReMe Studio

After installing `reme-ai[web]` or `reme-ai[core]`, the default HTTP origin also serves Studio:

```text
http://127.0.0.1:2333/
```

Disable it with `service.web_enabled=false` or select a build with `service.web_static_dir`. Missing static assets do not prevent the Job API from starting.

## One-shot Jobs

```bash
reme start job=search query="user preferences" limit=5
```

This selects the one-shot CLI Service while retaining the normal Application, Component, and Job lifecycle.

## Service discovery

```bash
reme find_reme
```

Ordinary `reme <action>` commands prefer the running service's actual backend, host, port, and transport. They fall back to local configuration only when no service is discovered.

## Security boundary

ReMe is local-first:

- the default binds to `127.0.0.1`;
- HTTP CORS allows any origin;
- Jobs may write, move, or delete files;
- the service layer has no general-purpose user authentication.

Remote access must be enabled explicitly with `reme start service.host=0.0.0.0`. Do not expose the service directly to
the public internet. Place it on a controlled network or behind an authenticated TLS reverse proxy, apply access
controls and request-size limits, and expose only necessary Jobs.

## OpenAPI

FastAPI exposes the active endpoints through `/docs`, `/redoc`, and `/openapi.json`. The Studio SPA fallback preserves these reserved paths.
