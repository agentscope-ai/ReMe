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

IPv6 literals such as `::1` can be used as the host; ReMe formats them correctly when HTTP or MCP clients connect.
If a service binds to the IPv6 wildcard `::`, in-process clients use the connectable loopback address `::1`.

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

The `proactive` configuration exposes `chat(query)` for the fixed `main` Agent. It resumes the current
native session; `new_chat_session` selects a fresh session and retains the history exposed by `list_chat_sessions`.
MCP does not expose Stream Jobs. Studio integration is separate from this backend.

## Agent Runtime

The `agent_runtime` Component owns sessions, event streams, inboxes and scheduling. `ChatAgentStep` and
`ProactiveAgentStep` define the two Agent behaviors under `reme/steps/proactive/agent/`.
Configure the Component through `components.agent_runtime.default` and the workspace-relative `reme_agent_dir`
(default `reme_agents`). Each Agent owns `config.json`, `MEMORY.md` and `sessions.jsonl`. Event and cron Agents also own
`events/` and `reminder/YYYY-MM-DD/<agent_id>_HHMMSS.md`; temporary Agents use `reme_agents/tmp/<agent_id>/`.
There is no global definition index or checkpoint file: startup rebuilds runtime state from source files.
Session registries preserve complete records across an interrupted append; an incomplete tail is ignored on read
and removed by the next append. Corrupt complete records fail explicitly.
Validated reminders are saved before input consumption is committed; the outbox retries delivery independently.
An interrupted, uncommitted round may run again. Consumption receipts retain reminder references until handoff.

Start the backend in one terminal, with `LLM_API_KEY`, `LLM_BASE_URL` and optional `CODEX_MODEL_NAME`:

```bash
reme start config=proactive
```

From another terminal:

```bash
reme manage_proactive_agent action=list
reme manage_proactive_agent action=create \
  'config={"name":"experiment","trigger_type":"event","description":"Monitor experiment logs and report failures or completion."}'
reme manage_inbox action=read agent_id=main
reme manage_inbox action=mark_read agent_id=main 'msg_ids=["MESSAGE_ID"]'
```

`manage_proactive_agent` supports `list/create/status/update/delete/query`. The creation name is the immutable
`agent_id`; `main` and `tmp` are reserved. Deletion stops owned work and retains files, events and native session history.
Creating the same name restores that Agent. `query` asks a temporary read-only fork without selecting it for future work.

`config.description` is creation-only: it initializes the Agent's `MEMORY.md`, with sections for the current task,
confirmed background and long-term findings. Task text is not stored in `config.json` or injected separately each round.
`list/update` return settings, status and workspace-relative `memory_path`; `create/status` also return `memory_content`
(up to 2,000 characters) and `memory_truncated`. Read the complete file when truncated. Configuration file paths are
not returned. Before changing or deleting a task, inspect `status`; edit task requirements directly in `memory_path`,
replace superseded requirements and read back the result. Use `update` for execution settings and `delete` to stop
owned work. Changes apply to subsequent rounds; an already-running round may have read the previous task.
Same-name restoration preserves existing memory. To migrate an older Agent, move its stored `description` into the
current-task section of `MEMORY.md`, retain its accumulated knowledge, then remove `description` from `config.json`.
`reme_agent_dir` now defaults to `reme_agents`; set it explicitly to `agents` to keep using an existing directory.
No workspace files or embedded references are moved automatically.

Chat's initial memory is configured through `components.agent_runtime.default.main.description`. Runtime startup
initializes its `MEMORY.md` using the same mechanism as task Agents; existing memory survives restarts and session
changes. Chat's `job_tools` are selected in its Step configuration.

Initial tasks can be declared in `components.agent_runtime.default.agents` in `proactive.yaml`, using the same creation
fields including `description`. Startup creates missing identities only; restarting does not overwrite edited memory
or settings, or restore deleted tasks. Two tasks are enabled by default: `finance` checks for major AI news hourly,
and the event-driven `work` tracks the user's current projects, progress, difficulties and discussion points from
incoming observations. Screen capture targets `work` when explicitly enabled; the collector remains disabled by default.

Task configuration uses `trigger_type` (`cron` or `event`). Rename the old `type` key to `trigger_type` in existing
Agent `config.json` files and API calls; the old key is no longer accepted.

An `event` Agent accepts observations through `append_event(agent_id, events)`, with `event_ts`, `source`, `description`
and optional workspace-relative `paths`, `id` and `metadata`. A `cron` Agent requires a cron expression and executes
its full task at the scheduled time. Both use the application timezone and a serial executor per Agent.
`source=cron` is reserved. Event tasks may set `min_trigger_interval_seconds`.

Chat and cron Agents enable native `web_search`, `web_extract` and command network access; event Agents stay offline.
Remove the obsolete `enable_web` field from existing task configurations and creation calls. `web_extract` saves complete
text or PDF files under `resource_dir` and returns paths plus a bounded text preview; it does not execute JavaScript
or extract PDF text. Temporary `query` forks remain read-only and offline.
`notify_window={start,end}` limits automatic inbox wakeups; messages can still be processed and saved outside it.
`temporary=true` is independent of trigger type. Temporary Agents become eligible for collection after seven days
without new accepted inputs; cron triggers count as inputs, while duplicates and retries do not. Collection waits for
running work, pending events, undelivered messages and retained Inbox references. Daily records and resources remain.
On restart cron schedules the next future trigger; queued events still run, but missed offline triggers are not backfilled.
Chat idle and retry timers restart in memory. Daily handoffs use `daily/YYYY-MM-DD/<agent_id>.md`; round details use
`<agent_id>_HHMMSS.md`, with numeric suffixes for same-second collisions. The framework keeps each reminder body in Markdown and
updates delivery, wakeup and read timestamps in its frontmatter under the Agent's `reminder/YYYY-MM-DD/<agent_id>_HHMMSS.md`.

Only `main` receives messages. Its `inbox_unread.jsonl` and `inbox_read.jsonl` contain only `{file,id}` references;
the read view retains the latest 1,000. Bodies and timestamps remain in canonical reminders. One lifecycle-owned
watcher and periodic rescans reconcile file edits and interrupted handoffs. Appending a valid reference to the read
view marks the original message read. Missing views are rebuilt from reminders.
Existing daily reminder JSONL logs require migration to Markdown frontmatter before startup.
`manage_inbox(action="read")` leaves messages unread, defaults to `unread_only=true`, and does not filter priority
unless `priority` is supplied. `unread_only=false` includes read history. Results are ordered by priority, then time.
After processing, call `action="mark_read"` with `msg_ids`, including when no user notification is needed.
Repeated marks preserve the original read time. Agent-read is distinct from user-read; there are no claim tokens.
Agent-bound tools infer the receiver; external calls must supply `agent_id`.

P0 requests a Chat turn immediately, or after `components.agent_runtime.default.inbox.p0_delay_seconds` (default 0).
P1 waits for continuous Chat idle (default 10 seconds); P2 may be read on demand or wakes Chat after 1800 idle seconds.
The latter delays are `inbox.p1_idle_seconds` and `inbox.p2_idle_seconds`. User and automatic turns share one lock;
P0 waits behind an active turn. Every completed, failed or cancelled Chat turn restarts the idle clock; message arrival
does not. A successful P0 wakeup is recorded separately from reading, preventing repeated wakeups for the same batch.
Chat promptly notifies for P0 and uses conversation context for P1/P2, finishing silently when no reminder is needed.

`proactive_run`, `proactive_query` and `inbox_chat` are service-private Jobs. The inbox background worker starts actual
Chat inference instead of appending reminder text as an assistant answer. The former direct-delivery and sender-review
paths are removed. Collectors remain opt-in; configure `jobs.screen_capture_cron.agent_id` and enable the job to publish
complete resource packages as observations. No resource watcher or automatic TTL deletion participates in this flow.

This is a breaking replacement of the previous inbox and PA APIs. Existing workspace files are retained without
automatic conversion. The backend uses CodexAgentWrapper; other wrappers need equivalent session capabilities.

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
