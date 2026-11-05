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

Studio shares one conversation UI between **Chat** and **Proactive Chat**. The default configuration exposes only
`chat`, with `query` and an optional `session_id` for independent conversations. Start with `config=proactive` to expose
the `pa_*` Jobs instead of `chat`. Entries are enabled according to the service's exposed Jobs.

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
session creation reject concurrent operations. Status queries do not wait for generation. PA Jobs currently require
`CodexAgentWrapper`; other wrappers raise `NotImplementedError`.
If the current native session is unavailable, `pa_session` returns `status: "error"`; create a new session to recover.
Once empty-session creation starts, selecting that session completes before cancellation is propagated.

`proactive` also exposes read-only `POST /pa_events` for background clients. With the service running, subscribe from
another terminal:

```bash
curl -N http://127.0.0.1:2333/pa_events \
  -H 'Content-Type: application/json' \
  -d '{}'
```

Each SSE `data:` frame contains a `StreamChunk`; its `chunk` is a typed
[`PaEvent`](../../reme/schema/pa_event.py). A first connection receives `snapshot`, containing the current session ID,
merged PA activity, and optional `SessionSnapshot` history. No session is created. Subsequent events are:

| Type | Client action |
| --- | --- |
| `session_changed` | Select the new session and clear the previous session's displayed history. |
| `message_upsert` | Replace or insert the complete `message`, keyed by `(session_id, message.id)`. |
| `status_changed` | Update activity; PA operations remain active until their serialization lock is released. |
| `reset` | Discard the old cursor; a fresh `snapshot` follows on the same connection. |
| `heartbeat` | Keep the connection alive; this event has no cursor. |

Save an event's opaque `cursor` after applying it. Reconnect with `{"cursor":"<saved-cursor>"}` to replay retained
changes before waiting for live updates. Process/runtime restart, upstream disconnection, or falling behind the bounded
buffer requires a reset. Snapshots and replay handoff are version-checked to avoid missing a concurrent switch or commit.
A failed history read produces a snapshot with `error`, no history, and no cursor; retain the existing display while it
retries at the heartbeat interval. Incremental delivery resumes after a successful snapshot. Clients should tolerate
repeated message upserts. Only completed proactive messages should
enter a reminder bubble queue; token streaming continues through `pa_chat`.

Message events follow successful proactive-message publication or a readable native history snapshot after a chat turn
settles, including cancellation. Files remain the recovery source; event buffers and cursors are in memory and do not
represent delivery/read receipts. Configure retention and heartbeat through `components.pa.default.max_events`,
`max_buffer_bytes`, and `heartbeat_seconds`. `jobs.pa_events.stream_queue_size: 1` bounds the HTTP output queue so slow
clients reset from history instead of accumulating unlimited pending events. Other stream Jobs keep their existing
unbounded default (`stream_queue_size: 0`). Custom configurations enabling `pa_events` must include the `pa` component
from `proactive.yaml`. The subscription follows that component's configured Codex wrapper.

The `proactive` config sets `service.timeout_graceful_shutdown: 5`. During HTTP shutdown, requests still active after
five seconds are cancelled before application components close, so an open subscription cannot hold shutdown
indefinitely. Custom HTTP configurations with long-lived subscriptions should also set a finite shutdown timeout.

PA prompts contain the configured workspace and `{digest_dir}/personal/REME_MEMORY.md` paths. The agent reads current
memory as needed and reads the complete latest file before merging updates. Memory contents are not injected into the
session prompt. Before a new session or proactive round, shared code initializes a missing memory file using
`default_memory` from `reme/steps/proactive/pa.yaml`, with per-path locking and atomic writes. The defaults cover financial
news, AI advances, and ReMe, distinguishing known interests from exploration directions. Existing files, including empty
ones, are preserved; there is no onboarding or empty-memory gate.

The same YAML supplies Remy's persona and memory maintenance/forgetting instructions through `PromptHandler`, with
English defaults and Chinese variants. The proactive config defaults to `language=zh`; use `language=en` for English
or a Step's `language` to override the application language. Customize localized keys through the Step `prompt_dict`
(for example, `system_prompt_zh` and `default_memory_zh`). Default-memory overrides only affect missing files.
Explicit forgetting requests instruct the agent to remove matching content without copying it elsewhere or restoring
it from old conversations or summaries. Maintenance removes confirmed obsolete state while retaining valid long-term
facts. Creating a session initializes missing memory but does not generate a reply. The session reuses its initial
prompt and tools; the memory path continues to refer to the latest file. Invalid paths or initialization failures
leave the current session selection unchanged.

The `proactive` config starts a serial `pa_loop`. Its startup and fallback delay comes from
`jobs.pa_loop.steps[0].default_wakeup_minutes` (five minutes by default). Memory initialization failures fail the round
before resource processing. Each round uses two fresh agent sessions:

- **Collect:** Remy reads confirmed memory, optional `{digest_dir}/personal/INTERESTS.md`, and changed resource packages.
  Before processing resources, it reads the current PA session file for recent tasks, requests, and feedback;
  missing or unreadable session files defer collection without acknowledging resources. No current PA session skips it.
  The prompt controls relevance, deduplication, web research, and interest exploration. Useful findings go into at most
  one `{daily_dir}/interest_HHMMSS.md`; filename collisions receive a numeric suffix. Code supplies `unsent` status and
  timezone-aware creation/update timestamps. Acknowledge packages through `resource_processed` only after assessment
  and, for useful material, successful saving.
- **Remind:** Remy reads finding notes and the native PA session file, checks `session_status`, and decides whether and
  how to interrupt. Older unsent notes remain eligible when there are no new findings. Private Job
  `send_reminder(message, files)` takes one message and one or more interest filenames, including `.md`; the target
  session is bound by code. The agent keeps these files read-only. The Job verifies the selected files and conversation
  have not changed, restores unloaded sessions through the owner, and verifies idle before appending the assistant
  message. Successful delivery marks unchanged selected files `sent`; it does not indicate that the user read it.

Confirmed user statements belong in `{digest_dir}/personal/REME_MEMORY.md`, maintained by foreground chat. The collector
maintains hypotheses in `INTERESTS.md`, grouped by topic with dates, user evidence, and uncertainty. Its own searches,
recommendations, repeated notes, and the user's silence do not confirm an interest. Prompts for both roles follow
role, context, workflow, and notes, and live beside their implementations under `reme/steps/proactive/`.
Each agent tool has its own module in `proactive/tools/`; resource collectors and their shared base live in
`proactive/data_connectors/`. Tool registration names and Job parameters are independent of these module paths.

Delivery calls the session append API directly, reusing one message ID for retries within the live round. File locks
and version checks preserve concurrent edits; a changed or unwritable note does not block other notes or later rounds.
There is no cross-round delivery journal. A crash after append succeeds but before note status is saved may cause a
repeat reminder. Runtime state supports only `unsent` and `sent`; the former `ready` workflow and dated note layout are
removed. Existing workspace data is not moved or rewritten automatically.

Remy may call private Job `set_next_wakeup(minutes=...)` to override only the next delay. Omission, failed rounds, and
restart use the configured default. `pa_proactive`, `send_reminder`, and `set_next_wakeup` are private Jobs bound to the
background agent's tools. Disable background reminders with `jobs.pa_loop.enabled=false`.

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
