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

The `proactive` configuration exposes `chat(query)` for the fixed `Remy` Agent. It resumes the current
native session; `new_chat_session` selects a fresh session and retains the history exposed by `list_chat_sessions`.
MCP does not expose Stream Jobs. Studio integration is separate from this backend.

## Agent Runtime

The `agent_runtime` Component coordinates sessions, the shared Agent Inbox, outputs and scheduling. `ChatAgentStep` and
`ProactiveAgentStep` define the two Agent behaviors under `reme/steps/proactive/agent/`.
Configure the Component through `components.agent_runtime.default` and the workspace-relative `reme_agent_dir`
(default `reme_agents`). Each Agent owns `config.json`, `sessions.jsonl` and `inbox/`. Remy stores personal memory in
`MEMORY.md`. Event and cron Agents own `TASK.md` and `reminder/YYYY-MM-DD/HHMMSS.md`. Temporary Agents use
`reme_agents/tmp/<agent_name>/`.
There is no global definition index or checkpoint file: startup rebuilds runtime state from source files.
Session registries preserve complete records across an interrupted append; an incomplete tail is ignored on read
and removed by the next append. Corrupt complete records fail explicitly.
Validated reminders are saved before input consumption is committed; the outbox retries delivery independently.
An interrupted, uncommitted round may run again. Consumption receipts retain reminder references until handoff.

Personal Runtime connects Chat, its temporary Reminder fork, a shared Inbox and background tasks through workspace files.
The Life task is an illustrative integration.

<p align="center">
  <img src="../figure/personal-runtime.svg" alt="Personal Runtime: Chat and its temporary Reminder fork, the Inbox, background tasks and shared workspace" width="100%">
</p>

Start the backend in one terminal, with `LLM_API_KEY`, `LLM_BASE_URL` and optional `REME_MODEL_NAME`:

```bash
reme start config=proactive
```

For Claude Code, set `CC_LLM_API_KEY` and the Anthropic-compatible `CC_LLM_BASE_URL`, then run
`reme start config=proactive_cc`. This preset shares the `.reme` workspace and the model selected by `REME_MODEL_NAME`
with `proactive`.
In both presets, Chat, Proactive, and Cron share `xhigh` effort and the `workspace-write` policy. Claude Code
implements it with a Bash sandbox and a separate workspace write restriction for the built-in file tools.
Shell network access is disabled by default. Codex enables live `web_search`. Claude Code uses its default tools,
including `WebSearch` and `WebFetch` when supported. Web tools run outside the shell and depend on provider support.
`proactive_cc` starts a local relay to the configured `CC_LLM_BASE_URL`. For Bailian's native `WebSearch`, the relay
sets `cc_entrypoint=cli` in the search request and passes the upstream response back to Claude Code.
Both presets declare a 983,616-token context window and target automatic compaction at 600,000 tokens. Claude Code's
configured auto-compact window is 633,000 tokens to account for its native reserve. Codex applies its 95% usable-window
setting, and each backend measures context usage with its own accounting.

From another terminal:

```bash
reme manage_proactive_agent action=list
reme manage_proactive_agent action=create agent_name=experiment agent_type=proactive \
  'task_description=Monitor experiment logs and report failures or completion.'
reme agent_inbox
```

`manage_proactive_agent` supports `list/create/status/update/delete/query` with top-level parameters and a task `agent_name`.
Choose 1–3 simple lowercase English words joined by underscores. `Remy` and `tmp` are reserved. Creation requires
`task_description` and `agent_type`, plus `cron` for a scheduled task. Send changed execution settings to `update`.
`query` takes a `question` and answers from a temporary read-only fork of a queryable task session.
See `skills/manage-proactive-agent/SKILL.md` for all six actions.

Creation writes `task_description` verbatim to `TASK.md`. Create, list and status return its workspace-relative `task_path`.
Read the latest file before editing requirements, keep useful background and verify the saved result. When changing an
agreed time, update both `cron` and the description. Changes apply to subsequent rounds. Deletion stops owned work
and retains history. Stop external collectors separately. Recreating a deleted task returns `restored=true` and reuses
its task description and history. Check the description against the new request.
`reme_agent_dir` selects the workspace-relative Agent directory and defaults to `reme_agents`.

Set Chat's initial personal memory with `components.agent_runtime.default.chat.initial_memory`. Runtime startup creates
Remy's `MEMORY.md` if it is absent. Personal memory survives restarts and session changes. Select Chat's `job_tools` in
its Step configuration.

The fixed Chat Agent uses `agent_name=Remy` and `agent_type=chat`. Agent names are stable identities;
API requests, responses, workspace directories and Desktop use the same name. Its files live under
`{reme_agent_dir}/Remy/`. Task names must be unique regardless of case; `Remy` and `tmp` are reserved.

Declare initial tasks under `components.agent_runtime.default.agents` in the proactive preset using the same creation
fields, including `task_description`. Startup creates missing identities and preserves existing descriptions, settings
and deletions. Two event tasks are enabled by default. `finance` analyzes collected news and quotes,
and the event-driven `work` tracks the user's current projects, progress, difficulties and discussion points from
incoming observations. Work observations capture screens and new Codex dialogue every five minutes. The separate
standalone screen-capture collector is disabled by default.
The default scheduled task `tech_news` also runs daily at 08:00 in the application timezone, summarizes important
technology news from the past 24 hours and sends a reminder when ready.

An individual Proactive Runtime consumes events, reconstructs context from workspace files and delivers saved reminders
through the Outbox to the Chat Inbox. The work example below illustrates possible issues and TODOs;
Claude Code trajectories can be submitted through the event API.

<p align="center">
  <img src="../figure/proactive-runtime.svg" alt="Work Proactive Runtime: Codex and Claude Code trajectories, screenshots, timestamped events, reminders and workspace file paths" width="100%">
</p>

Agent configuration uses `agent_type`: `chat`, `proactive` or `cron`. The management tool creates only
`proactive` and `cron` tasks; Chat is initialized by the runtime.

A `proactive` Agent accepts observations through `append_event(agent_name, events)`, with `event_ts`, `source`, `description`
and optional workspace-relative `paths`, `id` and `metadata`. A `cron` Agent requires a cron expression and executes
its full task at the scheduled time. Both use the application timezone and a serial executor per Agent.
`source=cron` is reserved. Event tasks may set `min_trigger_interval_seconds`, measured from the previous successful
completion. Inputs remain queued during this cooldown. The built-in finance task waits 10 seconds and claims at most
20 events per run.

With `config=proactive`, Chat, proactive, cron, and temporary forks share the Codex app server's native tool settings, including `web_search`.
Temporary `query` forks use the backend's read-only mode.
`notify_window={start,end}` limits automatic inbox wakeups; messages can still be processed and saved outside it.
`temporary=true` is independent of trigger type. Temporary Agents become eligible for collection after seven days
without new accepted inputs; cron triggers count as inputs, while duplicates and retries do not. Collection waits for
running work, pending events, undelivered messages and retained Inbox references. Daily records and resources remain.
Successful input receipts retain the greatest accepted reception time, preserving idle lifetime across priority processing
and restarts. On restart cron schedules the next future trigger; queued events still run, but missed offline triggers are not backfilled.
Chat idle and retry timers restart in memory. Daily handoffs use `daily/YYYY-MM-DD/<agent_name>.md`; round details use
`<agent_name>_HHMMSS.md`, with numeric suffixes for same-second collisions. The framework keeps each reminder body in Markdown and
updates delivery and read timestamps in its frontmatter under the Agent's `reminder/YYYY-MM-DD/HHMMSS.md`.
Round details and reminders share the same time and collision suffix. Existing reminder files keep their original paths.

`components.agent_inbox.default` uses the `event` backend (`AgentEventInbox`), inheriting the generic local
inbox implementation. `components.agent_runtime.default.inbox` selects that component. Each Agent stores pending
inputs in `inbox/inbox_unread.jsonl` and processed batches in `inbox/inbox_read.jsonl`; each batch contains `items`,
`consume_ts` and `metadata`. Claims reserve selected IDs without consuming them; successful execution commits a receipt.
All Agents receive `Event` records. Observations and cron triggers target task Agents; Chat events use the sender as
`source` and reference the reminder Markdown in `paths`, without copying its body. At most 1,000 processed inputs are
retained by default. Reminder Markdown remains the source for content and acknowledgement
timestamps; Chat reference views are rebuildable. Change reminder frontmatter to acknowledge messages externally.
One lifecycle-owned watcher and periodic rescans reconcile those source edits and interrupted handoffs.

Missing or corrupt individual reminders
retain their source files and event receipts, report their paths, and are isolated without blocking other reminders
or application startup. Restoring the original file automatically restores processing.

P0 forks Chat as soon as its session is available, after `components.agent_runtime.default.notifications.p0_delay_seconds`
(default 0). P1 waits for 10 continuous idle seconds, P2 for 1800, controlled by `notifications.p1_idle_seconds` and
`notifications.p2_idle_seconds`. Notification windows still apply. Only foreground Chat turns reset the idle clock.
The temporary fork inherits the current conversation, runs in the backend's read-only mode, and is disposed of after execution,
including failure or cancellation. Its hint and tool traces never enter the main conversation.

Codex, Claude Code and AgentScope share one session contract: a fork snapshots the latest accepted context without waiting for or interrupting its source. Running tools are not cloned. P0 reminder forks can run alongside the foreground Chat turn. Queued user input still defers reminder evaluation, and a new foreground instruction or session switch invalidates an older decision.

`append_session(delivery="immediate")` submits during generation; `delivery="idle"` waits for the turn to finish. Both return only after durable persistence. Inputs are consumed after confirmation. After timeout or cancellation, retry with the same message ID, role and content; a pending retry never dispatches twice. Processing times differ by backend:

| Backend | Snapshot and active append |
| --- | --- |
| Codex | Native fork and input queue. Native acceptance precedes persistence; the running turn may continue with the queued input. Ephemeral threads inherit model context but cannot return hydrated turns or durable append receipts. |
| Claude Code | CLI-native fork includes current queued input. Appends use the running invocation's client; a separate resumed CLI cannot update that client's context. Assistant input requires a hidden empty user flush; native batches are serialized so adjacent queued user frames cannot merge. |
| AgentScope | Copies live SDK state and seals unfinished work only in the branch. Queued observations are applied before model input or after reply completion, so SDK compression cannot overwrite them. |

See `reme/components/agent_wrapper/base_agent_wrapper.py` for the common contract. Claude Code and AgentScope forks remain file-backed; releasing runtime resources does not delete history. Their shared read-only hint restricts built-in tools; native command sandboxes and network settings remain SDK-specific.

Only the reminder fork receives `read_inbox()` and `send_chat_msg(content=...)`.
The framework selects the priority and batch; `read_inbox` accepts no arguments.
Reminder evaluation has a 120-second timeout and proactive execution a 1200-second timeout, configured with each
Step's `timeout_seconds`. Transient inbox failures retry with exponential backoff capped at 300 seconds; SDK type
or import errors stop retries until the environment is repaired and restarted. Chat status exposes inbox failures,
and cron logs unsuccessful partial collection results.

Reading is limited to the selected batch. Nonempty content is appended as an assistant message marked
`source="proactive"`; empty content commits a silent decision without inserting a message. Successful decisions
automatically consume the messages read, independently of user-read state. Failures remain retryable, and a switched
Chat session or a new foreground turn rejects the old fork's decision. The ordinary fork reply is never sent to the user.
Desktop derives its reminder list from these marked messages. `agent_inbox agent_name=<name>` pages any Agent's pending,
processing and processed events without claiming them; omitted `agent_name` selects Chat. `agent_outbox agent_name=<name>`
pages a task's committed reminders. `agent_reminder message_id=<id>` reads a canonical reminder; `mark_read=true`
acknowledges user reading without consuming its Inbox event.
Chat Inbox entries include a resolved `reminder` view from the canonical Markdown for display.

`agent_session` and `agent_events` expose full Chat history, but only the latest native turn for proactive/cron Agents,
including its input and all partial/final assistant messages. Task subscriptions replace the whole view with each
`snapshot`; `heartbeat` carries no messages. Each automatic event or cron run starts a fresh native session and rebuilds
context from the task description and daily summaries. Earlier sessions remain available as history. Main Chat continues its
current session; task questions fork the latest task session read-only without changing task execution history.

Event tasks accept `max_batch_events` (default 20); retries within the same process keep the original batch and leave new inputs for later rounds. `proactive_agent_step` defaults to a 1200-second total timeout; native automatic compaction inherits the global Codex App Server settings. Daily summaries carry current state; per-round details retain chronology and evidence.

`agent_reminder agent_name=Remy` also accepts a combined Chat message ID and returns `{id,items}`. `mark_read=true` acknowledges all retained source reminders, linked through their `chat_message_ids` frontmatter. Acknowledgement remains successful with an empty list after the sources have been removed. Only explicit user reading is synchronized; historical initialization, suppression and bubble presentation do not imply reading.

One background `agent_loop` schedules task execution and Chat reminder decisions with independent workers and retries.
Chat Inbox logs are rebuilt from reminder Markdown on startup; task input logs use the Event format.

`proactive_run`, `proactive_query` and `inbox_chat` are service-private Jobs. Collectors remain opt-in; configure `jobs.screen_capture_cron.agent_name` and enable the job to publish
complete resource packages as observations. No resource watcher or automatic TTL deletion participates in this flow.

The built-in proactive configuration selects CodexAgentWrapper. Claude Code and AgentScope implement the same session APIs and scheduling behavior.

## MCP

ReMe 0.4.2.2 uses FastMCP 4 and MCP Python SDK 2. Upgrade ReMe and host-side MCP bridges together;
standalone bridge environments require `fastmcp>=4.1.4,<5`. CLI tool catalogs use SDK 2 snake_case fields,
including `input_schema`, `output_schema`, and annotation hints.

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
