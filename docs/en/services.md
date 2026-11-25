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

The `proactive` preset uses `list_workspace`, `search_workspace`, `load_file`, `save_file` and `get_client_info` for workspace access.
Studio sends the complete source and uses the `mtime` from `load_file` as `expected_mtime`. A changed file causes a `file_conflict` response.
The default configuration provides `list`, `search`, `load`, `save` and `app_config`.

The `proactive` configuration exposes `ask_agent(query, agent_name=Remy)` for the fixed Chat Agent. It resumes the current
native session; `create_chat_session` selects a fresh session. The `demo` preset inherits both Jobs and adds
`list_chat_sessions`, `session_read` and `session_status` for history and native session inspection.
MCP does not expose Stream Jobs. Studio integration is separate from this backend.

## Proactive Runtime

The `proactive_manager` component discovers named Proactive Runtimes in the workspace, routes their events, and coordinates Remy and reminders. Each Runtime owns its inbox, schedule, serial execution, and retries. The `proactive_loop` background Job runs the Manager. The configured `AgentWrapper` provides model execution, tools, and native sessions.
Configure it through `components.proactive_manager.default`. The workspace-relative `reme_agent_dir` defaults to
`reme_agents`. Each task directory contains `config.json`, `TASK.md`, `sessions.jsonl`, `inbox/`, and reminder
state under `outbox/`. `Remy/` owns `MEMORY.md` and its own `sessions.jsonl`. The shared `daily_dir` stores task
results and reminder text.
Startup reconstructs runtime state from these files.
Session registries preserve complete records across an interrupted append; an incomplete tail is ignored on read
and removed by the next append. Corrupt complete records fail explicitly.
ReMe acknowledges the inputs after validating the record. The receipt retains the reminder path until the outbox has
registered it. A round interrupted before acknowledgement preserves the Markdown and can run again.

The Proactive Manager connects Remy, its temporary Reminder fork, and the named runtimes through workspace files.
The Life task is an illustrative integration.

<p align="center">
  <img src="../figure/chat-reminders.svg" alt="Proactive Runtime: Chat and its temporary Reminder fork, the Inbox, background tasks and shared workspace" width="100%">
</p>

Start the backend in one terminal, with `LLM_API_KEY`, `LLM_BASE_URL` and optional `REME_MODEL_NAME`:

```bash
reme start config=proactive
```

For Claude Code, set `CC_LLM_API_KEY` and the Anthropic-compatible `CC_LLM_BASE_URL`, then run
`reme start config=proactive_cc`. This preset shares the `.reme` workspace and the model selected by `REME_MODEL_NAME`
with `proactive`.
In both presets, Chat uses the wrapper default of `xhigh` effort and automatic task runs request `low`. Both use `workspace-write`. Claude Code
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
reme manage_proactive_agent action=create agent_name=experiment_watch \
  'task_description=Monitor experiment logs and report failures or completion.'
reme list_agent_inbox agent_name=experiment_watch
reme list_agent_inbox agent_name=Remy
reme list_task_reminders agent_name=experiment_watch
reme task_document action=read agent_name=experiment_watch
```

`manage_proactive_agent` supports `list/create/status/update/delete/query` with top-level parameters and a task `agent_name`.
Prefer simple lowercase English words joined by underscores. `Remy` is reserved. Creation requires
`task_description`. Set `cron` to create a time-triggered task, or omit it or pass `null` to create an event-triggered task.
The trigger type is fixed after creation. Send changed execution settings to `update`.
`query` takes a `question` and answers from a temporary read-only fork of a queryable task session.
See `reme/skills/manage-proactive-agent/SKILL.md` for all six actions. The `proactive` and `proactive_cc` presets
load their built-in skills from the installed ReMe package, including when launched outside a source checkout.

Creation writes `task_description` verbatim when `TASK.md` is absent. Create, status and update return an Agent view
with its workspace-relative `task_path`. List wraps those views in `{items,total}`.
Use `task_document action=read/save` to edit requirements. For a save, pass the `mtime` returned by read as
`expected_mtime`. For time-triggered tasks, update `cron` through `manage_proactive_agent action=update`. If the description states an agreed time,
update that text as well. Changes apply to subsequent rounds. Deletion stops owned work
and retains history. Stop external collectors separately. Restoring a deleted task keeps its trigger type, description, and history. Check the description against the new request.
`reme_agent_dir` selects the workspace-relative Agent directory and defaults to `reme_agents`.

Set Chat's initial personal memory with `components.proactive_manager.default.chat.initial_memory`. Manager startup creates
Remy's `MEMORY.md` if it is absent. Personal memory survives restarts and session changes. Select Chat's `job_tools` in
its Step configuration.

The fixed Chat Agent uses `agent_name=Remy` and `agent_type=chat`. Agent names are stable identities;
API requests, responses, workspace directories and Desktop use the same name. Its files live under
`{reme_agent_dir}/Remy/`. Task names must be unique regardless of case.

Declare initial Runtimes under `components.proactive_manager.default.runtimes` in the proactive preset using the same creation
fields, including `task_description`. Startup creates missing identities and preserves existing descriptions, settings
and deletions. An invalid Runtime definition is reported separately so other runtimes can continue. In the default preset,
`proactive_fin` follows spot gold, WTI crude oil, BTC/USDT, and the Nasdaq Composite with context from CLS news. `proactive_memory` tracks the user's
current projects, progress, difficulties and discussion points from
incoming observations. `codex_dialogue_cron` collects new Codex dialogue and image attachments;
`screen_capture_cron` separately captures all displays. Both run every five minutes and deliver to `proactive_memory`.
The dialogue collector also delivers to `proactive_note`, which tracks progress on existing notebook items.
The dialogue Job's `init_steps` establish the startup reading baseline without importing existing dialogue before
periodic collection begins.
The default time-triggered task `proactive_news` runs daily at 08:00 in the application timezone, summarizes important
technology news from the past 24 hours and sends a reminder when ready.

Each Runtime consumes events and reconstructs context from workspace files. The Manager delivers saved reminders
through the Outbox to the Chat Inbox. The work example below illustrates possible issues and TODOs;
Claude Code trajectories can be submitted through the event API.

<p align="center">
  <img src="../figure/proactive-runtime.svg" alt="Work Proactive Runtime: Codex and Claude Code trajectories, screenshots, timestamped events, reminders and workspace file paths" width="100%">
</p>

Use `append_task_events(task_names, events)` to send a batch to one or more event-triggered Proactive Runtimes.
Both arguments are nonempty arrays. Each event has `event_ts`, `source`, and `description`, with optional workspace-relative
`paths`, `id`, and `metadata`. Save referenced files first. The response reports accepted event IDs by task in
`answer.deliveries`. `answer.rejected` reports missing, deleted, or time-triggered targets and invalid batches, such as conflicting event IDs.
The response succeeds when all targets receive the events. Retry targets in `answer.errors`, such as a full inbox,
using the original event IDs and payload.

```bash
reme append_task_events task_names='["experiment_watch"]' \
  'events=[{"id":"experiment-log-1","event_ts":"2026-11-22T10:30:00+08:00","source":"experiment_log","description":"A new milestone was recorded."}]'
```

At each time specified by `cron` in the application timezone, the runtime queues an event with `source=cron`.
Agent views report `agent_type=time` for time-triggered tasks and `agent_type=event` for event-triggered tasks. The Notebook preset is event-triggered.
Each Runtime has its own inbox capacity and executes serially. An event-triggered task takes up to `max_batch_events`
observations per round (20 by default). A time-triggered task takes one scheduled trigger per round. In-process retries
keep the original batch. For event-triggered tasks, `min_trigger_interval_seconds` starts at the previous successful completion.

With `config=proactive`, Chat, background tasks, and temporary forks share the Codex app server's native tool settings, including `web_search`.
Temporary `query` forks use the backend's read-only mode.
`notify_window={start,end}` limits automatic Chat evaluation and sending. Committed reminders enter the inbox immediately and can be browsed outside the window.
Tasks remain in their stable directories until explicitly deleted; deletion retains source files and history. On restart cron schedules the next future trigger; queued events still run, but missed offline triggers are not backfilled.
Chat idle and retry timers restart in memory. Tasks save progress and their reminder decision together in
`daily/YYYY-MM-DD/{agent_name}_HHMMSS.md`. Same-second collisions add a numeric suffix. The body holds new information,
and `reminder_priority` and `reminder_message` describe the delivery decision in the frontmatter. The workspace-relative
record path is the reminder ID. The sender's `outbox/` mirrors the source path with a JSON file for delivery and
processing state.

`components.agent_inbox.default` uses the `event` backend (`AgentEventInbox`), inheriting the generic local
inbox implementation. `components.proactive_manager.default.inbox` selects that component. Each task stores pending
inputs in `inbox/inbox_unread.jsonl` and processed batches in `inbox/inbox_read.jsonl`; each batch contains `items`,
`consume_ts` and `metadata`. Claims reserve selected IDs without consuming them; successful execution commits a receipt.
The tasks receive `Event` records for observations and cron triggers. The processed history limit defaults to
1,000 events. Remy's inbox view reads the same reminder registrations as the task outboxes. The
runtime selects one eligible batch at a time by priority and delivery time, subject to the configured inbox item
and byte limits. A file watcher and periodic scans reconcile edits and interrupted handoffs.

Missing or corrupt reminders are reported and isolated without blocking other reminders. Restoring the original
file restores processing.

P0 forks Chat as soon as its session is available, after `components.proactive_manager.default.notifications.p0_delay_seconds`
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

The framework selects the priority and puts that batch of inbox entries in the reminder fork's prompt. The fork
reads relevant source records without following their links, then calls `submit_chat_reminder(content=...)`.
Reminder evaluation has a 120-second timeout and proactive execution a 1200-second timeout, configured with each
Step's `timeout_seconds`. Transient inbox failures retry with exponential backoff capped at 300 seconds; SDK type
or import errors stop retries until the environment is repaired and restarted. Chat status exposes inbox failures,
and cron logs unsuccessful partial collection results.

Reading is limited to the selected batch. Nonempty content is appended as an assistant message marked
`source="proactive"`; empty content commits a silent decision without inserting a message. Successful decisions
automatically consume the messages read by the fork. Failures remain retryable, and a switched
Chat session or a new foreground turn rejects the old fork's decision. The ordinary fork reply is never sent to the user.
Desktop derives its reminder list from these marked messages. `list_agent_inbox agent_name=<name>` pages a task's input
events. Set `agent_name=Remy` or omit it to view Chat's incoming reminders. The query shows pending, processing, and
processed entries without claiming them. `list_task_reminders agent_name=<name>` pages a task's delivered reminders.
Both reminder lists return reminder text and its current outbox state. `read_agent_reminder message_id=<relative-path>`
reads the source reminder.

`read_agent_session` and `watch_agent_session` expose the selected Chat session's history and the latest native turn of
each task. Subscriptions use `snapshot` for whole-view replacement and `update` for complete-message upserts. A
`heartbeat` carries no application state. Each automatic event or cron run starts a fresh native session and rebuilds
context from `TASK.md`, the current inputs and searches of past records. Earlier sessions remain available as history. Main Chat continues its
current session; task questions fork the latest task session read-only without changing task execution history.

`proactive_agent_step` defaults to a 1200-second total timeout. Native automatic compaction inherits the global Codex App Server settings.

`read_agent_reminder agent_name=Remy` also accepts a combined Chat message ID and returns `{id,items}`. The outbox
links the source reminders to that message through `chat_message_ids`.

One background `proactive_loop` schedules task execution and Chat reminder decisions with independent workers and retries.

`run_proactive_agent` and `process_chat_reminders` are service-private Jobs. The `proactive` preset includes the
Desktop runtime, initial tasks and independent finance, Codex dialogue and screenshot collectors. Collectors publish
observations explicitly and retain the source files. `demo` extends this preset with standalone examples.

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

Install `reme-ai[core]` and run `reme start config=proactive` to serve Studio:

```text
http://127.0.0.1:2333/
```

Disable it with `service.web_enabled=false` or select a build with `service.web_static_dir`. Missing static assets do not prevent the Job API from starting.

### Backend settings

The HTTP `reme start` process supervises the Python service. Studio remains available on the same address during a service restart. Open **Settings → Backend** to edit the workspace or, with the Codex preset, the LLM URL, API key, model, and reasoning effort. Select **Save and restart** to apply the changes. The model connection test runs in a temporary workspace. When a change fails to start, the supervisor tries the previous configuration.

Instances with default DashScope speech components or enabled DashScope search steps also accept a DashScope API key. The separate credential takes effect after saving and restarting and is shared by the configured speech recognition, speech synthesis, and search. Settings only displays whether a key is configured.

Backend settings are grouped into Workspace, Harness and model, and Speech. Studio opens setup when core settings are missing. Speech can be enabled later. Its API base URL uses the native DashScope protocol and defaults to `https://dashscope.aliyuncs.com/api/v1`. Enter model and voice IDs supported by that service. Select **Reset speech defaults** to restore the default speech parameters.

User settings are imported once from startup configuration, system environment variables, and the nearest `.env` in the startup directory or up to five parent directories. System environment variables take precedence over `.env` when importing settings. Subsequent starts read the instance's `settings.json`. Change model and speech parameters through Settings. API keys are encrypted with Fernet, using a master key in the server host's system credential store. Servers and containers may supply a separate `REME_ENCRYPTION_KEY`, generated with `Fernet.generate_key()` and retained across restarts.

The supervisor stores settings, credentials, and restart status under `~/.config/reme/instances/`. The startup directory, configuration, listen address, and port identify the default instance. Set `REME_DAEMON_DIR` to choose a private directory outside the workspace. Docker Compose keeps `/state` separate from `/data`. Management is available on a local loopback listener. To enable remote management, configure a random `REME_ADMIN_TOKEN` and connect over HTTPS. Enter the token in Studio when prompted. A separately hosted Web frontend must proxy `/admin/` through its own origin. Management has its own endpoints and is independent of Jobs and MCP.

## Notebook projects

Start the service with `reme start config=proactive`, then run these commands in another terminal.

```bash
reme todo action=list_projects
reme todo action=list project=ReMe scope=all
reme idea action=list project=ReMe scope=all
```

`todo action=list_projects` lists projects from both Todo and Idea records. The result includes `total` and `items`,
with `project`, `todo_count`, and `idea_count` in each item. The default `scope=all` includes active and archived
records. Set `scope=active` or `scope=archive` to count records in that scope. Use the returned project name to list
its Todo or Idea records.

New Todos default to `pending`. Use `in_progress` once work starts and `closed` when it finishes or is cancelled.
A `closed` Todo is archived. Pass `close_note` with `status=closed` to record the outcome or reason. Reopening clears
the closure time and note.

Active Todos are ordered by target date by default. Use `reme todo action=list order=created` for newest creation
time first, with older records that lack a creation time at the end. Studio offers both orders in the Todos view.

The event-triggered `proactive_note` runtime receives the same new Codex dialogue as `proactive_memory`. It loads `manage-todos-ideas`
to find related Todos and Ideas and read the full progress history before updating a Todo. It adds new progress, corrects
or removes individual notes when the evidence changes, and can move a Todo to `in_progress`. Remy handles creation,
closure, Idea updates, link changes, and other follow-up
through the standard reminder inbox. Each follow-up is one plain-language Markdown bullet with the record, evidence, and next step.

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
