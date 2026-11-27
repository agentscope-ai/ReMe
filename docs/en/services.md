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

The proactive preset starts one Remy Chat Runtime and four background Runtimes for project memory, Notebook follow-up,
finance and technology news. Configure them under `components.agent_runtime`; `components.agent_manager.default` manages
background identities only. Runtimes own sessions, execution locks, Inbox and Outbox resources. Jobs own execution and
scheduling; each call creates fresh Steps. The Application owns the shared AgentWrapper and App Server.

For Claude Code, start `reme start config=proactive_cc`. This preset uses Claude Code's native system prompt and appends
each Runtime's role instructions, including Remy's Chat prompt. Prompt snapshots are disabled so resumed calls use the
current appended instructions.

```bash
# Terminal one
reme start config=proactive
# Terminal two
reme manage_proactive_runtime action=list
reme manage_proactive_runtime action=create agent_type=event agent_name=experiment_watch \
  task_description='Follow experiment logs. Report errors and completion; record normal progress.'
```

`manage_proactive_runtime` supports `list/create/status/update/delete/query`. All background tasks use the private
`run_proactive_agent` Job. Configure preset `initial_task`, `skills`, `job_tools` and session options under
`components.agent_runtime`. A Runtime creates TASK.md only when missing and reads the current document on each run.
Dynamic creation accepts `task_description`, `skills` and `job_tools`; AgentManager applies `runtime_defaults` and checks
`allowed_job_tools`. Trusted injected tool constraints are configured by the operator. Existing task files are preserved.
Trigger type and session bindings are fixed after creation and retained on restoration. Dynamic settings remain in the
workspace's per-task config.json. Older domain Job names and definitions are unsupported; update task configurations to
the new binding schema while retaining TASK.md and history.
Management responses expose settings under config and execution through running; list returns items/total.
Client Agent views remain available through list_agents and the session endpoints.

Creation requires `agent_type=event` or `agent_type=time`. Time-triggered tasks require a `schedule` Cron expression
in the application timezone; event-triggered tasks have no schedule.
Trigger type, skills and tool bindings are immutable after creation. Update accepts enabled, schedule,
min_trigger_interval_seconds and notify_window. Pause allows the current round to finish. Delete cancels execution and
preserves files; restoring the same name retains its bindings and history. Use task_document action=read/save to edit TASK.md,
and pass the read response's mtime as expected_mtime when saving.

components.agent_runtime.Remy.initial_memory creates `{reme_agent_dir}/Remy/MEMORY.md` only when absent.
Remy's notification_config sets the P0 delay (0 seconds), P1 idle threshold (10 seconds) and P2 idle threshold (1800 seconds).
P0 can reach an active native foreground turn; P1/P2 wait for idle. The sender's enabled and notify_window settings also apply.

### Collection and Connector controls

Connectors are workflows. Each collection Job publishes a complete resource package and metadata.json before delivering
standard Events. Eight configured Jobs collect screens, Codex and Claude Code dialogue, finance news, gold, oil, Nasdaq
and BTC every five minutes. Dialogue init_steps establish a baseline without importing old history. Source directories
retain opaque cursors, attachments and pending delivery batches. BaseHarnessConnectorStep owns scanning, publication, handoff and cancellation;
Codex and Claude Code supply their native format adapters and share the same resource and response contracts.

Use the `manage_connector` MCP Job to manage preset collection Jobs:

| Action | Parameters |
| --- | --- |
| List presets, download switches and recipients | `{"action":"list"}` |
| Pause or resume downloads | `{"action":"pause","connector_job":"finance_news_cron"}` or `{"action":"enable","connector_job":"finance_news_cron"}` |
| Replace recipients | `{"action":"update","connector_job":"finance_news_cron","runtime_names":["proactive_fin","market_watch"]}` |

Only existing event-triggered tasks can be selected as recipients. An empty target list keeps downloads without new
handoffs. Changes persist at `{metadata_dir}/proactive/connectors/<connector_job>.json` and apply without restart to new
batches; frozen batches retain their original recipients. Pause retains packages and pending handoffs. Download controls
are independent of Runtime `enabled`. `list` reports `available` separately from the saved download switch; only configured,
started Jobs in `connector_jobs` can be changed. Collection frequency remains in service configuration.

### CloudEvents SSE subscriptions

With `config=proactive`, `manage_connector` can create an HTTP(S) GET subscription whose endpoint returns
`text/event-stream`, with one [CloudEvents 1.0](https://github.com/cloudevents/spec/blob/v1.0.2/cloudevents/spec.md)
JSON envelope per SSE `data` field. First create the receiving event-triggered task, then call the MCP Job or
`POST /manage_connector` with:

```json
{
  "action": "create",
  "name": "Experiment progress",
  "trigger_type": "event",
  "runtime_names": ["experiment_watch"],
  "binding": {"type": "sse", "url": "https://example.com/events", "format": "cloudevents"}
}
```

Creation returns `source_id` and starts the connection in the background; it does not confirm delivery. Use that ID
instead of `connector_job` with `pause`, `enable` or `update`; update replaces `runtime_names`, and an empty list keeps
collecting without new deliveries. `list` includes both presets and subscriptions. Subscription items report
`connected`, `last_error`, `last_event_id` and `pending_delivery_count`; `available` describes the manager's lifecycle.
Name, URL and format are fixed for a subscription. See the [request schema](https://github.com/agentscope-ai/ReMe/blob/main/reme/schema/proactive_agent.py).
Recipient updates keep the active connection; enabling an already connected subscription is idempotent.
Manual definition edits take effect after restart; control updates refuse to overwrite edits not yet loaded by the service.

Definitions and delivery checkpoints persist under `{metadata_dir}/proactive/sources/<source_id>/`; original CloudEvents
and their reception times live under `{resource_dir}/cloudevents/`. Internal Events reference those files. Identity uses
CloudEvents `source + id`; changed content under the same identity fails explicitly. Equivalent timestamp offsets and
omitted/null optional attributes are normalized for comparison, while original JSON is retained. Omitted event times use the first
saved reception time. Pending deliveries retain their original recipients, retry only unaccepted targets, and block
cursor advancement on rejection or error. Restore rejected receiving tasks before retrying. Pause closes the connection
and preserves files and pending work. Restart restores enabled subscriptions and reconnects with the committed
`Last-Event-ID`; historical replay depends on upstream support. HTTP 204 stops automatic reconnection until enable or
restart. SSE lines and buffered event data are bounded to 1 MiB. This combines SSE with CloudEvents JSON; provider-specific subscription
handshakes and custom authentication headers are not configured by this interface.

For a new task, start with `manage-proactive-runtime`: scheduled work such as a daily briefing can fetch its own sources;
event-triggered work needs an ongoing feed. Reuse a preset collector, or follow `manage-connector` to place a custom script
beside TASK.md under `scripts/`, save source material to `resource_dir`, and send Events through HTTP
`POST /manage_connector` with `action="send"`, `runtime_names` and `events`. Event paths are workspace-relative;
`source="time"` is framework-reserved. Inspect `deliveries`, `rejected` and `errors` even after partial failure.
Retry only `errors`, with unchanged event IDs and content, and retain the source cursor until every recipient is terminal.

### Reminder commits and recovery

Task EventInbox → NotificationOutbox → Remy NotificationInbox → decision fork → ChatOutbox → native user conversation.
A round acknowledges inputs only after successful inference and validated output. Daily frontmatter contains reminder_priority
(P0/P1/P2/skip) and reminder_message. Model failure or timeout preserves inputs and drafts. Invalid frontmatter gets one repair
in the same session. Once an input receipt is durable, reminder handoff can recover after restart without another model call.

The notification queue persists source references; reminder priority and content come from daily Markdown.
Receiver InboxReceipt owns send/skip processing facts, sender Outbox owns delivery facts, and ChatOutbox links native messages.
read_at means Remy processed a reminder, not that the user read it. A read-only fork calls submit_chat_reply with action=send
and content, or action=skip without content. Commit rechecks source versions, foreground activity, native session revisions
and sender eligibility. Recovery probes a fixed native message ID before acknowledging a lost-response delivery.

agent_manager_loop schedules tasks; chat_notifications_loop schedules reminders. Both are private BackgroundJobs.
run_chat_agent, run_proactive_agent, process_chat_notifications and submit_chat_reply are private execution endpoints.
The public ask_agent stream owns DONE. Task questions use temporary read-only forks of the latest task session without
consuming inputs or replacing its primary session.

Runtime components load built-in skill templates from `components/proactive/runtime/skills/`, localized through
PromptHandler, or reference project skills by name. Chat and Proactive Runtimes configure their own skills and tools. Generated
SKILL.md files are scoped to one invocation and cleaned up afterward. Wrapper skills use SkillSpec lists: name references
an existing skill; name/description/content defines generated content. Native Wrapper protocols remain unchanged.
An OS lock protects each generated skill name; a later invocation recovers abandoned links after a process exits.
Execution Jobs are fixed by the workflow. Runtime definitions no longer contain `run_job`, and notification loops no
longer accept `process_job`; remove these fields when migrating an earlier configuration.
The shared prompt loader is `reme/utils/prompt.py`; Runtime initialization does not inspect execution Steps.
Per-run query and input templates belong to `reme/steps/proactive/agent.yaml`. Customize them through
`jobs.run_proactive_agent.steps[0].prompt_dict`, using `query`, `record_instruction`, `event_input` and `time_input`
or their `_zh` variants. Runtime `prompt_dict` configures role and initialization text.
Notebook automatic-write constraints are injected through trusted Runtime configuration and checked by manage_notebook.

See [components](../../reme/components/proactive/), [workflows](../../reme/steps/proactive/) and
[configuration](../../reme/config/proactive.yaml).

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

Studio and all management clients use `X-ReMe-Admin-Token: <token>`. `Authorization: Bearer ...` is no longer accepted; upgrade clients together with the backend.

When a proxy terminates HTTPS, preserve the public `Host` and set `X-Forwarded-Proto: https`. Enable `service.proxy_headers=true` and restrict `service.forwarded_allow_ips` to trusted proxy IPs or CIDRs. Use `*` only when the listener is reachable exclusively through trusted proxies.

[ModelScope Docker Studios](https://github.com/modelscope/modelscope-skills/blob/main/skills/ms-studio-deploy/SKILL.md#docker-studio-reference) reserve `Authorization` and require port `7860`. Use `REME_PORT=7860`, `REME_CONFIG=proactive`, and the proxy settings above. Store `REME_ADMIN_TOKEN` and a persistent `REME_ENCRYPTION_KEY` in platform Secrets.

## Notebook projects

Start the service with `reme start config=proactive`, then run these commands in another terminal.

```bash
reme manage_notebook action=list_projects
reme manage_notebook kind=todo action=list project=ReMe scope=all
reme manage_notebook kind=idea action=list project=ReMe scope=all
```

`manage_notebook action=list_projects` lists projects from both Todo and Idea records. The result includes `total` and `items`,
with `project`, `todo_count`, and `idea_count` in each item. The default `scope=all` includes active and archived
records. Set `scope=active` or `scope=archive` to count records in that scope. Use the returned project name to list
its Todo or Idea records.

New Todos default to `pending`. Use `in_progress` once work starts and `closed` when it finishes or is cancelled.
A `closed` Todo is archived. Pass `close_note` with `status=closed` to record the outcome or reason. Reopening clears
the closure time and note.

Active Todos are ordered by target date by default. Use `reme manage_notebook kind=todo action=list order=created` for newest creation
time first, with older records that lack a creation time at the end. Studio offers both orders in the Todos view.

The event-triggered `proactive_note` runtime receives the same new Codex and Claude Code dialogue as `proactive_memory`. It loads `manage-notebook`
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
