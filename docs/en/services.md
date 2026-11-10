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

The PA component restores the selected chat from the last entry of `{metadata_dir}/pa/session_ids.json`.
Selecting a chat atomically moves its record to the end before publishing the selection; read cursors are retained.
Startup migrates an existing `{session_dir}/pa.json` selection into this list, then removes the old file after saving.
Native conversation history remains owned by Codex and is not deleted. Invalid metadata or unavailable history is reported
without silently selecting a replacement. Event cursors and activity are recreated after restart, and clients recover
through a fresh snapshot.

The `proactive` config sets `service.timeout_graceful_shutdown: 5`. During HTTP shutdown, requests still active after
five seconds are cancelled before application components close, so an open subscription cannot hold shutdown
indefinitely. Custom HTTP configurations with long-lived subscriptions should also set a finite shutdown timeout.

All three agents use `{workspace_dir}/{daily_dir}/personal/MEMORY.md` (by default, `daily/personal/MEMORY.md` within the
workspace). PA prompts contain this path and the configured workspace. The chat prompt instructs Remy to read memory
before responding and to read the complete latest file before merging updates. Memory contents are not injected into the
session prompt. Before a new session or proactive round, shared code initializes a missing memory file using
`default_memory` from `reme/steps/proactive/agents/base.yaml`, with per-path locking and atomic writes. The defaults cover financial
news and AI advances, and require valuable proactive
reminders without unimportant or repetitive interruptions. Existing files, including empty
ones, are preserved; there is no onboarding or empty-memory gate.

The proactive package groups model execution and adjacent prompts under `agents/`, agent tools under `tools/`,
external collection under `data_connectors/`, scheduling and delivery under `background/`, and session/status APIs under
`api/`. Request checks live in `base.py`; application-owned state, files and sessions live in `reme/components/pa/`.
Generic JSON validation lives in `reme/steps/file_io/`.

`agents/main.py` supplies the chat (`pa_chat_step`) and new-session Steps; `api/sessions.py` supplies session APIs; `agents/main.yaml` supplies Remy's persona
and memory maintenance/forgetting instructions through `PromptHandler`, with English defaults and Chinese variants.
The public jobs remain `pa_chat`, `pa_session`, and `pa_new_session`.
The proactive config defaults to `language=zh`; use `language=en` for English
or a Step's `language` to override the application language. Customize localized keys through the Step `prompt_dict`
(for example, `system_prompt_zh` and `default_memory_zh`). Default-memory overrides only affect missing files.
Explicit forgetting requests instruct the agent to remove matching content without copying it elsewhere or restoring
it from old conversations or summaries. Maintenance removes confirmed obsolete state while retaining valid long-term
facts. Creating a session initializes missing memory but does not generate a reply. Each chat builds its prompt and
tool bindings from current configuration; these parameters are not stored in the selection file or an in-memory
session cache. The memory path continues to refer to the latest file. Invalid paths or initialization failures
leave the current session selection unchanged.

The roles use concise bilingual prompts with a few generic examples; model choice and reasoning follow
`reme/config/proactive.yaml`. The Codex client uses a 65536-token context budget and a 56000-token automatic compaction threshold;
these are client settings, not the provider's maximum context claim. Native long-term memory remains enabled.
`CODEX_MODEL_NAME` can override the model; review the model catalog capabilities when changing it. File changes load
on the next service start and apply to subsequent chats in the restored session.

Each proactive round forks the selected Main conversation with the proactive system prompt and Job tools already
configured, waits for native compaction to finish, then sends at most one work query in that same branch. Main is unchanged.
An empty Main starts a fresh worker without fork or compaction; unreadable source sessions still fail explicitly.
The query reads current memory, completed conversations, inbox feedback, task results and message delivery history;
handles explicit configuration requests; selects and verifies useful findings; and submits one message decision. See `reme/steps/proactive/agents/proactive.py` and its bilingual YAML prompts.
Research is optional, with a prompt budget of three leads and five search API calls. There are no per-stage queries or
intermediate model reports. Latest explicit feedback takes precedence over inherited background and recovery inputs.
Paused, inactive, evening-closed or invalid configurations never start a proactive model call, including compaction.
Main queues topics, feedback and settings requests through `write_proactive_inbox`; they wait for the next permitted
wakeup and never trigger an extra model call. A disabled agent or invalid configuration requires external re-enabling
or repair before queued requests can be processed.

Reminder wording preserves core findings, necessary background, relevant times and verified source URLs, remaining concise
but complete. Value, novelty, timing, frequency and Master's responses guide the decision; repeated nonresponse calls for
fewer interruptions. Empty content is an explicit silence decision. Code requires a live server-bound round and an active
sending query, permits one decision per round, and checks the latest message-history snapshot before nonempty submissions.
Accepted reminders retain independent background delivery, supersession, daily greeting limits and evening closure.

Consumed inputs are checkpointed before cursor advancement and replayed after failures or restart. A successful work query
clears handled recovery records; an interrupted query retains them. Preparation, compaction and work share the active-window
deadline. Temporary tasks fork Main only when due, inheriting its context, model and tools; cron tasks retain their own sessions.
The runtime coordinates shared file operations, session consumption and event replay through their dedicated owners.

PA chat, selection and event Steps require the configured `PaRuntime`; there is no metadata-state fallback.
The runtime owns session lifecycle and one typed active round. Internal task and reminder owners keep their own locks;
the delivery worker can continue after an agent round ends. Reminder submission requires a live server-bound round.
Inbox tools share `proactive_inbox.py`; message tools share `chat_messages.py`. Task tools are configured only in YAML, with
screen capture and finance as defaults. Session selection is restored only from `session_ids.json`; the old `pa.json`
startup migration has been removed.

Before Jobs start, the PA component creates missing files under `{metadata_dir}/pa/`:

| File | Initial content |
| --- | --- |
| `session_ids.json` | `[]`; the last entry selects the current chat |
| `proactive.json` | `components.pa.default.default_proactive` |
| `cron.json` | `components.pa.default.default_tasks` |
| `proactive_inbox.md` | Empty |
| `proactive_inbox_receipts.jsonl` | Empty request receipts |
| `temp_tasks.json` | `[]` |
| `handoff.md` | Empty handoff for the next proactive round |
| `chat_inbox.jsonl` | Empty |
| `cron_runs.jsonl` | Empty |

Inbox names identify their recipient: `proactive_inbox.md` carries Main's instructions to Proactive, with
`proactive_inbox_cursor.json` consumption state and `proactive_inbox_receipts.jsonl` request receipts.
`chat_inbox.jsonl` carries background messages addressed to Main and their delivery states.
Cron documents live under `cron/{task_id}/`; one-shot documents live under `temporary/{id}/`, both inside this PA directory.

Defaults are configured in `reme/config/proactive.yaml`. The proactive defaults are `start_time: "0700"`,
`end_time: "2330"`, and `interval_minutes: 5`. Existing files, including empty text files and empty task lists, are kept;
invalid existing JSON fails explicitly instead of being replaced with defaults. Each agent initializes missing personal
memory before invoking its model, using the localized defaults in `agents/base.yaml`.

Wakeups use the application timezone. Start is inclusive and end exclusive; overnight windows are supported, and equal
endpoints mean all day. Within active hours, the next delay starts after the previous round finishes. Outside the window,
the loop waits for the next opening. Sleep checks JSON edits at most every 30 seconds, so changing the file requires no
restart. Invalid configuration pauses model work while reporting the error. The former YAML wakeup delay, in-memory
override and `set_next_wakeup` tool are removed. Resource cleanup remains an independent CronJob.
Each round checks active hours before preparation and again before its work query. Preparation and in-flight work are
interrupted at the window deadline, retaining recovery records. Evening closure prevents subsequent ordinary rounds.
Polling and policy checks spend no model tokens.

Optional conversational hints default to enabled. Set `hints_enabled: false` in `proactive.json` to disable sampling
and injection. `hint_probabilities` separately configures `morning`, `midday` and `evening`, defaulting to 1.0, 0.5 and 0.5;
each accepts numbers from 0 to 1. YAML `default_proactive` seeds new files; missing fields or periods use schema defaults.
Within active hours, sampling occurs at most once
per local day in each window: 06:00–12:00, 12:00–14:00, and the last 30 minutes before an evening closing time (no earlier
than 18:00). Overnight and all-day activity use an evening window after 21:00, never an evening hint in the morning.
Misses also consume the opportunity. A current chat is required. Results are saved in lazily created `hint_state.json`
to prevent rerolls after restart and avoid the last selected variant of each kind. Bilingual directions come from
`agents/proactive.yaml` and are injected into the single work query after compaction. The agent may ignore them; ordinary useful
updates remain independent of sampling. Actual delivery is recorded in `chat_inbox.jsonl`. Sampled directions are not
personal memories or instructions to repeat in future rounds.
The 18 directions concern everyday life, care and playful surprises; only one evening direction summarizes confirmed todos.
Special reminder kinds are `morning`, `midday` and `evening`: each requires this round's matching sampled hint and
consumes one slot per local date, including failed or discarded submissions; a failed closing decision never reopens that window after an uncertain delivery. Normal information uses `normal`, without
explicit greeting wording. Every nonempty evening decision is the final proactive message for that window, even without
literal good-night wording. Pending closing messages cannot be replaced by ordinary reminders. Sent closing messages
stop proactive queries until the next opening; misses and empty decisions do not close the day. The reminder journal
owns this state across restarts and session switches, without a separate quiet-state file.

The conversation tool consumes completed unread rounds across the latest 100 registered chats during the work query.
Each user input stays with its replies; tools and injected messages are excluded. Defaults remain 20 rounds,
2000 characters per message and 20000 total. Read or checkpoint errors preserve cursors.

`write_proactive_inbox(message)` queues timestamped Markdown for Proactive's next wakeup. Put topics, reminder feedback
and settings requests in the message; Proactive interprets them and edits its own configuration during its work query.
Queued or received means the message was saved or read, not that its requested work or setting change is complete.
While notifications are paused, outside active hours or blocked by invalid settings, explicit inbox requests still get
a settings-only round. It can repair or resume settings without researching or sending reminders; failed requests remain retryable.
`read_proactive_inbox()` checkpoints appended or edited content before advancing a content-hash
cursor, retaining the source file. It returns the complete unread Markdown text directly, or an empty string when unchanged;
queued messages appear as `timestamp content`, without request IDs or source labels; handwritten Markdown is preserved.
No follow-up file read is needed. This replaces the previous path-based response.
Main chat and Proactive work share the personal-memory lock.

`manage_temp_proactive_agent` lists, creates, updates, pauses, resumes or cancels one-shot tasks. Creation requires a clear
`query`; `run_at` and `delay_minutes` are mutually exclusive, and fractional delays are supported. Existing tasks use their
stable `task_id`; revisions invalidate stale workers. The `temp_proactive` background Job owns waiting and cancellation. When due, it invokes the service-private
`temp_agent` Job, which forks the source Main conversation and executes one query. Its complete final result is saved to
`{metadata_dir}/pa/temporary/{id}/result.md` before immediate delivery, with no separate sending query or character cap.
Restart recovery delivers saved results without repeating model work. An interrupted query without a saved result fails
explicitly; resume creates a new revision for a deliberate retry. Overdue tasks execute once and label delayed results.
Submission, native history insertion and Desktop display/read receipts remain distinct.
Use native file tools to read cron definitions in `cron.json`, execution records in `cron_runs.jsonl` under `{metadata_dir}/pa/`,
and result files under `{metadata_dir}/pa/cron/`. Recorded states do not prove a worker is
currently running. Read result bodies without changing consumption or user-read markers; no separate task-status tool remains.

Proactive uses `get_chat_messages(limit=10, offset=0)` to read each reminder's latest state, newest creation
first, without consuming history. Results include complete message content and delivery states; use `limit` and `offset` to paginate.
Generating a background result or consuming it as an agent does not mean Master has read it or a reminder was delivered;
neither state alone establishes that the content was shared.
JSONL appends share an ID for successive `pending`, `sent`, `discarded` or `failed`
states, with creation/update/expiry/sent times and the target session. This replaces the former `is_draft` contract.
Main handles reminder follow-ups from the current conversation and explicit user context, without querying this history.

`send_chat_message(content, kind="normal")` uses a backend-bound sender policy; the model cannot override delivery.
For ordinary Proactive it accepts one decision per round, without a presentation-size character cap. `kind` is one of
`normal`, `morning`, `midday`, `evening`; empty content uses `normal`. Empty content withdraws pending content
for the bound chat and returns `skipped`. A newly accepted nonempty reminder supersedes the old pending reminder and returns
`pending`, including when Main is already idle. The native `reminder_delivery` background Job waits for ten continuous
idle seconds after submission, resetting when chat
starts again. Delivery rechecks the current session, native turn, active hours and expiry. The ordinary pending lifetime ends after one
wake interval or at the current window deadline, whichever comes first. Missing or changed sessions, closed active hours and expiry discard the reminder.
Restart rebuilds pending state from JSONL and repeats the checks and idle delay. Retries use the same native message ID.
Sent means inserted into chat, not read; insertion and JSONL status publication are separate operations.
`pa_events` snapshots and status events carry a derived `notifications` window with absolute start/end times and the
closing message ID and enabled state. Urgent cron and temporary messages carry `delivery: immediate` and bypass quiet mode, proactive pauses,
active windows and evening closure. Temporary tasks fulfill explicit requests even when their content is not urgent.
Explicit voice switches and volume still apply; temporary quiet mode does not block immediate narration.
Desktop bubbles and proactive system notifications recheck it before display, allowing only the
closing message after that decision and dropping stale queues. Display stops at the deadline, without replaying old
messages the next day. Chat history remains intact. Stop the service before explicitly migrating an existing journal;
runtime schemas and tools require the new contract:

```bash
python scripts/migrate_pa_reminder_kinds.py /path/to/workspace/reme_metadata/pa/chat_inbox.jsonl
```

The script validates every row, retains an original `.bak`, identifies saved hint rounds and marks other old records
as `normal` before atomic replacement.

The proactive agent has inbox-reading, task-result, screen-capture, message-history, message-submission
and JSON-validation Job tools. Session status is a common session operation, not a proactive tool. Relevance and semantic deduplication remain
agent decisions. Main maintains long-term memory; Proactive's memory query only reads and understands it.
Recent concerns and unfinished work belong in the handoff.
Forgotten facts must not be restored through old conversations or summaries.

Desktop's Remy sidebar switches between reminders and a scheduled-task result inbox. Read/unread badges appear on each
card; the result inbox supports task filtering and pagination. `proactive_inbox_view(action="list", ...)` is read-only,
with `kind` set to `all`, `task_result` or `instruction`, `filter` set to `all`, `pending` or `unread`, and pagination.
The `unread` and `task_id` filters require `task_result`; `unread` means not yet viewed by the user.
`action="read", entry_id=...` returns current source text and receipt
without advancing consumption. Partial reads show a warning and suppress a definitive pending count.
`task_results(action="read", path=..., mark_read=true)` independently records user reading while retaining Proactive's
consumption state. Received does not mean work completed; edited instructions are processed again.
Result lists skip invalid documents and report `complete=false` with `warnings`; originals remain untouched.
History selection and Proactive consumption also skip invalid results; consumption reports warnings in response metadata.
Reading creates no chat message.
Reminders include results from other source sessions. They open a read-only conversation preview while Main continues;
`pa_select_session(session_id)` selects that source for further discussion when Main is idle.
Snapshots and immediate-message events carry a bounded `inbox_messages` projection rebuilt from the delivery journal.
Desktop presentation/read and speech receipts remain separate from native `sent` state.

### Scheduled Task Queries

The chat agent maintains `{metadata_dir}/pa/cron.json`: a JSON list with exactly four required fields: stable `id`,
five-field `cron`, nonblank `query`, and boolean `enabled`. Each run decides whether its findings need immediate attention;
there is no configured delivery type. Read the latest file before patching, preserve other tasks and identities,
save atomically, then read back and call `check_json`. The scheduler validates and reloads valid edits automatically.
Remove extra fields from old definitions explicitly; runtime loading performs no compatibility conversion.
Screenshot tasks use the regular capture tool to compare current frames; there is no pre-model whole-run preflight.
Default tasks observe screen activity every five minutes and fetch relevant financial news hourly at the top of the hour.
They also search for fresh developments matching the user's interests daily at 08:00, 12:00, and 18:00. All three times
share the `daily_fresh_news` task identity and history.
Defaults seed missing files only; update an existing `cron.json` explicitly to change existing task schedules.

The `cron_schedule` background Job checks file modification time every second and owns one ReMe `CronJob` per enabled task.
Valid edits add, replace or stop schedules; invalid edits are reported while existing valid schedules remain active.
Replacement closes the old Job before starting the new one, so the same task cannot overlap. Jobs and agent work close
with the application. Each task starts at its next cron time; executions are serial per task and missed triggers are skipped.
`tasks_follow_active_hours` in `proactive.json` defaults to `true`: task execution checks the same `start_time` /
`end_time` and skips closed hours without model tokens. Set it to `false` for all-day cron execution; edits take effect
without restarting. Evening closure stops proactive communication; independent tasks follow their configured active hours.

Each task keeps one native Codex session across scheduled runs. One query executes the task, reads the latest valid result
and reminder history, and writes useful cumulative findings to the supplied staging Markdown path. With no valuable update,
it writes no file and returns `NO_UPDATE`. Summaries retain findings, necessary context, verified sources and a merged
24-hour timeline. Urgent findings may call `send_chat_message_immediate(content)` after reading fresh message history;
code publishes the report before submitting the reminder. Message text has no presentation-size character cap.

Published results live at `{metadata_dir}/pa/cron/{task_id}/YYYY-MM-DD/HHMMSSffffff.md`. Task identity and start time identify
the execution. `state.json` stores the native session, current run and durable completion intent. Recovery finishes saved
publication and delivery without another model query or duplicate result; obsolete reminder candidates are dropped while
retaining the report. Changed definitions supersede unfinished work while retaining the task's conversation. Missing native
sessions are recreated; transport failures do not discard history. Urgent reminders expire after 15 minutes and retain their
original target Main session. Proactive consumption and Desktop's `user_read_at` remain independent.

This replaces the four-query workflow, `recovery.json` and cron handoffs. Runtime loading does not convert old workflow
state or temporary-task records; recreate or migrate those records offline before upgrading, retaining source files.

Stop the service before upgrading an old workspace, migrate offline, then restart:

```bash
python -m scripts.migrate_pa_tasks /path/to/workspace
```

Use `--check` to validate and preview without writing; pass `--metadata-dir` and `--daily-dir` for custom directories.
The script copies old daily/cron and root-level results into PA, renames the three journals, and rewrites stored result,
task-state references. It retains originals and backs up in-place reference updates; conflicts fail before writing.
Historical inline run results become consumed Markdown archives, preserving content without triggering new proactive inputs.
`chat_inbox.jsonl` is published last. Startup rejects an unmigrated layout rather than seeding empty replacement state.
After migration, only the new paths are authoritative; retained originals are backups.

During its work query, proactive calls private tool `read_cron_results(limit=3)` once, then reads every returned path.
This consumes cron results only; one-shot temporary task results follow their separate immediate-delivery path.
The tool returns up to three latest unread snapshot paths **per configured task**, including paused tasks and previous days.
Bodies are read from files to avoid truncation in tool responses. Removed tasks are no longer scanned; their
files remain intact. Full inputs are checkpointed before frontmatter changes: selected snapshots become `read`, and older
unread snapshots covered by the cumulative summaries become `superseded`. Read means consumed by proactive, not read by
the user or sent as a reminder. Recovery context preserves inputs if the round fails. Relevance and reminder value remain
agent decisions. Proactive may also call `screen_capture` and available native `web_search` tools to investigate leads;
the proactive config enables live native search, subject to the configured provider's support.

The former `run_due_tasks` Job, Step and wakeup-driven scheduler are removed. `cron_agent` now accepts a complete `task`
object and is private; there is no compatibility entrypoint. Raw resource retention remains a separate CronJob.

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
