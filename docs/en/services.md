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
`default_memory` from `reme/steps/proactive/base_pa_step.yaml`, with per-path locking and atomic writes. The defaults cover financial
news and AI advances, and require valuable proactive
reminders without unimportant or repetitive interruptions. Existing files, including empty
ones, are preserved; there is no onboarding or empty-memory gate.

`pa_chat.py` supplies the chat Step (`pa_chat_step`), new-session and session-status Steps; `pa_chat.yaml` supplies Remy's persona
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

The `proactive` configuration forks the selected Main conversation to prepare a context summary, then starts a clean
agent session for seven work queries: combined conversation, memory and inbox reading; unread task summaries; injected-message
history; source research; message decisions; optional wake-configuration changes; and a closing TODO.
When the selected Main has no native chat turns, Proactive skips the opening summary and runs the seven work queries
with empty conversation context. Proactive messages alone do not count as chat history. Empty conversations do not block
proactive work; errors reading an existing session still fail explicitly.
Every wakeup prepares fresh Main context. The opening summary uses only the inherited Main conversation. The first work query
calls the private `read_conversation(max_turns=10)` tool and reads personal memory, inbox feedback and previous TODOs together
to understand newly completed Main rounds and maintain supported lasting memory. Tool bindings stay fixed.
Research checks reminder history first, selects at most three useful leads and searches and reads primary sources;
without a useful lead, it skips searching. All four agents preserve core conclusions, necessary background and evidence,
remaining concise but complete without fixed word, sentence or bullet counts.
External findings cite verified original URLs, optionally as `[Source name](URL)`; local files or screenshots use source names
and relevant times, without invented links. Unverified external claims are withheld.
Reminders select worthwhile findings by relevance, urgency and impact, merging related details without filling a quota.
Reminders explain findings and their relevance conversationally, keeping verification work internal; source links must
support the core claim, rather than only background. Verification, selection and wording remain agent judgments.
`query_message` weighs whether interrupting is worthwhile before choosing content. Repeated nonresponse calls for fewer
interruptions; an occasional gentle check-in may ask whether Master prefers fewer messages, without demanding a reply.
Light, welcome affection is allowed without guilt or invented distress. Empty content withdraws the older pending proactive
message; new content supersedes it while retaining discarded history.
Code accepts submissions, including empty decisions, only during `query_message`; other phases cannot consume the decision.
Nonempty decisions require a current reminder-history snapshot; the same validation applies to tool and runtime callers.
History changes during reading or before submission require a fresh read and decision. Empty silence decisions need no snapshot.
The phase clears after the model call completes or is cancelled. Accepted reminders retain independent background delivery.
Current preferences override older summaries; workspace names and screenshots do not establish project interests.
Consumed inputs and partial progress are checkpointed in `proactive_context.jsonl` before cursor advancement.
The final `query_todo` returns only unfinished work worth continuing, concrete next actions and essential sources or recheck
conditions. Completed items, forgotten facts and message drafts are excluded. Code atomically saves it to
`{metadata_dir}/pa/proactive_todo.md` and injects freshly read contents into the next work session. Current conversation,
memory and feedback take precedence; TODOs are not send approval. The sleep prompt uses the current interval (five minutes
by default); active hours and explicit pause settings still govern wakeups.
Failed or interrupted rounds retain recovery context and the previous TODO; successful TODO publication clears recovery.
The old `handoff.txt` is no longer created, read or updated; existing files are left intact.

Main context for Proactive and one-shot preparation fork their source session and append only a summary query,
retaining its instructions, model, turn settings and tool definitions. Cron-task handoffs continue in the work session.
Forking a known live empty thread without a persisted rollout starts an independent thread with its inherited settings
and tools, without inserting placeholder messages into the source. Persisted history continues to use native forks.
Scene-specific queries share a common requirement to preserve supported preferences,
interests, requirements, recent projects and goals, distinguishing lasting preferences, recent focus and one-shot requests.
Handoff prompts request a complete, clear context summary without prescribing a Markdown format.
Summary-only behavior is a prompt instruction; tool definitions remain available. Code saves the returned summary without
replacing the working session ID. The wrapper retains effective settings in session-side `.config.jsonl` files so forks can
restore them after unload or restart, with fresh application tool credentials. Legacy sessions without a settings record use
native fork defaults until their next normal invocation records the effective settings. Prefix preservation enables cache
reuse; actual cache hits depend on the provider.

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
| `proactive_context.jsonl` | Empty recovery journal |
| `proactive_todo.md` | Empty; later holds unfinished work |
| `chat_inbox.jsonl` | Empty |
| `cron_runs.jsonl` | Empty |

Inbox names identify their recipient: `proactive_inbox.md` carries Main's instructions to Proactive, with
`proactive_inbox_cursor.json` consumption state and `proactive_inbox_receipts.jsonl` request receipts.
`chat_inbox.jsonl` carries background messages addressed to Main and their delivery states.
Cron documents live under `cron/{task_id}/`; one-shot documents live under `temporary/{id}/`, both inside this PA directory.

Defaults are configured in `reme/config/proactive.yaml`. The proactive defaults are `start_time: "0700"`,
`end_time: "2330"`, and `interval_minutes: 5`. Existing files, including empty text files and empty task lists, are kept;
invalid existing JSON fails explicitly instead of being replaced with defaults. Each agent initializes missing personal
memory before invoking its model, using the localized defaults in `base_pa_step.yaml`.

Wakeups use the application timezone. Start is inclusive and end exclusive; overnight windows are supported, and equal
endpoints mean all day. Within active hours, the next delay starts after the previous round finishes. Outside the window,
the loop waits for the next opening. Sleep checks JSON edits at most every 30 seconds, so changing the file requires no
restart. Invalid configuration pauses model work while reporting the error. The former YAML wakeup delay, in-memory
override and `set_next_wakeup` tool are removed. Resource cleanup remains an independent CronJob.
Every model query rechecks active hours. Closed hours or an evening decision stop further proactive queries, retaining
partial checkpoints for the next opening. In-flight background queries are interrupted at the window deadline. Polling and policy checks spend no model tokens.

Optional conversational hints default to enabled. Set `hints_enabled: false` in `proactive.json` to disable sampling
and injection. `hint_probabilities` separately configures `morning`, `midday` and `evening`, defaulting to 1.0, 0.5 and 0.5;
each accepts numbers from 0 to 1. YAML `default_proactive` seeds new files; missing fields or periods use schema defaults.
Within active hours, sampling occurs at most once
per local day in each window: 06:00–12:00, 12:00–14:00, and the last 30 minutes before an evening closing time (no earlier
than 18:00). Overnight and all-day activity use an evening window after 21:00, never an evening hint in the morning.
Misses also consume the opportunity. A current chat is required. Results are saved in lazily created `hint_state.json`
to prevent rerolls after restart and avoid the last selected variant of each kind. Bilingual directions come from
`proactive_agent.yaml` and are injected only before the reminder decision. The agent may ignore them; ordinary useful
updates remain independent of sampling. Actual delivery is recorded in `chat_inbox.jsonl`. Sampled directions are not
personal memories or instructions to repeat in future rounds.
The 18 directions concern everyday life, care and playful surprises; only one evening direction summarizes confirmed todos.
Special reminder kinds are `morning`, `midday` and `evening`: each requires this round's matching sampled hint and
consumes one slot per local date, including failed or discarded submissions; a failed closing decision never reopens that window after an uncertain delivery. Normal information uses `normal`, without
explicit greeting wording. Every nonempty evening decision is the final proactive message for that window, even without
literal good-night wording. Pending closing messages cannot be replaced by ordinary reminders. Sent closing messages
stop proactive queries until the next opening; misses and empty decisions do not close the day. The reminder journal
owns this state across restarts and session switches, without a separate quiet-state file.

The backend consumes completed unread rounds across the latest 100 registered chats before context preparation.
Each user input stays with its replies; tools and injected messages are excluded. Defaults remain 20 rounds,
2000 characters per message and 20000 total. Read or checkpoint errors preserve cursors.

`write_proactive_inbox(message)` queues timestamped Markdown for Proactive's next wakeup. Put topics, reminder feedback
and settings requests in the message; Proactive interprets them and edits its own configuration during its configuration phase.
Queued or received means the message was saved or read, not that its requested work or setting change is complete.
While notifications are paused, outside active hours or blocked by invalid settings, explicit inbox requests still get
a settings-only round. It can repair or resume settings without researching or sending reminders; failed requests remain retryable.
`read_proactive_inbox()` checkpoints appended or edited content before advancing a content-hash
cursor, retaining the source file. It returns `input_path`, `round_id` and `record_kind` for reading the complete checkpoint,
plus `receipt_path` for original requests and latest statuses. Feedback bodies do not pass through bounded tool output.
Main chat and the Proactive memory phase share a write lock for personal memory.

`manage_temp_proactive_agent(action="list")` returns `items` and `total`, including each task's ID, query, execution time and
state. Listing is read-only; use the returned IDs when managing tasks.
`manage_temp_proactive_agent(action="create", query=..., run_at? | delay_minutes?, request_id?)` persists one-shot work tied to the source
Main session. Preparation runs in the background after that conversation releases its current turn. At the deadline,
a clean agent executes `task.query` and calls `send_chat_message_immediate(content)` before completing the same turn;
there is no separate sending turn. The management tool accepts update, pause,
resume or cancel with `task_id`; only create accepts `request_id`, and only create/update accept query/time fields.
`delay_minutes` supports fractional minutes, such as `0.5` for 30 seconds.
Revisions invalidate stale workers. Overdue tasks execute once on
restart and label delayed results. Temporary results display immediately without interrupting Main; voice still follows
quiet settings. Native history insertion waits for the source session to become available and uses a stable message ID.
Results remain in Main's Chat inbox and become read after their bubble is actually shown; submission alone does not mark them read.
Preparation failures retain an error and `retry_at`, retrying after 30 seconds. Prepared context and query results live in
`{metadata_dir}/pa/temporary/{id}/`, with identity and revision metadata for recovery if progress-state publication was interrupted.
At execution time, the agent receives an existing source transcript path for optional fresh context;
when the empty source has not produced a transcript, it uses the prepared context instead.
The immediate tool saves the result before delivery, so restart recovery delivers saved content without reexecuting the query.
It is bound to the active temporary task and accepts at most 3980 characters.
Use native file tools to read cron definitions in `cron.json`, execution records in `cron_runs.jsonl` under `{metadata_dir}/pa/`,
and result files under `{metadata_dir}/pa/cron/`. Recorded states do not prove a worker is
currently running. Read result bodies without changing consumption or user-read markers; no separate task-status tool remains.

Proactive uses `get_chat_messages(limit=10, offset=0)` to read each reminder's latest state, newest creation
first, without consuming history. Results include complete message content and delivery states; use `limit` and `offset` to paginate.
JSONL appends share an ID for successive `pending`, `sent`, `discarded` or `failed`
states, with creation/update/expiry/sent times and the target session. This replaces the former `is_draft` contract.
Main handles reminder follow-ups from the current conversation and explicit user context, without querying this history.

`send_chat_message(content, kind="normal")` uses a backend-bound sender policy; the model cannot override delivery.
For ordinary Proactive it accepts one decision per round, with the message schema's 4000-character upper bound. `kind` is one of
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
agent decisions. Long-term memory is maintained by Main and the Proactive memory phase; recent concerns and unfinished work belong in the handoff.
Forgotten facts must not be restored through old conversations or summaries.

Desktop's Remy sidebar switches between reminders and a scheduled-task result inbox. Read/unread badges appear on each
card; the result inbox supports task filtering and pagination. `proactive_inbox_view(action="list", ...)` is read-only,
with `kind` set to `all`, `task_result` or `instruction`, `filter` set to `all`, `pending` or `unread`, and pagination.
The `unread` and `task_id` filters require `task_result`; `unread` means not yet viewed by the user.
`action="read", entry_id=...` returns current source text and receipt
without advancing consumption. Partial reads show a warning and suppress a definitive pending count.
`task_results(action="read", path=..., mark_read=true)` independently records user reading while retaining Proactive's
consumption state. Received does not mean work completed; edited instructions are processed again.
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
`tasks_follow_active_hours` in `proactive.json` defaults to `true`: execution and summary queries check the same `start_time` /
`end_time` and skips closed hours without model tokens. Set it to `false` for all-day cron execution; edits take effect
without restarting. Evening closure stops proactive communication; independent tasks follow their configured active hours.

Each execution creates a fresh session with current personal-memory contents and the task's previous handoff injected
into the system prompt. Query 1 executes the configured task with all available tools and read-only personal memory.
Code supplies the latest completed result path across dates as `previous_path` for query 2.
It leads with this run's confirmed findings, preserving core conclusions, necessary background, times and evidence.
External material keeps original URLs; local files or screenshots use source names and relevant times. Results stay concise but
complete and end with an accumulated timeline that merges repeated themes and continuous periods, without appending each execution.
With no valuable update it writes no file and returns exactly `NO_UPDATE`; a missing file without this decision is an error.
After publication, query 3 considers urgency. Only an urgent candidate requires `get_chat_messages`, including content,
delivery state and pagination when needed. If still urgent and new, it calls `send_chat_message_immediate`; otherwise it
finishes quietly. Changed history requires a fresh read and decision before submission. Stable message IDs make retries safe.
Native tool definitions and injected arguments stay constant across all four queries; the server accepts this run's history
and immediate-send calls only during query 3, revoking that authority when the query ends.
No result or no Main conversation skips query 3. Query 4 returns an internal handoff in the same work session, atomically saved as `{metadata_dir}/pa/cron/{task_id}/handoff.md`.
Findings and cumulative results stay concise while retaining essential context and sources, without a fixed 400-character template.
Internal handoffs share the requirements for progress, decisions, constraints, preferences, interests, recent projects,
goals, unfinished work, next steps and necessary sources, within file/context budgets.
Background findings do not prove the user has seen them. Code validates the staged summary, adds
`task_id`, `run_id`, `created_at`, `status: unread` and `user_read_at: null`, and publishes it exclusively as:

```text
{metadata_dir}/pa/cron/{task_id}/YYYY-MM-DD/HHMMSS.md
```

Same-second collisions append the run ID instead of overwriting a result. Code manages `recovery.json` without exposing
its path in the system prompt. It retains completed phases,
the native work session and the target Main conversation bound at run start. Subsequent schedules resume unfinished work.
Completed query text remains in the native work session rather than being copied into recovery JSON.
Message or handoff failures do not delete published results, republish them or resend accepted messages.
Handoffs are saved even without new findings. Changed task definitions take precedence over old recovery records.
Urgent cron messages expire after 15 minutes; results remain available. Delivery records link back to their result path.
`cron_runs.jsonl` records run ID, phase, times, session ID, result and handoff paths or errors; Markdown owns result content.
Proactive consumption preserves the independent `user_read_at` field used by Desktop task-result browsing.

Stop the service before upgrading an old workspace, migrate offline, then restart:

```bash
python -m scripts.migrate_pa_tasks /path/to/workspace
```

Use `--check` to validate and preview without writing; pass `--metadata-dir` and `--daily-dir` for custom directories.
The script copies old daily/cron and root-level results into PA, renames the three journals, and rewrites stored result,
handoff and recovery references. It retains originals and backs up in-place reference updates; conflicts fail before writing.
Historical inline run results become consumed Markdown archives, preserving content without triggering new proactive inputs.
`chat_inbox.jsonl` is published last. Startup rejects an unmigrated layout rather than seeding empty replacement state.
After migration, only the new paths are authoritative; retained originals are backups.

During its task query, proactive calls private tool `read_cron_results(limit=3)` once, then reads every returned path.
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
