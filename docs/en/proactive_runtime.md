---
title: Proactive Runtime
description: How ReMe follows ongoing tasks, saves progress, and brings useful reminders into a conversation.
---

# Proactive Runtime

ReMe gives agents local, file-native long-term memory. Its Proactive Runtime brings that memory into ongoing tasks. It accepts new information or schedules a run, delegates execution to an agent runtime such as Codex through an `AgentWrapper`, saves the results, and lets the user know when their attention is needed. One Proactive Runtime coordinates multiple tasks and Remy's conversation.

The [full illustrated guide is available in Chinese](../zh/proactive_runtime.md).

## How a task keeps running

Start ReMe with `reme start config=proactive`. Outside collectors can send observations through `append_task_events(agent_name, events)`. An optional `cron` schedule appends a trigger to the same task inbox. For example, one task can check new experiment logs as they arrive and summarize progress every day at 18:00.

Each run opens a fresh native session and reads `TASK.md`. Consecutive observations can be processed together. Each scheduled trigger gets its own run in queue order. The prompt shows both the planned time of a scheduled trigger in **Events for this run** and the actual start time in **Background**. A trigger planned for 08:00 and processed at 08:07 shows both times.

The agent searches the shared `daily_dir` for earlier findings and reminders. It checks the latest `recent_history_hours` first (24 hours by default) and makes up to `max_local_search_calls` keyword or regex searches per run (10 by default). The search budget is a prompt instruction. If the agent exceeds it, ReMe logs one warning and lets the run finish. Diagnostics under `reme.agent_run` include tool calls, model status, and the run outcome. Failed rounds preserve written records and retain their inputs for retry. The retry delay starts at 5 seconds and increases to at most 5 minutes by default.

### One record per run

The agent writes to `daily/YYYY-MM-DD/{agent_name}_HHMMSS.md` by default. A numeric suffix prevents same-second collisions. Observation runs with no key information can finish without a file by successfully replying `NO_KEY_INFORMATION`. A scheduled run always requires a record.

The Markdown body contains new information and its sources. It may be empty. Local links are relative to the record's directory. The agent writes two frontmatter fields:

- `reminder_priority` is `P0` for a due reminder or a decision that needs immediate attention, `P1` for important progress, `P2` for a useful discovery, or `skip`.
- `reminder_message` contains the reminder, the requested result, or a brief reason for skipping delivery.

The task can specify its own delivery conditions and priority. The agent checks historical records for repeated information and reminders.

### Commit and delivery

After a completed turn, ReMe can correct invalid frontmatter with one read-only follow-up in the same session.
ReMe applies the validated correction only if the record version still matches and preserves the Markdown body.

ReMe holds the file lock while checking the record, acknowledging its inputs, and registering any reminder in the
task outbox. A valid record can complete after a model error or timeout. If acknowledgement fails before the receipt
is durable, the Markdown stays in place and the inputs remain pending. Once the receipt is durable, ReMe can resume
registration after a restart.

The workspace-relative record path is the reminder ID. The task outbox stores `delivered_at`, `read_at`, and linked
Chat message IDs. Remy's inbox reads the same registrations and loads the reminder text from the original daily record.
Remy selects reminders by priority and delivery time within the configured limits and notification window. A Chat decision
rechecks the current reminder under the file write lock so an edit can invalidate a stale decision.

Earlier native sessions remain available as history. Task questions use a temporary read-only fork of the latest task session, including while a run is in progress. The built-in Codex configuration sets `project_doc_max_bytes=0` and disables automatic skill injection for task runs. Global instructions under `CODEX_HOME` load independently.

## Remy and reminders {#chat-and-reminders}

Remy brings background tasks and reminders into the main Chat conversation. Users manage tasks through Chat and can change their requirements, inspect progress, pause execution, or end a task.

Each Agent has one stable `agent_name`. Background tasks use an optional `cron` schedule, which also determines their display type. `Remy` is reserved for the main Chat Agent. Creation writes `task_description` verbatim when `TASK.md` is absent. Use `task_document action=read/save` to edit requirements and pass the `mtime` returned by read as `expected_mtime` when saving. Use `manage_proactive_agent action=update` to change execution settings.

Committed reminders reach the Chat Inbox. Priority, Chat idle time, and the task's notification window determine when reminders become eligible for processing. A temporary read-only fork receives the selected inbox entries in its prompt. It uses the current conversation to decide which entries warrant reading the source record, and leaves links inside the record unopened. It skips known, duplicate, handled or outdated information without assigning a new priority or adding action requirements. Explicit requests for reminders at a specified time or condition still apply. A silent Chat decision marks the reminder as processed in its source record.

User feedback can update the relevant task description or configuration for later runs. The same record provides the reminder and its processing state.

## Current work observations

`codex_dialogue_cron` collects new Codex user messages and final answers. `screen_capture_cron` independently captures
all displays. Both run every five minutes and send observations to the `work` Agent. The dialogue Job first runs its
`init_steps` at startup to establish the reading baseline without importing old history. Claude Code and fuller tool
execution trajectories require additional connectors. Collected observations must distinguish requested work, attempted
actions, and verified results.

See [Services and Deployment](./services.md#proactive-runtime) for configuration and interfaces.


## Task storage

Tasks can receive observations with or without a schedule. Set the optional `cron` field to add scheduled triggers and set it to `null` to stop producing new ones. Agent views derive the display field `agent_type` from the presence of a schedule.
Queued inputs keep their original trigger type after a schedule change. The runtime processes each scheduled trigger in its own round and batches consecutive observations separately.
The observation cooldown starts after the last successful run. A scheduled trigger at the front of the queue can run during that cooldown. A trigger behind waiting observations keeps its place in the queue.

Each task owns a directory under `reme_agent_dir`. Deleting a task stops its execution and preserves its files. Temporary query sessions are recorded as independent branches.

Remy's inbox holds the active batch in memory and applies the configured count and byte limits. Task input inboxes use durable claim, acknowledgement and retry.

`list_agent_inbox` returns input events for background tasks and reminder records for Remy. Remy's reminder records have the same fields as `list_task_reminders`, including the source path as `id` and the current `state`. Remy's inbox also reports `processing` for reminders in an active batch.

## Studio protocol

Studio reads `get_client_info` for workspace settings, served jobs and speech capabilities. Task create, status
and update return the same Agent view as the sidebar; the list action wraps these views in `items` and `total`.
`status` is authoritative for execution state; clients derive the running indicator from `status=active`.

`read_agent_session` returns the Agent, ordered messages and activities keyed by `agent_name`. A subscription starts
with `snapshot`; reconnects, session changes and history replacements use the same atomic replacement. `update`
contains complete-message upserts and current Agent/activity state. `heartbeat` carries no application state. Only
Remy's subscription includes the global Agent roster and background activities.

The backend projects native history and live output into text, summary, tool and attachment blocks. For Remy, the
session subscription carries live message updates and `ask_agent` reports turn boundaries and failures. For task
questions, `ask_agent` streams complete message upserts in DATA chunks. DONE closes the stream. Native transcripts
and workspace files remain authoritative.

From the repository root, run `python reme_studio/client/scripts/client-contract.py` to regenerate
`reme_studio/client/src/wire.ts` and the serialization fixture from the Python schemas. Add `--check` to verify them.
