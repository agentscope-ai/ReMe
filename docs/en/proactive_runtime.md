---
title: Proactive Agent
description: How ReMe follows ongoing tasks, saves progress, and brings useful reminders into a conversation.
---

# ReMe Proactive

ReMe supports agents that follow an ongoing task across multiple runs. New information or a scheduled trigger starts a run. The agent reads its requirements and previous progress, processes the inputs, saves its findings, and decides whether a reminder is needed. Between runs, it waits without continuously calling the model.

The [full illustrated guide is available in Chinese](../zh/proactive_runtime.md).

## How a task keeps running

Start ReMe with `reme start config=proactive`. Each automatic run claims one batch of events or one scheduled trigger and opens a fresh native session. Event tasks look for key information in their events and attachments. Scheduled tasks do the work specified in `TASK.md`. Both use the same prompt and require a nonblank `TASK.md`.

The agent searches the shared `daily_dir` for earlier findings and reminders. It checks the latest `recent_history_hours` first (24 hours by default) and makes up to `max_local_search_calls` keyword or regex searches per run (10 by default). The search budget is a prompt instruction. If the agent exceeds it, ReMe logs one warning and lets the run finish. Diagnostics under `reme.agent_run` include tool calls, model status, and the run outcome. Failed runs retain their inputs for retry. The retry delay starts at 5 seconds and increases to at most 5 minutes by default.

### One record per run

The agent writes to `daily/YYYY-MM-DD/{agent_name}_HHMMSS.md` by default. A numeric suffix prevents same-second collisions. Event tasks with no key information can finish without a file by successfully replying `NO_KEY_INFORMATION`. A scheduled run always requires a record.

The Markdown body contains new information and its sources. It may be empty. Local links are relative to the record's directory. The agent writes two frontmatter fields:

- `reminder_priority` is `P0` for a due reminder or a decision that needs immediate attention, `P1` for important progress, `P2` for a useful discovery, or `skip`.
- `reminder_message` contains the reminder, the requested result, or a brief reason for skipping delivery.

The task can specify its own delivery conditions and priority. The agent checks historical records for repeated information and reminders.

### Commit and delivery

ReMe holds the workspace file-job lock while checking the record version, committing the input receipt, and marking initial delivery. A valid record can commit after a model error or timeout. If execution fails before the receipt is durable, ReMe removes the new record only if its version still matches the model's output. Once the receipt is durable, Outbox can retry interrupted delivery independently.

The workspace-relative record path is the reminder ID. Outbox writes `delivered_at` into the record. Remy adds `read_at` when it processes the reminder and `chat_message_ids` when it links a Chat message. Remy selects reminders by priority and delivery time within the configured limits. The notification window controls automatic Chat processing. Users can browse committed reminders at any time. A Chat decision rechecks the current reminder under the file write lock so an edit can invalidate a stale decision.

After upgrading from a build that stored standalone reminders under `reme_agents/{agent_name}/reminder/`, those files remain available in the workspace. Automatic delivery and reminder views use the new daily records. Finish pending reminders on the older build if they still need to be delivered.

Earlier native sessions remain available as history. Task questions use a temporary read-only fork of the latest task session, including while a run is in progress. The built-in Codex configuration sets `project_doc_max_bytes=0` and disables automatic skill injection for task runs. Global instructions under `CODEX_HOME` load independently.

## Personal Runtime

Personal Runtime connects the main Chat Agent, background tasks, and reminder handling. Users manage tasks through Chat and can change their requirements, inspect progress, pause execution, or end a task.

Each Agent has one stable `agent_name`. Background tasks use an optional `cron` schedule, which also determines their display type. `Remy` is reserved for the main Chat Agent. Creation writes `task_description` verbatim when `TASK.md` is absent. Use the returned `task_path` to read and edit requirements. Use `manage_proactive_agent action=update` to change execution settings.

Committed reminders reach the Chat Inbox. Priority, Chat idle time, and the task's notification window determine when reminders become eligible for processing. A temporary Reminder fork receives the selected batch and uses the current conversation and the backend's read-only mode to decide whether the user still needs to be told. It skips known, duplicate, handled or outdated information without assigning a new priority or adding action requirements. Explicit requests for reminders at a specified time or condition still apply. A silent Chat decision marks the reminder as processed in its source record.

User feedback can update the relevant task description or configuration for later runs. The same record provides the reminder and its processing state.

## Current work observations

`codex_dialogue_cron` collects new Codex user messages and final answers. `screen_capture_cron` independently captures
all displays. Both run every five minutes and send observations to the `work` Agent. The dialogue Job first runs its
`init_steps` at startup to establish the reading baseline without importing old history. Claude Code and fuller tool
execution trajectories require additional connectors. Collected observations must distinguish requested work, attempted
actions, and verified results.

See [Services and Deployment](./services.md#agent-runtime) for configuration and interfaces.


## Task storage

The optional `cron` field determines when a task runs. Omit it for event-driven tasks, set an expression for scheduled tasks, and set it to null to return to event-driven execution. Agent views derive `agent_type` from the schedule.
Queued inputs keep their original trigger type after a schedule change. The runtime processes each scheduled trigger in its own round and batches consecutive observations separately.
The configured cooldown applies when the next batch contains observations. A scheduled trigger at the front of the queue can run during that cooldown.

Each task owns a directory under `reme_agent_dir`. Deleting a task stops its execution and preserves its files. Temporary query sessions are recorded as independent branches.

Remy's Inbox reads delivered reminders from the daily records. It holds the active batch in memory and applies the configured count and byte limits. Task input inboxes use durable claim, acknowledgement and retry.

## Desktop protocol

The desktop reads `get_client_info` for workspace settings, served jobs and speech capabilities. Task create, status
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

Python client schemas generate `reme_desktop/src/shared/wire.ts` and an executable serialization fixture. Regenerate
with `python -m reme_desktop.scripts.client-contract`; `--check` verifies that committed artifacts match the schemas.
