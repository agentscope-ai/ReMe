---
title: Proactive Runtime
description: How ReMe follows ongoing tasks, saves progress, and brings useful reminders into a conversation.
---

# Proactive Runtime

ReMe gives agents local, file-native long-term memory. Its Proactive Runtime brings that memory into ongoing tasks. It accepts new information or schedules a run, delegates execution to an agent runtime such as Codex through an `AgentWrapper`, saves the results, and lets the user know when their attention is needed. Each ongoing task has a named Proactive Runtime. AgentManager loads and manages background Runtimes from the workspace. Each Runtime sends reminders to Remy through its Outbox.

The [full illustrated guide is available in Chinese](../zh/proactive_runtime.md).

## How a task keeps running

Start ReMe with `reme start config=proactive`. Collectors send observations to event-triggered Runtimes through `manage_connector(action="send", runtime_names=..., events=...)`. Time-triggered tasks use an explicit `agent_type: time` and `schedule` expression. Use an event-triggered task to check new experiment logs and a separate time-triggered task to summarize progress every day at 18:00.

Each run opens a fresh native session and reads `TASK.md`. Inputs are selected by event time, then by event ID when times match. Observations can be processed together; each scheduled trigger gets its own run. The query retains the scheduled trigger's planned time; the record skill receives the actual run start time. A trigger planned for 08:00 and processed at 08:07 supplies those two times respectively.

The AgentWrapper supplies backend instructions and environment context. Remy's role prompt belongs to Chat.
The [execution Step](../../reme/steps/proactive/agent.py) assembles each background run from `TASK.md`, the trigger
inputs and the selected skills, then calls the Harness once. Its bilingual query and input templates live in
[`agent.yaml`](../../reme/steps/proactive/agent.yaml). The Runtime owns the stable capability binding, native sessions
and reliable result commitment. The Step fills each round's time, paths and search budget into a fresh copy of
the record skill, leaving cached templates unchanged.

Customize execution prompts through `jobs.run_proactive_agent.steps[0].prompt_dict`, using `query`,
`record_instruction`, `event_input` and `time_input` or their `_zh` variants. Runtime `prompt_dict` configures role
and initialization text.

Preset `TASK.md` text keeps task-specific work, output requirements and reminder conditions. The shared query appends
`record_instruction` only when `proactive-record` is enabled; the skill supplies the record format and paths.

Background Agents load the shared `proactive-record` skill by default. After completing the task work, they use it
for history, duplicate records and reminders, reminder priority and content, and Markdown output. A task whose
`skills` omit `proactive-record`, including `skills=[]`, can complete without a daily record and does not generate
record-based reminders. Notebook also loads `manage-notebook`. Runtime management remains with Remy's
`manage-proactive-runtime` skill.

The record skill searches the shared `daily_dir` for earlier findings and reminders. It checks the latest `recent_history_hours` first (24 hours by default) and makes up to `max_local_search_calls` keyword or regex searches per run (10 by default). The search budget is a prompt instruction. If the agent exceeds it, ReMe logs one warning and lets the run finish. Diagnostics under `reme.agent_run` include tool calls, model status, and the run outcome. Failed rounds retain their inputs for retry. The retry delay starts at 5 seconds and increases to at most 5 minutes by default.

### One record per run

With `proactive-record` enabled, the agent writes to `daily/YYYY-MM-DD/{agent_name}_HHMMSS.md` by default. A numeric suffix prevents same-second collisions. Observation runs with no key information can finish without a file by successfully replying `NO_KEY_INFORMATION`. Scheduled runs using the skill always require a record.

The Markdown body contains new information and its sources. It may be empty. Local links are relative to the record's directory. The agent writes two frontmatter fields:

- `reminder_priority` is `P0` for a due reminder or a decision that needs immediate attention, `P1` for important progress, `P2` for a useful discovery, or `skip`.
- `reminder_message` contains the reminder or requested result. With `skip`, it can be empty or give a brief reason.

The reader accepts leading blank lines and uses only these two fields for delivery. The proactive Notebook writes
the same Markdown format and loads the `manage-notebook` skill with the `manage_notebook` tool with kind=todo/idea.

The task can specify its own delivery conditions and priority. The agent checks historical records for repeated information and reminders.

### Commit and delivery

Without `proactive-record`, a successful model run commits its input receipt without validating a record or
registering a reminder.

The skill checks and corrects its output within the same execution. ReMe validates the saved file before committing
the run. A model error, timeout, cancellation or validation failure removes this round's draft if it has not been
changed externally, while preserving the inputs for a later task retry.

ReMe holds the file lock while checking the record, acknowledging its inputs, and registering any reminder in the
task outbox. If acknowledgement fails before the receipt is durable, the same draft cleanup applies and the inputs
remain pending. Once the receipt is durable, ReMe keeps the record and can resume registration after a restart.

After an abrupt process exit, inputs without a durable receipt remain pending; committed reminders resume from
their receipts and outbox state without repeating model execution. An abrupt exit cannot run draft cleanup, so
an uncommitted draft may remain in the daily directory. A record file alone does not prove commitment or delivery.

The workspace-relative record path is the reminder ID. The task outbox stores delivery facts. Remy persists notification references and processing receipts; its Chat outbox links native messages. Reminder text comes from the original daily record.
Remy selects reminders by priority and delivery time within the configured limits and notification window. A Chat decision
rechecks the current reminder under the file write lock so an edit can invalidate a stale decision.

Earlier native sessions remain available as history. Task questions use a temporary read-only fork of the latest task session, including while a run is in progress. The built-in Codex configuration sets `project_doc_max_bytes=0` and disables automatic skill injection by default. Global instructions under `CODEX_HOME` load independently.

## Remy and reminders {#chat-and-reminders}

Remy brings background tasks and reminders into the main Chat conversation. Users manage tasks through Chat and can change their requirements, inspect progress, pause execution, or end a task.

Each Agent has one stable `agent_name`. The trigger type of a background task is fixed at creation. `Remy` is reserved for the main Chat Agent. Creation writes `task_description` verbatim when `TASK.md` is absent. Use `task_document action=read/save` to edit requirements and pass the `mtime` returned by read as `expected_mtime` when saving. Use `manage_proactive_runtime action=update` to change execution settings.

Committed reminders reach the Chat Inbox. Priority, Chat idle time, and the task's notification window determine when reminders become eligible for processing. A temporary read-only fork receives the selected inbox entries in its prompt. It uses the current conversation to decide which entries warrant reading the source record, and leaves links inside the record unopened. It skips known, duplicate, handled or outdated information without assigning a new priority or adding action requirements. Explicit requests for reminders at a specified time or condition still apply. A silent Chat decision acknowledges the selected references in Remy's durable Inbox; it does not rewrite the source record.

User feedback can update the relevant task description or configuration for later runs. The daily record provides reminder content; the receiving Inbox owns processing state.

## Current work observations

[`codex_dialogue_cron`](./auto_resource.md#scheduled-codex-dialogue-collection) collects new Codex dialogue and
attachments. [`claude_code_dialogue_cron`](./auto_resource.md#scheduled-claude-code-dialogue-collection) collects
Claude Code dialogue using the same resource and Event contracts. Both send to `proactive_memory` and `proactive_note`.
`screen_capture_cron` independently captures all displays and sends to `proactive_memory`. All run every five minutes;
dialogue `init_steps` establish a baseline without importing old history. Fuller tool execution trajectories require
additional connectors. Collected observations must distinguish requested work, attempted
actions, and verified results.

See [Services and Deployment](./services.md#proactive-runtime) for configuration and interfaces.


## Task storage

Creation requires `agent_type: time` or `agent_type: event`. A time-triggered task also requires a `schedule` Cron expression; an event-triggered task has no schedule. Persisted configuration and Agent views use the same explicit type.
Time-triggered tasks can change their schedules. Events already in the queue keep their original planned time. The runtime processes each time trigger in its own round and can batch consecutive observations.
The observation cooldown starts after the last successful receipt. Time-triggered and event-triggered inputs belong to separate task identities.

Each task owns a directory under `reme_agent_dir`. Deleting a task stops its execution and preserves its files. Temporary query sessions are recorded as independent branches.

Remy's inbox durably retains references and processing receipts, and applies count and byte limits to each active batch. A missing or invalid source record remains pending and reports an error without blocking other reminders. Task input inboxes use durable claim, acknowledgement and retry.

`list_agent_inbox` returns input events for background tasks and reminder records for Remy. Remy's reminder records have the same fields as `list_task_reminders`, including the source path as `id` and the current `state`. Remy's inbox also reports `processing` for reminders in an active batch.

## Studio protocol

Studio reads `get_client_info` for capabilities. Management responses carry raw Runtime settings under config and a running flag. Sidebar views are supplied by list_agents and session endpoints.

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
