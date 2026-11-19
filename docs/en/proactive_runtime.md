---
title: Proactive Agent
description: How ReMe follows ongoing tasks, saves progress, and brings useful reminders into a conversation.
---

# ReMe Proactive

ReMe supports agents that follow an ongoing task across multiple runs. New information or a scheduled trigger starts a run. The agent reads its requirements and previous progress, processes the inputs, saves its findings, and decides whether a reminder is needed. Between runs, it waits without continuously calling the model.

The [full illustrated guide is available in Chinese](../zh/proactive_runtime.md).

## How a task keeps running

Five parts support the workflow:

- **Event inbox:** stores inputs in workspace files. Each agent processes one batch at a time; successful runs acknowledge their inputs, while failed runs leave them available for retry.
- **Scheduled triggers:** add a timer event to the task inbox when a configured schedule is due. The task prompt supplies the planned trigger time, actual execution time, and application timezone.
- **Workspace files:** keep task requirements in `TASK.md`, progress and evidence in daily records, and previous reminders in reminder files. Each run reads the latest requirements and progress before continuing.
- **Permissions:** the framework and Harness control file access, tools, and network access. Chat, event tasks, and scheduled tasks can search the web and open pages.
- **Outbox:** publishes reminder references after the reminder has been saved and input completion has been committed. Failed delivery can be retried independently of task execution. A full Chat Inbox defers publication while already queued reminders continue to be consumed.

Users own the workspace files and can inspect or edit them. Each automatic event or scheduled run starts a fresh native session and rebuilds context from the task description and daily summaries. Earlier sessions remain available as history. Task questions fork the latest task session and ask directly with its existing instructions and context, including while it is running. Query forks are read-only and do not replace the task session; the main Chat Agent continues its own conversation.

Event tasks use `proactive_query` to analyze incoming material, record new information and decide whether to remind. Scheduled tasks use `cron_query` to perform the task, record what happened, decide whether to report and update today's summary on every run. The prompts share environment and path aliases. Code validates the final files before committing the run, and logs distinguish validation, successful commits and delivery.

Failed task runs retry with increasing delays, starting at `retry_seconds` (5 seconds by default) and normally capped at 5 minutes. A successful run resets the delay. Per-run diagnostics are in response metadata `reme.agent_run`; process-local counters under `agent_runs` group runs, tokens, tool calls and elapsed time by `agent_type`. Model status and run result are separate, so valid artifacts accepted after a model error still count as a completed run.

The model writes records, reminders and summaries to private round drafts. After validation, publication uses the file jobs' per-path locks and checks the original content and file versions. Concurrent user edits are preserved and reported as conflicts. Links between drafts are rewritten to their final destinations.

Complete, valid drafts can be committed even if the model times out or reports failure afterward. Failed rounds restore only files they published whose versions still match. After uncertain receipt I/O, recovery checks the durable consumption log: uncommitted outputs are restored, while committed results are retained. Explicit cancellation stops the round and cleans up drafts. This recovery does not cover forced process termination.

Event tasks compare incoming material with existing summaries and related records, distinguishing facts from inference. A newly met reminder condition also counts as new information. Both task types require a nonblank `TASK.md`; missing daily summaries are optional.

Both task types allow ten local keyword or regex search calls using `rg`/`grep` per run by default. Multiple searches in one tool call count once. Ordinary file reads, image viewing and writes are excluded. Once enough information is available or the search limit is reached, the agent finishes with the available information and marks or omits unverified details.

The `proactive_agent_step` parameters `max_local_search_calls` and `max_timeline_entries` set the prompt limits for local search calls and appended timeline entries. Both default to 10 and accept positive integers. Exceeding the search limit logs one warning per run and execution continues. Diagnostics report `tool_calls`, `web_lookup_calls`, and `local_search_calls`. Codex's saved calls fill gaps in live events, including rejected calls. Local search accounting covers named search tools, native search actions, and explicit `rg`/`grep` commands, including pipelines. `rg --files` counts as a file listing.

For event tasks, the record uses frontmatter `action: write` for new information and includes a nonempty body. With no new information, the agent writes `action: skip` with an empty body and ends the run without creating a reminder or changing the summary. The decision file is retained. Missing required outputs, invalid actions and blank bodies where content is required fail the run and leave inputs available for retry.

Every record with `action: write` requires a nonempty, updated daily summary. For event tasks, this also applies when the reminder is skipped; the skipped reminder explains why no notification is needed. “Today's records” merges related items and retains necessary background and record links. “Today's reminders” preserves earlier entries and adds generated reminders without treating them as received or read.

Scheduled tasks report every run at P0 by default. Explicit delivery conditions and priorities in `TASK.md` override those defaults. The scheduled prompt requires an execution record with `action: write` each run. Both task types use the same reminder file format. Set `action` to `P0`, `P1` or `P2` to submit the body for delivery. Set `action: skip` and give a brief reason to skip delivery. Briefings and summaries describe the actual results and include necessary sources and result links.

Event tasks follow `TASK.md` and first exclude content that does not need a reminder, then select the first matching level in P0/P1/P2 order. They do not guess what the user knows from other conversations. Their reminders use one natural, concise sentence. After final validation of the reminder priority and updated summary, Outbox adds related paths, the reminder ID, creation time and acknowledgement fields before committing input completion. A generated reminder has not necessarily been delivered.

Before adding reminder metadata, Outbox checks the round's published file version under the file jobs' shared path lock. A concurrent user save preserves the user's file and leaves the input available for retry.

Remy's inbox is derived from committed reminder files and their acknowledgement fields. It selects one eligible batch by priority and creation time within the configured item and byte limits. Reminders outside the selected batch remain available for later evaluation. Disabled tasks, deleted tasks and notification windows affect eligibility.

Sending and silent decisions revalidate the reminder's message, priority and related paths under the file jobs' shared path locks, held through decision commit and acknowledgement. If the user has saved new content, the stale decision fails and the updated reminder remains unprocessed.

The default `proactive` configuration sets `project_doc_max_bytes=0` on the shared Codex app-server, disabling automatic
repository `AGENTS.md` loading for all Agents. Event and scheduled task execution sessions do not automatically inject
skill instructions; the main Chat Agent retains skills. Global instructions under `CODEX_HOME` load independently of this setting.

## Personal Runtime

Personal Runtime connects the main Chat Agent, background tasks, and reminder handling. Users manage tasks through Chat and can change their requirements, inspect progress, pause execution, or end a task.

Each Agent has one stable `agent_name`. Background tasks use an optional `cron` schedule, which also determines their display type. `Remy` is reserved for the main Chat Agent. Creation writes `task_description` verbatim when `TASK.md` is absent. Use the returned `task_path` to read and edit requirements. Use `manage_proactive_agent action=update` to change execution settings.

Committed reminders reach the Chat Inbox. Priority, Chat idle time, and the task's notification window determine when reminders become eligible for processing. A temporary Reminder fork receives the selected batch and uses the current conversation and the backend's read-only mode to decide whether the user still needs to be told. It skips known, duplicate, handled or outdated information without assigning a new priority or adding action requirements. Explicit requests for reminders at a specified time or condition still apply. A silent Chat decision preserves the task's generated record, reminder and summary.

User feedback can update the relevant task description or configuration for later runs. Reminder creation, delivery, Chat processing, and actual user reading are tracked separately.

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

Remy's Inbox reads committed reminder Markdown and its acknowledgement fields. It holds the active batch in memory and applies the configured count and byte limits. Task input inboxes use durable claim, acknowledgement and retry. Link workspace files with `[[path/to/file.md|Label]]` in task artifacts.

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
