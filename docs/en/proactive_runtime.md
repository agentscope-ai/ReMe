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
- **Scheduled triggers:** add a timer event to the task inbox when a configured schedule is due. The agent follows the task description, with the current time supplied in the system prompt; trigger times remain in scheduler records.
- **Workspace files:** keep task requirements in `TASK.md`, progress and evidence in daily records, and previous reminders in reminder files. Each run reads the latest requirements and progress before continuing.
- **Permissions:** the framework and Harness control file access, tools, and network access. Chat, event tasks, and scheduled tasks can search the web and open pages.
- **Outbox:** publishes reminder references after the reminder has been saved and input completion has been committed. Failed delivery can be retried independently of task execution.

Users own the workspace files and can inspect or edit them. Each automatic event or scheduled run starts a fresh native session and rebuilds context from the task description and daily summaries. Earlier sessions remain available as history. Task questions fork the latest task session and ask directly with its existing instructions and context, including while it is running. Query forks are read-only and do not replace the task session; the main Chat Agent continues its own conversation.

Each run uses one query with five steps: read the task and existing summaries, process the inputs and related history, write the current record, decide whether to remind, and update today's summary with useful progress. Shared environment and path aliases belong in the system prompt. Code validates the final files and branch consistency before committing the run; logs distinguish validation, successful commits and delivery.

Complete, valid files can be committed even if the model times out or reports failure afterward. Incomplete outputs, failed validation or a failed commit restore the previous daily summary and remove only this round's new record and uncommitted reminder. Durably acknowledged results are retained. Explicit cancellation stops the round. This recovery does not cover forced process termination.

Before deciding whether information is new, the agent searches related daily records when background or duplicate checks are needed. It distinguishes facts from inference and event times from collection or recording times. Due reminders and newly satisfied reminder conditions also count as new information. A missing or blank `TASK.md` fails the run.

The record uses frontmatter `action: write` for new information and includes a nonempty body. With no new information, the agent writes `action: skip` with an empty body and ends the run without creating a reminder or changing the summary. The decision file is retained. Missing required outputs, invalid actions and blank bodies where content is required fail the run and leave inputs available for retry.

When there is new information, the agent writes its reminder decision. A skipped reminder explains why no notification is needed; both the new record and skipped reminder are retained. New progress may also update the journal without generating a reminder; earlier reminder entries remain intact. For a generated reminder, the agent updates today's summary under “Today's journal” and “Today's reminders”, merging related items and retaining necessary record links. The summary records generated reminders without treating them as received or read.

Task requirements in `TASK.md`, current inputs and related history determine whether to generate a reminder; the agent does not guess what the user knows from other conversations. It first excludes content that does not need a reminder, then selects the first matching level in P0/P1/P2 order. Explicitly requested recurring reminders are assessed at each occurrence. After final validation of the reminder priority and updated summary, Outbox adds related paths, the reminder ID, creation time and acknowledgement fields before committing input completion. A generated reminder has not necessarily been delivered.

The default `proactive` configuration sets `project_doc_max_bytes=0` on the shared Codex app-server, disabling automatic
repository `AGENTS.md` loading for all Agents. Event and scheduled task execution sessions do not automatically inject
skill instructions; the main Chat Agent retains skills. Global instructions under `CODEX_HOME` load independently of this setting.

## Personal Runtime

Personal Runtime connects the main Chat Agent, background tasks, and reminder handling. Users manage tasks through Chat and can change their requirements, inspect progress, pause execution, or end a task.

Each Agent has one stable `agent_name` and an `agent_type`: `chat`, `proactive` or `cron`. The main Chat Agent is `Remy`; task names use 1–3 simple lowercase English words joined by underscores. Creation writes `task_description` verbatim to `TASK.md` only if the file is absent. Create, list and status return `task_path`; read the current file before updating the task description and read it back after saving. Use the configuration to change execution settings.

Committed reminders reach the Chat Inbox. Priority, Chat idle time, and the task's notification window determine when reminders become eligible for processing. A temporary Reminder fork receives the selected batch and uses the current conversation and the backend's read-only mode to decide whether the user still needs to be told. It skips known, duplicate, handled or outdated information without assigning a new priority or adding action requirements. Explicit requests for reminders at a specified time or condition still apply. A silent Chat decision preserves the task's generated record, reminder and summary.

User feedback can update the relevant task description or configuration for later runs. Reminder creation, delivery, Chat processing, and actual user reading are tracked separately.

## Current work observations

`codex_dialogue_cron` collects new Codex user messages and final answers. `screen_capture_cron` independently captures
all displays. Both run every five minutes and send observations to the `work` Agent. The dialogue Job first runs its
`init_steps` at startup to establish the reading baseline without importing old history. Claude Code and fuller tool
execution trajectories require additional connectors. Collected observations must distinguish requested work, attempted
actions, and verified results.

See [Services and Deployment](./services.md#agent-runtime) for configuration and interfaces.
