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
- **Scheduled triggers:** add a timer event to the task inbox when a configured schedule is due.
- **File memory:** keeps task requirements in `MEMORY.md`, progress and evidence in daily records, and previous reminders in reminder files. Each run reads the latest requirements and progress before continuing.
- **Permissions:** the framework and Harness control file access, tools, and network access. Event tasks currently analyze incoming files offline; scheduled tasks can search the web and extract pages.
- **Outbox:** publishes reminder references after the reminder has been saved and input completion has been committed. Failed delivery can be retried independently of task execution.

Users own the workspace files and can inspect or edit them. Tasks reuse their native sessions, with context compaction configured separately from the durable memory and progress files.

## Personal Runtime

Personal Runtime connects the main Chat Agent, background tasks, and reminder handling. Users manage tasks through Chat and can change their requirements, inspect progress, pause execution, or end a task.

Committed reminders reach the Chat Inbox. A temporary, read-only, offline Reminder fork uses the current conversation to decide whether to add a reminder to the main conversation or stay quiet. Priority, Chat idle time, and the task's notification window determine when reminders become eligible for processing.

User feedback can update the relevant task memory or configuration for later runs. Reminder creation, delivery, Chat processing, and actual user reading are tracked separately.

## Current work observations

The current collector captures screenshots and new Codex user messages and final answers. Claude Code and fuller tool execution trajectories require additional connectors. Collected observations must distinguish requested work, attempted actions, and verified results.

See [Services and Deployment](./services.md#agent-runtime) for configuration and interfaces, and [Work Observations](./work_observations.md) for collector details.
