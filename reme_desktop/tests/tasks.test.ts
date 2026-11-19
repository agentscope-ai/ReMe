import assert from "node:assert/strict";
import test from "node:test";
import { requestWorkspace } from "../src/main/workspace.ts";
import type { ReMeClient } from "../src/main/reme-client.ts";
import {
  changedTaskSettings,
  taskPayload,
  type TaskRequest,
  type TaskSettings,
} from "../src/shared/tasks.ts";

const settings: TaskSettings = {
  enabled: false,
  cron: "0 8 * * *",
  min_trigger_interval_seconds: 0,
  notify_window: null,
};

test("task management forwards explicit settings and keeps task edits scoped by identity", async () => {
  const calls: unknown[] = [];
  const client = {
    request: async (job: string, body: object) => {
      calls.push({ job, body });
      return { success: true, answer: {}, metadata: {} };
    },
  } as unknown as ReMeClient;
  for (const request of [
    {
      type: "task",
      action: "create",
      agent_name: "tech_news",
      task_description: "原样\n",
      settings,
    },
    {
      type: "task",
      action: "update",
      agent_name: "tech_news",
      settings: { enabled: false, notify_window: null },
    },
    { type: "task", action: "read", agent_name: "tech_news" },
    {
      type: "task",
      action: "save",
      agent_name: "tech_news",
      content: "New\n",
      expected_mtime: "revision",
    },
  ] satisfies TaskRequest[])
    await requestWorkspace(client, request, new AbortController().signal);
  assert.deepEqual(calls, [
    {
      job: "manage_proactive_agent",
      body: {
        action: "create",
        agent_name: "tech_news",
        ...settings,
        task_description: "原样\n",
      },
    },
    {
      job: "manage_proactive_agent",
      body: {
        action: "update",
        agent_name: "tech_news",
        enabled: false,
        notify_window: null,
      },
    },
    { job: "agent_task", body: { action: "read", agent_name: "tech_news" } },
    {
      job: "agent_task",
      body: {
        action: "save",
        agent_name: "tech_news",
        content: "New\n",
        expected_mtime: "revision",
      },
    },
  ]);
});

test("task IPC rejects unrelated identities, unsafe writes and retired operations", () => {
  for (const request of [
    { action: "delete", agent_name: "Remy" },
    { action: "delete", agent_name: "TMP" },
    { action: "read", agent_name: "../task" },
    { action: "save", agent_name: "task", content: "Text" },
    {
      action: "save",
      agent_name: "task",
      content: " ",
      expected_mtime: "revision",
    },
    { action: "update", agent_name: "task", settings: {} },
    { action: "update", agent_name: "task", settings: { agent_name: "other" } },
    { action: "query", agent_name: "task" },
    { action: "status", agent_name: "task" },
    { action: "read", agent_name: "task", path: "daily/other.md" },
  ])
    assert.throws(() =>
      taskPayload({ type: "task", ...request } as TaskRequest),
    );
});

test("settings edits only patch changed fields, preserving concurrent pause and notification changes", () => {
  assert.deepEqual(
    changedTaskSettings(settings, { ...settings, cron: "0 9 * * *" }),
    { cron: "0 9 * * *" },
  );
  assert.deepEqual(
    changedTaskSettings(
      { ...settings, notify_window: { start: "09:00", end: "18:00" } },
      settings,
    ),
    { notify_window: null },
  );
  assert.deepEqual(changedTaskSettings(settings, { ...settings }), {});
});
