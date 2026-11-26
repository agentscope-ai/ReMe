import assert from "node:assert/strict";
import test from "node:test";
import { requestWorkspace } from "../src/workspace.ts";
import type { ReMeClient } from "../src/reme-client.ts";
import {
  changedTaskSettings,
  taskPayload,
  type TaskRequest,
  type ProactiveRuntimeSettings,
} from "../src/tasks.ts";

const settings: ProactiveRuntimeSettings = {
  enabled: false,
  cron: "0 8 * * *",
  min_trigger_interval_seconds: 0,
  notify_window: null,
};

test("task management forwards settings and document operations for a tmp task", async () => {
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
      agent_name: "tmp",
      task_description: "原样\n",
      settings,
    },
    {
      type: "task",
      action: "update",
      agent_name: "tmp",
      settings: { enabled: false, notify_window: null },
    },
    { type: "task", action: "read", agent_name: "tmp" },
    {
      type: "task",
      action: "save",
      agent_name: "tmp",
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
        agent_name: "tmp",
        ...settings,
        task_description: "原样\n",
      },
    },
    {
      job: "manage_proactive_agent",
      body: {
        action: "update",
        agent_name: "tmp",
        enabled: false,
        notify_window: null,
      },
    },
    { job: "task_document", body: { action: "read", agent_name: "tmp" } },
    {
      job: "task_document",
      body: {
        action: "save",
        agent_name: "tmp",
        content: "New\n",
        expected_mtime: "revision",
      },
    },
  ]);
});

test("task IPC rejects unrelated identities, unsafe writes and retired operations", () => {
  for (const request of [
    { action: "read", agent_name: "Remy" },
    { action: "read", agent_name: "REMY" },
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
    { action: "delete", agent_name: "task" },
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
