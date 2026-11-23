import assert from "node:assert/strict";
import test from "node:test";
import { resolveWorkspaceLink } from "../src/links.ts";
import { requestWorkspace } from "../src/workspace.ts";
import type { ReMeClient } from "../src/reme-client.ts";

test("workspace links resolve documents, wikilinks and contained native paths", () => {
  assert.equal(
    resolveWorkspaceLink("../detail.md", "daily/2026-11-15/summary.md"),
    "daily/detail.md",
  );
  assert.equal(
    resolveWorkspaceLink(
      "#reme-file=daily%2Freport.md",
      "reme_agents/Remy/MEMORY.md",
    ),
    "daily/report.md",
  );
  assert.equal(
    resolveWorkspaceLink(
      "file:///work/reme/resource/image.png",
      "",
      "/work/reme",
    ),
    "resource/image.png",
  );
  assert.equal(
    resolveWorkspaceLink("/work/reme/daily/report.md#result", "", "/work/reme"),
    "daily/report.md",
  );
  assert.equal(resolveWorkspaceLink("#reme-file=note"), "note.md");
  assert.equal(
    resolveWorkspaceLink("C:\\work\\reme\\daily\\note.md", "", "C:/work/reme"),
    "daily/note.md",
  );
  for (const url of [
    "../outside.md",
    "/work/other/note.md",
    "/work/reme/../outside.md",
    "file://remote/work/reme/note.md",
    "javascript:alert(1)",
    "~/.env",
    "%00",
    "#heading",
    "%E0%A4%A",
  ]) {
    assert.equal(resolveWorkspaceLink(url, "", "/work/reme"), undefined, url);
  }
});

test("workspace resource requests use the new scoped contracts and reject retired requests", async () => {
  const calls: [string, unknown][] = [];
  const client = {
    request: async (job: string, body: unknown) => {
      calls.push([job, body]);
      return { success: true, answer: {}, metadata: {} };
    },
  } as unknown as ReMeClient;
  const signal = new AbortController().signal;
  await requestWorkspace(
    client,
    { type: "outbox", agentName: "finance", state: "pending" },
    signal,
  );
  await requestWorkspace(
    client,
    { type: "inbox", agentName: "任务", state: "processing", offset: 20 },
    signal,
  );
  const record = `daily/${"archive/".repeat(16)}2026-11-22/finance_090000.md`;
  await requestWorkspace(
    client,
    { type: "reminder", agentName: "finance", id: record },
    signal,
  );
  assert.deepEqual(calls, [
    [
      "list_task_reminders",
      { agent_name: "finance", state: "pending", offset: 0, limit: 20 },
    ],
    [
      "list_agent_inbox",
      { agent_name: "任务", state: "processing", offset: 20, limit: 20 },
    ],
    [
      "read_agent_reminder",
      {
        agent_name: "finance",
        message_id: record,
      },
    ],
  ]);
  for (const request of [
    { type: "proactiveInbox" },
    { type: "outbox", agentName: "Remy", offset: -1 },
    { type: "inbox", agentName: "finance", limit: 101 },
    { type: "reminder", agentName: "Remy", id: "" },
  ]) {
    await assert.rejects(
      requestWorkspace(
        client,
        request as Parameters<typeof requestWorkspace>[1],
        signal,
      ),
    );
  }
  assert.equal(calls.length, 3);
});
