import assert from "node:assert/strict";
import test from "node:test";
import { resolveWorkspaceLink } from "../src/shared/links.ts";
import { plainText } from "../src/renderer/resources.ts";
import { requestWorkspace } from "../src/main/workspace.ts";
import type { ReMeClient } from "../src/main/reme-client.ts";

test("workspace links resolve documents, wikilinks and contained native paths", () => {
  assert.equal(
    resolveWorkspaceLink("../detail.md", "daily/2026-11-15/summary.md"),
    "daily/detail.md",
  );
  assert.equal(
    resolveWorkspaceLink(
      "#reme-file=daily%2Freport.md",
      "reme_agents/chat/MEMORY.md",
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

test("resource summaries keep text without Markdown control characters", () => {
  assert.equal(
    plainText("## 新发现\n\n**实验完成**，查看[原文](daily/result.md)。"),
    "新发现 实验完成，查看原文。",
  );
  assert.equal(plainText("- A\n- B\n\n`code`"), "A B code");
});

test("desktop resource requests use the new scoped contracts and reject retired requests", async () => {
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
    { type: "outbox", agentId: "finance", state: "pending" },
    signal,
  );
  await requestWorkspace(
    client,
    { type: "inbox", agentId: "任务", state: "processing", offset: 20 },
    signal,
  );
  await requestWorkspace(
    client,
    { type: "reminder", agentId: "finance", id: "message-1", markRead: true },
    signal,
  );
  assert.deepEqual(calls, [
    [
      "agent_outbox",
      { agent_id: "finance", state: "pending", offset: 0, limit: 20 },
    ],
    [
      "agent_inbox",
      { agent_id: "任务", state: "processing", offset: 20, limit: 20 },
    ],
    [
      "agent_reminder",
      {
        agent_id: "finance",
        message_id: "message-1",
        mark_read: true,
      },
    ],
  ]);
  for (const request of [
    { type: "proactiveInbox" },
    { type: "outbox", agentId: "chat", offset: -1 },
    { type: "inbox", agentId: "finance", limit: 101 },
    { type: "reminder", agentId: "chat", id: "", markRead: true },
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
