import assert from "node:assert/strict";
import test from "node:test";
import { prepareWorkspaceSnapshot } from "../app/workspace-persistence.ts";

test("workspace snapshot reloads saved files but preserves unsaved drafts", () => {
  const saved = {
    id: "file:saved.md",
    type: "markdown",
    title: "saved.md",
    path: "saved.md",
    content: "saved",
    savedContent: "saved",
  };
  const draft = {
    id: "file:draft.md",
    type: "markdown",
    title: "draft.md",
    path: "draft.md",
    content: "draft",
    savedContent: "saved",
  };

  const snapshot = prepareWorkspaceSnapshot([saved, draft], draft.id);

  assert.deepEqual(snapshot.tabs[0], {
    ...saved,
    content: "",
    savedContent: "",
    loading: true,
    error: undefined,
  });
  assert.deepEqual(snapshot.tabs[1], {
    ...draft,
    loading: false,
    error: undefined,
  });
  assert.equal(snapshot.activeTabId, draft.id);
});

test("workspace snapshot keeps chat session and marks interrupted blocks failed", () => {
  const chat = {
    id: "agent:one",
    type: "agent",
    title: "Chat",
    sessionId: "session-1",
    streaming: true,
    messages: [
      {
        id: "assistant",
        role: "assistant",
        content: "",
        blocks: [
          {
            id: "think:one",
            type: "think",
            sourceType: "think",
            payloads: ["working"],
            status: "streaming",
            expanded: true,
          },
          {
            id: "tool:completed",
            type: "tool",
            name: "search",
            callPayloads: [],
            resultPayloads: ["Found"],
            status: "done",
            expanded: false,
          },
        ],
      },
    ],
  };

  const snapshot = prepareWorkspaceSnapshot([chat], chat.id);
  const restored = snapshot.tabs[0];

  assert.equal(restored.sessionId, "session-1");
  assert.equal(restored.streaming, false);
  assert.equal(restored.messages[0].blocks[0].status, "error");
  assert.equal(restored.messages[0].blocks[0].expanded, false);
  assert.equal(restored.messages[0].blocks[1].status, "done");
});

test("workspace snapshot preserves an open memory graph", () => {
  const graph = {
    id: "graph:wiki",
    type: "graph",
    title: "wiki graph",
    root: "wiki",
  };
  const snapshot = prepareWorkspaceSnapshot([graph], graph.id);

  assert.deepEqual(snapshot.tabs, [graph]);
  assert.equal(snapshot.activeTabId, graph.id);
});

test("restoring chat-only Studio drops PA views without losing chats or unsaved files", () => {
  const draft = {
    id: "file:draft.md",
    type: "markdown",
    path: "draft.md",
    content: "unsaved draft",
    savedContent: "saved",
  };
  const chat = {
    id: "agent:chat",
    type: "agent",
    mode: "chat",
    title: "Existing chat",
    sessionId: "chat-session",
    messages: [{ id: "question", role: "user", content: "Keep this question" }],
  };
  const legacy = {
    id: "agent:legacy",
    type: "agent",
    title: "Legacy chat",
    sessionId: "legacy-session",
    messages: [],
  };
  const pa = {
    ...chat,
    id: "agent:pa",
    mode: "pa",
    sessionId: "shared-pa-session",
  };

  const snapshot = prepareWorkspaceSnapshot([draft, chat, legacy, pa], pa.id);

  assert.deepEqual(
    snapshot.tabs.map((tab) => tab.id),
    [draft.id, chat.id, legacy.id],
  );
  assert.equal(snapshot.tabs[0].content, "unsaved draft");
  assert.equal(snapshot.tabs[1].sessionId, "chat-session");
  assert.deepEqual(snapshot.tabs[1].messages, chat.messages);
  assert.equal(snapshot.tabs[2].sessionId, "legacy-session");
  assert.equal(snapshot.activeTabId, legacy.id);
  assert.equal(
    prepareWorkspaceSnapshot([draft, chat, pa], chat.id).activeTabId,
    chat.id,
  );
  assert.deepEqual(prepareWorkspaceSnapshot([pa], pa.id), {
    tabs: [],
    activeTabId: undefined,
  });
});
