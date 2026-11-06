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
    mode: "chat",
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

test("workspace restoration keeps independent chats, one PA view, and file drafts", () => {
  const draft = {
    id: "draft",
    type: "markdown",
    path: "draft.md",
    content: "unsaved",
    savedContent: "",
  };
  const agent = (id, mode) => ({
    id,
    type: "agent",
    mode,
    title: "",
    messages: [],
  });
  const snapshot = prepareWorkspaceSnapshot(
    [
      draft,
      agent("one", "chat"),
      agent("two", "chat"),
      agent("pa-old", "pa"),
      agent("pa-current", "pa"),
      agent("legacy"),
      agent("invalid", "unknown"),
    ],
    "pa-current",
  );
  assert.deepEqual(
    snapshot.tabs.map((tab) => tab.id),
    ["draft", "one", "two", "pa-current", "legacy"],
  );
  assert.equal(snapshot.tabs[0].content, "unsaved");
  assert.equal(snapshot.activeTabId, "pa-current");
});

test("workspace upgrade preserves legacy chat selection, session, and local messages", () => {
  const legacy = {
    id: "agent:legacy",
    type: "agent",
    title: "Existing conversation",
    sessionId: "existing-session",
    streaming: false,
    messages: [
      {
        id: "local",
        role: "user",
        content: "Browser-only message",
        localOnly: true,
      },
    ],
  };
  const snapshot = prepareWorkspaceSnapshot([legacy], legacy.id);

  assert.deepEqual(snapshot, {
    tabs: [{ ...legacy, mode: "chat" }],
    activeTabId: legacy.id,
  });
  assert.deepEqual(
    prepareWorkspaceSnapshot(snapshot.tabs, snapshot.activeTabId),
    snapshot,
  );
});
