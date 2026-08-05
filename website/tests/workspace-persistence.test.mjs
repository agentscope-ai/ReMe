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

test("workspace snapshot keeps chat session and finishes interrupted blocks", () => {
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
        ],
      },
    ],
  };

  const snapshot = prepareWorkspaceSnapshot([chat], chat.id);
  const restored = snapshot.tabs[0];

  assert.equal(restored.sessionId, "session-1");
  assert.equal(restored.streaming, false);
  assert.equal(restored.messages[0].blocks[0].status, "done");
  assert.equal(restored.messages[0].blocks[0].expanded, false);
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
