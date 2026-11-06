import assert from "node:assert/strict";
import test from "node:test";
import { StateUpdates } from "../src/main/view-state.ts";
import { mergeState } from "../src/renderer/desktop.ts";
import type { DesktopState } from "../src/shared/protocol.ts";

const state = () =>
  ({
    revision: 1,
    messages: [],
    pending: null,
    workspace: "/workspace",
    focusMessageId: null,
    behavior: {
      animation: { id: 1, clip: "idle", mode: "loop" },
      label: "idle",
      notice: null,
      error: "",
    },
  }) as unknown as DesktopState;

test("history and local prompts never enter companion patches, and revisions alone create no IPC", () => {
  const value = state();
  const companion = new StateUpdates(true);
  const chat = new StateUpdates(false);
  companion.update(value);
  chat.update(value);
  value.messages = [
    {
      id: "m",
      role: "user",
      source: "chat",
      content: "private text",
      completed: true,
      chunks: [],
    },
  ];
  value.pending = {
    id: "p",
    query: "private query",
    text: "one",
    streaming: true,
    sessionId: "s",
  };
  value.revision++;
  assert.equal(companion.update(value), null);
  assert.equal(chat.update(value)?.messages?.[0].content, "private text");
  value.pending.text = "two";
  value.revision++;
  assert.equal(chat.update(value)?.pending?.text, "two");
  value.behavior = {
    ...value.behavior,
    label: "thinking",
    animation: { id: 2, clip: "thinking", mode: "loop" },
  };
  value.revision++;
  const patch = companion.update(value)!;
  assert.deepEqual(Object.keys(patch).sort(), ["behavior", "revision"]);
  assert.ok(!JSON.stringify(patch).includes("private"));
  const chatPatch = chat.update(value)!;
  assert.equal(chatPatch.messages, undefined);
  value.revision++;
  assert.equal(companion.update(value), null);
  assert.equal(chat.update(value), null);
});

test("a delayed bootstrap cannot replace newer patches, or drop untouched fields", () => {
  const baseline = state();
  const merged = [
    { revision: 0, workspace: "old" },
    { revision: 2, unread: 3 },
    { revision: 3, connection: "connected" as const },
  ].reduce(mergeState, baseline);
  assert.equal(merged.workspace, "/workspace");
  assert.equal(merged.unread, 3);
  assert.equal(merged.connection, "connected");
  assert.equal(mergeState(merged, { revision: 2, unread: 0 }), merged);
});
