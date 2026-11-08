import assert from "node:assert/strict";
import test from "node:test";
import { StateUpdates, viewState } from "../src/main/view-state.ts";
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
      streaming: false,
      status: null,
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

test("only the selected message preview crosses into the companion, and full chat text stays intact", () => {
  const value = state();
  const message = {
    id: "answer",
    role: "assistant" as const,
    source: "chat" as const,
    content: "a".repeat(1000),
    completed: false,
    streaming: true,
    status: null,
  };
  value.messages = [message];
  value.surface = {
    kind: "message",
    message,
    presentationId: 1,
    paused: false,
  };
  const companion = viewState(value, true);
  assert.deepEqual(companion.messages, []);
  assert.equal(companion.surface?.kind, "message");
  if (companion.surface?.kind !== "message") throw new Error("Missing preview");
  assert.equal(companion.surface.message.id, message.id);
  assert.equal(companion.surface.message.content, `${"a".repeat(400)}…`);
  assert.equal(viewState(value, false).messages[0].content.length, 1000);
  assert.equal(
    message.content.length,
    1000,
    "projection cannot truncate canonical history",
  );
});
