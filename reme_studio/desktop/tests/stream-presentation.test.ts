import assert from "node:assert/strict";
import test from "node:test";
import {
  DesktopUpdates,
  mergeState,
  STREAM_RENDER_MS,
} from "../src/renderer/state.ts";
import type { StudioState, DisplayMessage } from "../../client/src/protocol.ts";

const message = (text: string, completed = false): DisplayMessage => ({
  id: "reply",
  turnId: "turn",
  role: "assistant",
  source: "chat",
  content: text,
  items: [
    { id: "intro", type: "text", text: "Checking", status: "completed" },
    {
      id: "answer",
      type: "text",
      text,
      status: completed ? "completed" : "running",
    },
  ],
  completed,
  streaming: !completed,
  status: null,
});
const state = (messages = [message("First")]): StudioState =>
  ({
    revision: 0,
    sessionId: "session",
    selectedAgentName: "Remy",
    messages,
    pending: null,
  }) as StudioState;

test("IPC replacements retain unchanged messages and timeline items, including after history insertion", () => {
  const history = { ...message("Saved", true), id: "saved", turnId: "saved" };
  const initial = state([history, message("First")]);
  const next = mergeState(initial, {
    revision: 1,
    messages: structuredClone([history, message("First second")]),
  });
  assert.equal(next.messages[0], initial.messages[0]);
  assert.notEqual(next.messages[1], initial.messages[1]);
  assert.equal(next.messages[1].items![0], initial.messages[1].items![0]);
  const replay = mergeState(next, {
    revision: 2,
    messages: structuredClone(next.messages),
  });
  assert.equal(replay.messages, next.messages);
  const prepended = mergeState(replay, {
    revision: 3,
    messages: [
      { ...history, id: "earlier", turnId: "earlier" },
      ...structuredClone(replay.messages),
    ],
  });
  assert.equal(prepended.messages[1], history);
  assert.equal(prepended.messages[2], next.messages[1]);
  assert.equal(mergeState(prepended, { revision: 2, messages: [] }), prepended);
});

test("burst deltas publish the latest complete snapshot on a timer without needing animation frames", (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  const published: StudioState[] = [];
  const updates = new DesktopUpdates(state(), (value) => published.push(value));
  updates.push({ revision: 1, messages: [message("First second")] });
  updates.push({ revision: 2, messages: [message("First second third")] });
  assert.equal(published.length, 0);
  t.mock.timers.tick(STREAM_RENDER_MS);
  assert.equal(published.length, 1);
  assert.equal(published[0].revision, 2);
  assert.equal(published[0].messages[0].content, "First second third");
  updates.push({ revision: 3, messages: [message("Last token")] });
  updates.flush();
  assert.equal(published.at(-1)!.messages[0].content, "Last token");
  t.mock.timers.tick(STREAM_RENDER_MS);
  assert.equal(published.length, 2);
});

test("tools and final snapshots flush queued prose in order and never leave delayed terminal updates", (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  const published: StudioState[] = [];
  const updates = new DesktopUpdates(state(), (value) => published.push(value));
  updates.push({ revision: 1, messages: [message("Before tool")] });
  const toolMessage = message("Before tool");
  toolMessage.items!.push({
    id: "read",
    type: "tool",
    name: "read",
    input: "file.md",
    output: "",
    status: "running",
    actions: [],
  });
  updates.push({ revision: 2, messages: [toolMessage] });
  assert.equal(published.length, 1);
  assert.deepEqual(
    published[0].messages[0].items!.map((item) => item.id),
    ["intro", "answer", "read"],
  );
  updates.push({ revision: 3, messages: [message("Corrected final", true)] });
  assert.equal(published.length, 2);
  assert.equal(published[1].messages[0].content, "Corrected final");
  assert.equal(published[1].messages[0].streaming, false);
  t.mock.timers.tick(STREAM_RENDER_MS * 2);
  assert.equal(published.length, 2);
});

test("context changes, connection failures and disposal cannot publish a queued old view", (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  const published: StudioState[] = [];
  const updates = new DesktopUpdates(state(), (value) => published.push(value));
  updates.push({ revision: 1, messages: [message("Queued")] });
  updates.push({
    revision: 2,
    selectedAgentName: "monitor",
    sessionId: "other",
    messages: [message("Other")],
  });
  assert.equal(published[0].selectedAgentName, "monitor");
  updates.push({ revision: 3, messages: [message("Other delta")] });
  updates.push({ revision: 4, connection: "reconnecting" });
  assert.equal(published[1].messages[0].content, "Other delta");
  assert.equal(published[1].connection, "reconnecting");
  updates.push({ revision: 5, messages: [message("Disposed")] });
  updates.close();
  t.mock.timers.tick(STREAM_RENDER_MS * 2);
  updates.flush();
  assert.equal(published.length, 2);
});
