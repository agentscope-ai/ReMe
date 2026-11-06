import assert from "node:assert/strict";
import test from "node:test";
import { SessionModel } from "../src/main/session-model.ts";
import { displayMessages } from "../src/shared/messages.ts";
import type {
  DesktopState,
  SessionMessage,
  SessionSnapshot,
} from "../src/shared/protocol.ts";

const message = (
  id: string,
  content = "提醒",
  completed = true,
): SessionMessage => ({
  id,
  content,
  role: "assistant",
  source: "proactive",
  completed,
  chunks: [],
});
const snapshot = (
  id: string,
  messages: SessionMessage[] = [],
): SessionSnapshot => ({
  session_id: id,
  state: { session_id: id, status: "idle", active_flags: [] },
  messages,
});

test("replayed upserts replace by ID and never concatenate complete snapshots", () => {
  const model = new SessionModel();
  model.apply({
    type: "snapshot",
    activities: {},
    cursor: "epoch:1",
    snapshot: snapshot("s", [message("a", "原文")]),
  });
  model.apply({
    type: "message_upsert",
    session_id: "s",
    cursor: "epoch:2",
    message: message("a", "最终原文"),
  });
  model.apply({
    type: "message_upsert",
    session_id: "s",
    cursor: "epoch:2",
    message: message("a", "最终原文"),
  });
  assert.deepEqual(
    model.messages.map((entry) => entry.content),
    ["最终原文"],
  );
  model.apply({ type: "heartbeat", cursor: "never-save-this" });
  assert.equal(model.cursor, "epoch:2");
});

test("reset and error snapshots preserve history until a successful replacement", () => {
  const model = new SessionModel();
  model.apply({
    type: "snapshot",
    activities: {},
    cursor: "old:1",
    snapshot: snapshot("s", [message("a")]),
  });
  model.apply({ type: "reset", reason: "cursor_expired" });
  assert.equal(model.cursor, undefined);
  model.apply({
    type: "snapshot",
    activities: {},
    session_id: "s",
    error: "upstream offline",
  });
  model.apply({
    type: "message_upsert",
    session_id: "s",
    cursor: "new:3",
    message: message("bad"),
  });
  assert.deepEqual(
    model.messages.map((entry) => entry.id),
    ["a"],
  );
  assert.equal(model.cursor, undefined);
  model.apply({
    type: "snapshot",
    activities: {},
    cursor: "new:4",
    snapshot: snapshot("s", [message("a"), message("missed")]),
  });
  assert.equal(model.waitingForSnapshot, false);
  assert.equal(model.cursor, "new:4");
  assert.equal(model.messages.length, 2);
});

test("session switches discard previous history and malformed upserts cannot advance cursor", () => {
  const model = new SessionModel();
  model.apply({
    type: "snapshot",
    activities: {},
    cursor: "e:1",
    snapshot: snapshot("old", [message("old")]),
  });
  model.apply({ type: "session_changed", session_id: "new", cursor: "e:2" });
  assert.deepEqual(model.messages, []);
  assert.throws(() =>
    model.apply({
      type: "message_upsert",
      session_id: "new",
      cursor: "e:3",
      message: { id: "bad" },
    }),
  );
  assert.equal(model.cursor, "e:2");
  assert.throws(() =>
    model.apply({
      type: "status_changed",
      activities: {},
      session_id: "old",
      cursor: "e:4",
      status: "idle",
    }),
  );
});

test("native turn IDs reconcile streamed answers without duplicating prompts", () => {
  const user = {
    ...message("user", "你好"),
    role: "user" as const,
    source: "chat" as const,
    turn_id: "t",
  };
  const assistant = {
    ...message("assistant", "最终快照"),
    source: "chat" as const,
    turn_id: "t",
  };
  const state = {
    sessionId: "s",
    messages: [user, assistant],
    pending: {
      id: "local",
      query: "你好",
      text: "过时的增量",
      sessionId: "s",
      turnId: "t",
      streaming: true,
    },
  } as DesktopState;
  assert.deepEqual(
    displayMessages(state).map((entry) => entry.content),
    ["你好", "最终快照"],
  );
  state.pending!.turnId = "unaccepted-turn";
  assert.equal(displayMessages(state).length, 4);
});

test("invalid snapshots and statuses leave accepted history and cursor intact", () => {
  const model = new SessionModel();
  model.apply({
    type: "snapshot",
    activities: {},
    cursor: "accepted:1",
    snapshot: snapshot("s", [message("a")]),
  });
  for (const bad of [
    { ...snapshot("different"), state: undefined },
    {
      ...snapshot("different"),
      state: { session_id: "other", status: "idle" },
    },
    {
      ...snapshot("different"),
      state: { session_id: "different", status: "unknown" },
    },
    snapshot("different", [message("duplicate"), message("duplicate")]),
  ]) {
    assert.throws(() =>
      model.apply({
        type: "snapshot",
        activities: {},
        cursor: "invalid:2",
        snapshot: bad,
      }),
    );
    assert.equal(model.sessionId, "s");
    assert.equal(model.cursor, "accepted:1");
    assert.deepEqual(
      model.messages.map((entry) => entry.id),
      ["a"],
    );
  }
  assert.throws(() =>
    model.apply({
      type: "status_changed",
      activities: {},
      session_id: "s",
      cursor: "invalid:3",
      status: "unknown",
    }),
  );
  assert.equal(model.status, "idle");
  assert.equal(model.cursor, "accepted:1");
});

test("activity updates are scoped to the selected session and require the new contract", () => {
  const model = new SessionModel();
  const activity = {
    session_id: "s",
    turn_id: "t",
    source: "chat",
    phase: "answering",
  };
  model.apply({
    type: "snapshot",
    session_id: "s",
    snapshot: snapshot("s"),
    activities: { chat: activity },
    cursor: "e:1",
  });
  assert.equal(model.activities.chat?.phase, "answering");
  for (const activities of [
    undefined,
    [],
    { chat: { ...activity, session_id: "other" } },
    { chat: { ...activity, source: "proactive" } },
    { chat: { ...activity, phase: "unknown" } },
  ]) {
    assert.throws(() =>
      model.apply({
        type: "status_changed",
        session_id: "s",
        activities,
        status: "active",
        cursor: "e:2",
      }),
    );
    assert.equal(model.cursor, "e:1");
    assert.equal(model.activities.chat?.phase, "answering");
  }
  model.apply({ type: "session_changed", session_id: "new", cursor: "e:3" });
  assert.deepEqual(model.activities, {});
});
