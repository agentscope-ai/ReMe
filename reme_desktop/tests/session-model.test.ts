import assert from "node:assert/strict";
import test from "node:test";
import { SessionModel } from "../src/main/session-model.ts";
import { displayMessages } from "../src/shared/messages.ts";
import type {
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
  delivery: "idle",
  notification_eligible: true,
  chunks: [],
});

test("task activity is independent of chat selection and rejects malformed run identities atomically", () => {
  const model = new SessionModel();
  const task = {
    session_id: "native-task-session",
    turn_id: "native-task-turn",
    source: "task",
    phase: "working",
    run_id: "run",
    task_id: "research",
  };
  model.apply({
    type: "snapshot",
    activities: {},
    session_id: "s",
    snapshot: snapshot("s"),
    task_activities: { run: task },
    cursor: "e:1",
  });
  assert.equal(model.taskActivities.run.task_id, "research");
  for (const task_activities of [
    null,
    [],
    { other: task },
    { run: { ...task, source: "chat" } },
    { run: { ...task, phase: "unknown" } },
    { run: { ...task, task_id: null } },
  ]) {
    assert.throws(() =>
      model.apply({
        type: "status_changed",
        activities: {},
        session_id: "s",
        task_activities,
        cursor: "e:2",
      }),
    );
    assert.equal(model.cursor, "e:1");
    assert.equal(model.taskActivities.run.task_id, "research");
  }
  model.apply({ type: "session_changed", session_id: "new", cursor: "e:2" });
  assert.equal(model.taskActivities.run.task_id, "research");
  model.apply({
    type: "status_changed",
    activities: {},
    session_id: "new",
    task_activities: {},
    cursor: "e:3",
  });
  assert.deepEqual(model.taskActivities, {});
  model.clear();
  assert.deepEqual(model.taskActivities, {});
});
const snapshot = (
  id: string,
  messages: SessionMessage[] = [],
): SessionSnapshot => ({
  session_id: id,
  state: { session_id: id, status: "idle", active_flags: [] },
  messages,
});

test("reminder display uses the server window, rejects old backlog and keeps only the closing message", () => {
  const model = new SessionModel();
  const starts_at = "2026-11-08T09:00:00+08:00";
  const ends_at = "2026-11-08T23:00:00+08:00";
  const current = {
    ...message("closing"),
    created_at: "2026-11-08T22:30:00+08:00",
  };
  const other = { ...message("other"), created_at: current.created_at };
  const old = { ...message("old"), created_at: "2026-11-07T22:59:00+08:00" };
  const at = (clock: string) => Date.parse(`2026-11-08T${clock}:00+08:00`);
  model.notifications = { starts_at, ends_at, enabled: true };
  assert.equal(model.canNotify(current, at("08:59")), false);
  assert.equal(model.canNotify(current, at("09:00")), true);
  assert.equal(model.canNotify(current, at("22:59")), true);
  assert.equal(model.canNotify(current, at("23:00")), false);
  assert.equal(model.canNotify(old, at("22:59")), false);
  assert.equal(model.canNotify(message("unknown-time"), at("22:59")), false);
  model.notifications.closing_message_id = current.id;
  assert.equal(model.canNotify(current, at("22:59")), true);
  assert.equal(model.canNotify(other, at("22:59")), false);
  model.notifications = null;
  assert.equal(model.canNotify(current, at("22:59")), false);
});

test("invalid notification windows cannot advance accepted policy or cursor", () => {
  const model = new SessionModel();
  model.apply({
    type: "snapshot",
    activities: {},
    cursor: "epoch:1",
    session_id: "s",
    snapshot: snapshot("s"),
    notifications: {
      enabled: true,
      starts_at: "2026-11-08T09:00:00+08:00",
      ends_at: "2026-11-08T23:00:00+08:00",
    },
  });
  const policy = model.notifications;
  assert.throws(() =>
    model.apply({
      type: "status_changed",
      activities: {},
      cursor: "epoch:2",
      session_id: "s",
      notifications: { starts_at: "invalid", ends_at: "invalid" },
    }),
  );
  assert.equal(model.notifications, policy);
  assert.equal(model.cursor, "epoch:1");
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
  } as Parameters<typeof displayMessages>[0];
  assert.deepEqual(
    displayMessages(state, undefined, "zh").map((entry) => entry.content),
    ["你好", "最终快照"],
  );
  state.pending!.turnId = "unaccepted-turn";
  assert.equal(displayMessages(state, undefined, "zh").length, 4);
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

test("an immediate message bypasses ordinary windows only until its own expiry", () => {
  const model = new SessionModel();
  const now = Date.now();
  const urgent = {
    ...message("urgent"),
    delivery: "immediate" as const,
    expires_at: new Date(now + 1000).toISOString(),
  };
  assert.equal(model.canNotify(urgent, now), true);
  assert.equal(model.canNotify(urgent, now + 1000), false);
  assert.equal(model.canNotify({ ...urgent, expires_at: null }, now), true);
  for (const expires_at of ["invalid", ""]) {
    assert.equal(model.canNotify({ ...urgent, expires_at }, now), false);
  }
});
