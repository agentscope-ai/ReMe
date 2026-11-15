import assert from "node:assert/strict";
import test from "node:test";
import { SessionModel } from "../src/main/session-model.ts";
import type { AgentEvent, SessionMessage } from "../src/shared/protocol.ts";

const reminder: SessionMessage = {
  id: "reminder-1",
  role: "assistant",
  source: "proactive",
  content: "实验已经完成。",
  completed: true,
  delivery: "idle",
  notification_eligible: true,
  origin_session_id: "chat-1",
  chunks: [],
};

function snapshot(
  messages: SessionMessage[],
  sessionId = "chat-1",
): AgentEvent {
  return {
    type: "snapshot",
    session_id: sessionId,
    activities: {},
    snapshot: {
      session_id: sessionId,
      state: { session_id: sessionId, status: "idle", active_flags: [] },
      messages,
    },
  };
}

test("reminders use the same authoritative history for snapshots and upserts", () => {
  const model = new SessionModel();
  model.apply(snapshot([reminder]));
  assert.equal(model.messages[0].source, "proactive");
  assert.equal(model.canNotify(model.messages[0], Date.now()), true);
  model.apply({
    type: "message_upsert",
    session_id: "chat-1",
    activities: {},
    message: { ...reminder, content: "已经核实，实验完成。" },
  });
  assert.equal(model.messages.length, 1);
  assert.equal(model.messages[0].content, "已经核实，实验完成。");
  model.apply(snapshot([]));
  assert.deepEqual(model.messages, []);
});

test("switching conversations clears reminder messages from the old session", () => {
  const model = new SessionModel();
  model.apply(snapshot([reminder]));
  model.apply(snapshot([], "chat-2"));
  assert.equal(model.sessionId, "chat-2");
  assert.deepEqual(model.messages, []);
});

test("explicit immediate reminders remain subject to message eligibility and expiry", () => {
  const model = new SessionModel();
  const now = Date.now();
  assert.equal(
    model.canNotify({ ...reminder, delivery: "immediate" }, now),
    true,
  );
  assert.equal(
    model.canNotify({ ...reminder, notification_eligible: false }, now),
    false,
  );
  assert.equal(
    model.canNotify(
      { ...reminder, expires_at: new Date(now).toISOString() },
      now,
    ),
    false,
  );
});
