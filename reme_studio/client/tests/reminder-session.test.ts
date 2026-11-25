import assert from "node:assert/strict";
import test from "node:test";
import { SessionModel } from "../src/session-model.ts";
import { message, snapshot } from "./fixtures.ts";

const reminder = message({
  id: "reminder-1",
  source: "proactive",
  content: "实验已经完成。",
  notification_eligible: true,
  origin_session_id: "session",
});

test("reminders use the same authoritative history for snapshots and batched upserts", () => {
  const model = new SessionModel();
  model.apply(snapshot([reminder]));
  assert.equal(model.messages[0].source, "proactive");
  assert.equal(model.messages[0].notification_eligible, true);
  model.apply({
    ...snapshot([{ ...reminder, content: "已经核实，实验完成。" }]),
    type: "update",
  });
  assert.equal(model.messages.length, 1);
  assert.equal(model.messages[0].content, "已经核实，实验完成。");
  model.apply(snapshot([]));
  assert.deepEqual(model.messages, []);
});

test("switching conversations atomically replaces messages and rejects stale updates", () => {
  const model = new SessionModel();
  model.apply(snapshot([reminder]));
  model.apply(snapshot([], "chat-2"));
  assert.equal(model.sessionId, "chat-2");
  assert.deepEqual(model.messages, []);
  assert.throws(() => model.apply({ ...snapshot([reminder]), type: "update" }));
  assert.equal(model.sessionId, "chat-2");
});

test("malformed message blocks and activities cannot overwrite a valid session", () => {
  const model = new SessionModel();
  model.apply(snapshot([reminder]));
  const original = model.messages;
  assert.throws(() =>
    model.apply(snapshot([{ ...reminder, role: "system" as never }])),
  );
  assert.throws(() =>
    model.apply(
      snapshot([{ ...reminder, items: [{ type: "unknown" } as never] }]),
    ),
  );
  assert.throws(() =>
    model.apply({
      ...snapshot([]),
      activities: {
        Remy: { session_id: "session", turn_id: "turn", phase: "working" },
      },
    }),
  );
  assert.equal(model.messages, original);
  assert.equal(model.sessionId, "session");
});
