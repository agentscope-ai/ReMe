import assert from "node:assert/strict";
import test from "node:test";
import { captureMessage, remeSessionId } from "./dist/messages.js";

test("captures direct DSH user and assistant messages with stable ids", () => {
  const user = captureMessage({
    type: "user/message",
    seq: 7,
    time: 1786681234567,
    data: {
      role: "user",
      content: [{ type: "text", text: "Remember the blue deployment." }],
      source: { kind: "user" },
    },
  }, "session-a");
  assert.equal(user.id, "dsh-fa57a52dbf08-7");
  assert.equal(user.role, "user");
  assert.equal(user.created_at, "2026-08-14T04:20:34.567Z");

  const assistant = captureMessage({
    type: "assistant/message",
    seq: 9,
    data: {
      message: {
        role: "assistant",
        content: [{ type: "text", text: "I will remember that." }],
        source: { kind: "model" },
      },
    },
  }, "session-a");
  assert.equal(assistant.id, "dsh-fa57a52dbf08-9");
  assert.equal(assistant.role, "assistant");
});

test("does not launder plugin context into memory", () => {
  assert.equal(captureMessage({
    type: "user/message",
    seq: 1,
    data: {
      role: "user",
      content: [{ type: "text", text: "recalled content" }],
      source: { kind: "plugin", plugin: "reme-memory" },
    },
  }, "session-a"), null);
});

test("maps arbitrary DSH ids to safe fixed-length ReMe ids", () => {
  assert.match(remeSessionId("unsafe/session id"), /^dsh-[a-f0-9]{24}$/);
  assert.equal(remeSessionId("unsafe/session id"), remeSessionId("unsafe/session id"));
});
