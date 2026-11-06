import assert from "node:assert/strict";
import test from "node:test";
import { NotificationPolicy } from "../src/main/notifications.ts";
import type { AgentActivity } from "../src/shared/protocol.ts";

const activity = (
  phase: AgentActivity["phase"],
  turn = "t",
): AgentActivity => ({ session_id: "s", turn_id: turn, source: "chat", phase });

test("only long foreground completions alert, once per phase, with no historical celebration", () => {
  const policy = new NotificationPolicy();
  assert.equal(policy.observe(activity("completed"), false, true, 0), null);
  assert.equal(policy.observe(activity("completed"), true, true, 100), null);
  policy.observe(activity("thinking", "long"), true, true, 1000);
  policy.observe(activity("working", "long"), true, true, 20_000);
  const notice = policy.observe(
    activity("completed", "long"),
    true,
    true,
    31_001,
  );
  assert.equal(notice?.title, "Remy 已完成回复");
  assert.equal(
    policy.observe(activity("completed", "long"), true, true, 32_000),
    null,
  );
  policy.observe(activity("thinking", "short"), true, true, 33_000);
  assert.equal(
    policy.observe(activity("completed", "short"), true, true, 34_000),
    null,
  );
});

test("quiet/focused/disabled outcomes are consumed, never queued for a later burst", () => {
  const policy = new NotificationPolicy();
  policy.observe(activity("thinking"), true, false, 0);
  assert.equal(policy.observe(activity("waiting"), true, false, 100), null);
  assert.equal(policy.observe(activity("waiting"), true, true, 200), null);
  assert.ok(policy.observe(activity("failed"), true, true, 300));
  policy.reset();
  assert.equal(policy.observe(activity("failed"), false, true, 400), null);
  assert.equal(policy.observe(activity("failed"), true, true, 500), null);
});

test("system notification captions use the current desktop language", () => {
  const policy = new NotificationPolicy();
  const notice = policy.observe(activity("waiting"), true, true, 0, "en");
  assert.equal(notice?.title, "Remy needs your attention");
  assert.equal(notice?.body, "Click to return to the conversation");
});
