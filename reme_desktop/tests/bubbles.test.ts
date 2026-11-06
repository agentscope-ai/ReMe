import assert from "node:assert/strict";
import test from "node:test";
import { BubbleQueue } from "../src/main/bubbles.ts";
import { companionPlacement } from "../src/main/placement.ts";

test("bubble expiry pauses on hover, and stale actions cannot close a newer bubble", () => {
  const presented: string[] = [];
  const queue = new BubbleQueue((id) => presented.push(id), 0);
  queue.offer({ id: "a", text: "短提醒" });
  queue.tick(1000, false);
  queue.pause("a", true, 2000);
  assert.equal(queue.tick(12000, false)?.id, "a");
  queue.pause("a", false, 12000);
  assert.equal(queue.tick(18000, false)?.id, "a");
  assert.equal(queue.tick(19001, false), null);
  queue.offer({ id: "b", text: "另一条" });
  queue.tick(20000, false);
  queue.dismiss("a");
  queue.pause("a", true, 21000);
  assert.equal(queue.active?.id, "b");
  assert.deepEqual(presented, ["a", "b"]);
});

test("blocked reminders wait, backlog coalesces, snooze bypasses ordinary cooldown", () => {
  const superseded: string[] = [];
  const queue = new BubbleQueue(
    () => {},
    10 * 60_000,
    (id) => superseded.push(id),
  );
  queue.offer({ id: "old", text: "旧提醒" });
  queue.offer({ id: "latest", text: "新提醒" });
  assert.equal(queue.tick(0, true), null);
  assert.equal(queue.tick(100, false)?.id, "latest");
  assert.deepEqual(superseded, ["old"]);
  queue.dismiss("latest");
  queue.offer({ id: "another", text: "普通提醒" });
  assert.equal(queue.tick(200, false), null);
  queue.offer({ id: "snooze", text: "用户要求稍后提醒", snoozed: true });
  assert.equal(queue.tick(300, false)?.id, "snooze");
});

test("bubble expansion keeps the canvas anchor fixed on negative-coordinate displays", () => {
  const area = { x: -1920, y: -200, width: 1920, height: 1080 };
  const anchor = { x: -1870, y: -150 };
  const small = companionPlacement(anchor, 1, false, area);
  const expanded = companionPlacement(small.anchor, 1, true, area);
  assert.deepEqual(expanded.anchor, small.anchor);
  assert.equal(
    expanded.bounds.x + expanded.layout.body.left,
    small.bounds.x + small.layout.body.left,
  );
  assert.equal(
    expanded.bounds.y + expanded.layout.body.top,
    small.bounds.y + small.layout.body.top,
  );
  assert.ok(expanded.bounds.x >= area.x);
  assert.ok(expanded.bounds.y >= area.y);
  assert.ok(expanded.bounds.x + expanded.bounds.width <= area.x + area.width);
});
