import assert from "node:assert/strict";
import test from "node:test";
import { BubbleQueue } from "../src/main/bubbles.ts";
import { companionPlacement } from "../src/main/placement.ts";

test("bubble expiry pauses on hover, and stale actions cannot close a newer bubble", () => {
  const queue = new BubbleQueue(0);
  queue.offer({ id: "a", text: "短提醒" });
  const first = queue.tick(1000, false)!;
  assert.equal(queue.presented(first.id, first.presentationId, 1000), true);
  assert.equal(queue.presented(first.id, first.presentationId, 1001), false);
  queue.pause("a", true, 2000);
  assert.equal(queue.tick(12000, false)?.id, "a");
  queue.pause("a", false, 12000);
  assert.equal(queue.tick(18000, false)?.id, "a");
  assert.equal(queue.tick(19001, false), null);
  queue.offer({ id: "b", text: "另一条" });
  const next = queue.tick(20000, false)!;
  assert.equal(queue.presented(first.id, first.presentationId, 20000), false);
  assert.equal(queue.presented(next.id, next.presentationId, 20000), true);
  queue.dismiss("a");
  queue.pause("a", true, 21000);
  assert.equal(queue.active?.id, "b");
});

test("blocked reminders wait, backlog coalesces, snooze bypasses ordinary cooldown", () => {
  const superseded: string[] = [];
  const queue = new BubbleQueue(10 * 60_000, (id) => superseded.push(id));
  queue.offer({ id: "old", text: "旧提醒" });
  queue.offer({ id: "latest", text: "新提醒" });
  assert.equal(queue.tick(0, true), null);
  const latest = queue.tick(100, false)!;
  assert.equal(latest.id, "latest");
  queue.presented(latest.id, latest.presentationId, 100);
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
  const small = companionPlacement(anchor, 1, null, area);
  const expanded = companionPlacement(small.anchor, 1, "reminder", area);
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

test("interrupted reminders preserve their visible time without consuming another appearance", () => {
  const queue = new BubbleQueue(600_000);
  queue.offer({ id: "a", text: "message" });
  const bubble = queue.tick(1000, false)!;
  queue.presented(bubble.id, bubble.presentationId, 1000);
  assert.equal(queue.tick(2000, true), null);
  assert.equal(queue.tick(60_000, true), null);
  assert.equal(
    queue.tick(61_000, false)?.presentationId,
    bubble.presentationId,
  );
  assert.equal(
    queue.presented(bubble.id, bubble.presentationId, 61_000),
    false,
  );
  assert.equal(queue.tick(67_999, false)?.id, "a");
  assert.equal(queue.tick(68_001, false), null);
  queue.offer({ id: "b", text: "another" });
  assert.equal(queue.tick(69_000, false), null);
});

test("an unpainted bubble consumes no cooldown or hourly quota; old acknowledgements are rejected", () => {
  const queue = new BubbleQueue(600_000);
  queue.offer({ id: "a", text: "message" });
  const first = queue.tick(0, false)!;
  assert.equal(queue.tick(5001, false), null);
  const retry = queue.tick(35_002, false)!;
  assert.equal(retry.id, first.id);
  assert.notEqual(retry.presentationId, first.presentationId);
  assert.equal(queue.presented(first.id, first.presentationId, 35_002), false);
  assert.equal(queue.presented(retry.id, retry.presentationId, 35_002), true);
});

test("confirmed reminders share a three-per-hour budget", () => {
  const queue = new BubbleQueue(0);
  for (let index = 0; index < 3; index++) {
    queue.offer({ id: String(index), text: "reminder" });
    const bubble = queue.tick(index * 100, false)!;
    assert.ok(queue.presented(bubble.id, bubble.presentationId, index * 100));
    queue.dismiss(bubble.id);
  }
  queue.offer({ id: "fourth", text: "later" });
  assert.equal(queue.tick(1000, false), null);
  assert.equal(queue.tick(3_600_001, false)?.id, "fourth");
});

test("a native show acknowledgement still counts if foreground work preempts the queue", () => {
  const queue = new BubbleQueue(600_000);
  queue.offer({ id: "native", text: "reminder" });
  const notification = queue.tick(0, false)!;
  queue.tick(1000, true);
  assert.equal(
    queue.presented(notification.id, notification.presentationId, 2000),
    true,
  );
  queue.dismiss(notification.id);
  queue.offer({ id: "next", text: "another" });
  assert.equal(queue.tick(3000, false), null);
});
