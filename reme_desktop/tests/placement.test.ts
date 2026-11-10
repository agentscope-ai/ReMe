import assert from "node:assert/strict";
import test from "node:test";
import { companionPlacement } from "../src/main/placement.ts";

test("bubble appearance preserves the pet anchor on every edge and on negative-coordinate screens", () => {
  for (const area of [
    { x: 0, y: 0, width: 1440, height: 900 },
    { x: -1920, y: -1080, width: 1920, height: 1080 },
    { x: 1920, y: -200, width: 640, height: 480 },
  ]) {
    for (const scale of [0.7, 0.75, 1, 1.25, 1.5]) {
      for (const point of [
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1],
        [0.5, 0.5],
      ]) {
        const anchor = {
          x: area.x + area.width * point[0],
          y: area.y + area.height * point[1],
        };
        const bare = companionPlacement(anchor, scale, null, area);
        for (const kind of [null, "status", "message"] as const) {
          const placed = companionPlacement(bare.anchor, scale, kind, area);
          const { bounds, layout } = placed;
          assert.deepEqual(placed.anchor, bare.anchor);
          assert.equal(
            bounds.x + layout.body.left,
            bare.bounds.x + bare.layout.body.left,
          );
          assert.equal(
            bounds.y + layout.body.top,
            bare.bounds.y + bare.layout.body.top,
          );
          assert.ok(bounds.x >= area.x && bounds.y >= area.y);
          assert.ok(bounds.x + bounds.width <= area.x + area.width);
          assert.ok(bounds.y + bounds.height <= area.y + area.height);
          if (!kind) continue;
          const { body, bubble } = layout;
          assert.ok(
            bubble.left + bubble.width <= body.left ||
              bubble.left >= body.left + body.size ||
              bubble.top + bubble.height <= body.top ||
              bubble.top >= body.top + body.size,
            "bubble must not cover the character",
          );
          const extent =
            bubble.side === "top" || bubble.side === "bottom"
              ? bubble.width
              : bubble.height;
          assert.ok(bubble.tip >= 0 && bubble.tip <= extent);
        }
      }
    }
  }
});

test("position prefers above, flips below near the top, and uses the side when vertical space is tight", () => {
  const area = { x: 0, y: 0, width: 1440, height: 900 };
  const above = companionPlacement(
    { x: 720, y: 800 },
    1,
    "message",
    area,
  ).layout;
  assert.equal(above.bubble.side, "top");
  assert.equal(above.body.top - above.bubble.top - above.bubble.height, 12);
  assert.equal(
    above.bubble.left + above.bubble.tip,
    above.body.left + above.body.size / 2,
  );
  const below = companionPlacement(
    { x: 720, y: 196 },
    1,
    "message",
    area,
  ).layout;
  assert.equal(below.bubble.side, "bottom");
  assert.equal(below.bubble.top - below.body.top - below.body.size, 12);
  const side = companionPlacement({ x: 720, y: 340 }, 1, "message", {
    ...area,
    height: 500,
  }).layout;
  assert.equal(side.bubble.side, "right");
  assert.equal(side.bubble.left - side.body.left - side.body.size, 12);
  const left = companionPlacement({ x: 1000, y: 340 }, 1, "message", {
    ...area,
    height: 500,
  }).layout;
  assert.equal(left.bubble.side, "left");
  assert.equal(left.body.left - left.bubble.left - left.bubble.width, 12);
});

test("frames have bounded comfortable sizes and use a compact frame when the work area requires it", () => {
  const area = { x: 0, y: 0, width: 1440, height: 900 };
  const anchor = { x: 720, y: 800 };
  for (const scale of [0.7, 1, 1.5]) {
    const message = companionPlacement(anchor, scale, "message", area).layout
      .bubble;
    assert.ok(message.width >= 260 && message.width <= 320);
    assert.ok(message.height >= 180 && message.height <= 212);
    const status = companionPlacement(anchor, scale, "status", area).layout
      .bubble;
    assert.ok(status.width >= 180 && status.width <= 220);
    assert.equal(status.height, 38);
  }
  const compact = companionPlacement({ x: 240, y: 220 }, 1, "message", {
    x: 0,
    y: 0,
    width: 480,
    height: 420,
  });
  assert.equal(compact.layout.bubble.side, "bottom");
  assert.ok(compact.layout.bubble.width >= 240);
  assert.ok(
    compact.layout.bubble.height >= 160 && compact.layout.bubble.height < 198,
  );
});
