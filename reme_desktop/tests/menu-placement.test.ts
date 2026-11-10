import assert from "node:assert/strict";
import test from "node:test";
import { companionPlacement } from "../src/main/placement.ts";

test("pet menus stay inside the work area without moving the character at screen edges", () => {
  for (const area of [
    { x: 0, y: 24, width: 1280, height: 776 },
    { x: -1024, y: -400, width: 1024, height: 768 },
  ]) {
    for (const scale of [0.75, 1, 1.5]) {
      for (const anchor of [
        { x: area.x, y: area.y },
        { x: area.x + area.width, y: area.y + area.height },
      ]) {
        const before = companionPlacement(anchor, scale, null, area);
        const menu = companionPlacement(anchor, scale, "menu", area);
        const position = (
          placement: ReturnType<typeof companionPlacement>,
        ) => ({
          x: placement.bounds.x + placement.layout.body.left,
          y: placement.bounds.y + placement.layout.body.top,
        });
        assert.deepEqual(position(menu), position(before));
        assert.ok(menu.bounds.x >= area.x && menu.bounds.y >= area.y);
        assert.ok(menu.bounds.x + menu.bounds.width <= area.x + area.width);
        assert.ok(menu.bounds.y + menu.bounds.height <= area.y + area.height);
        assert.ok(
          menu.layout.bubble.width >= 200 && menu.layout.bubble.height >= 160,
        );
      }
    }
  }
});
