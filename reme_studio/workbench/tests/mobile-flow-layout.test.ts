import assert from "node:assert/strict";
import test from "node:test";
import { verticalConnection } from "../src/mobile/flow-connections.ts";

test("mobile handoffs cross only the gap between collapsed stage cards", () => {
  const route = verticalConnection(
    { left: 17, right: 373, top: 6, bottom: 114 },
    { left: 17, right: 373, top: 138, bottom: 246 },
  );
  assert.equal(route.path, "M 195 113 L 195 139");
  assert.deepEqual(route.chevrons, [
    "M 191 121 l 4 4 4 -4",
    "M 191 127 l 4 4 4 -4",
  ]);
});

test("expanding a stage moves its outgoing handoff below the content", () => {
  const route = verticalConnection(
    { left: 17, right: 373, top: 6, bottom: 434 },
    { left: 17, right: 373, top: 458, bottom: 566 },
  );
  assert.equal(route.path, "M 195 433 L 195 459");
});
