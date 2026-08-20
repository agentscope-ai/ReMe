import assert from "node:assert/strict";
import test from "node:test";
import { nextDailyRun } from "../dist/dsh/scheduler.js";

test("computes today's or tomorrow's daily dream run", () => {
  const before = new Date(2026, 7, 19, 22, 30, 0);
  const today = nextDailyRun("0 23 * * *", before);
  assert.equal(today.getDate(), 19);
  assert.equal(today.getHours(), 23);

  const after = new Date(2026, 7, 19, 23, 30, 0);
  const tomorrow = nextDailyRun("0 23 * * *", after);
  assert.equal(tomorrow.getDate(), 20);
  assert.equal(tomorrow.getHours(), 23);
});

test("rejects unsupported or invalid cron expressions", () => {
  assert.throws(() => nextDailyRun("*/5 * * * *"), /daily form/);
  assert.throws(() => nextDailyRun("99 23 * * *"), /invalid/);
});
