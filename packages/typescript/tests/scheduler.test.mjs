import assert from "node:assert/strict";
import test from "node:test";
import { nextDailyRun } from "../dist/dsh/scheduler.js";

test("computes today's or tomorrow's daily dream run", () => {
  const before = new Date("2026-08-19T14:30:00Z");
  const today = nextDailyRun("0 23 * * *", "Asia/Shanghai", before);
  assert.equal(today.toISOString(), "2026-08-19T15:00:00.000Z");

  const after = new Date("2026-08-19T15:30:00Z");
  const tomorrow = nextDailyRun("0 23 * * *", "Asia/Shanghai", after);
  assert.equal(tomorrow.toISOString(), "2026-08-20T15:00:00.000Z");
});

test("uses the configured IANA timezone instead of the process timezone", () => {
  const now = new Date("2026-08-19T12:00:00Z");
  assert.equal(
    nextDailyRun("0 23 * * *", "UTC", now).toISOString(),
    "2026-08-19T23:00:00.000Z",
  );
  assert.equal(
    nextDailyRun("0 23 * * *", "America/Los_Angeles", now).toISOString(),
    "2026-08-20T06:00:00.000Z",
  );
});

test("skips a nonexistent local time across daylight-saving changes", () => {
  const beforeSpringForward = new Date("2026-03-08T09:00:00Z");
  assert.equal(
    nextDailyRun(
      "30 2 * * *",
      "America/Los_Angeles",
      beforeSpringForward,
    ).toISOString(),
    "2026-03-09T09:30:00.000Z",
  );
});

test("runs only once on a local date when daylight saving time repeats", () => {
  const beforeFirstOccurrence = new Date("2026-11-01T07:00:00Z");
  assert.equal(
    nextDailyRun(
      "30 1 * * *",
      "America/Los_Angeles",
      beforeFirstOccurrence,
    ).toISOString(),
    "2026-11-01T08:30:00.000Z",
  );

  const afterFirstOccurrence = new Date("2026-11-01T08:45:00Z");
  assert.equal(
    nextDailyRun(
      "30 1 * * *",
      "America/Los_Angeles",
      afterFirstOccurrence,
    ).toISOString(),
    "2026-11-02T09:30:00.000Z",
  );
});

test("rejects unsupported or invalid cron expressions", () => {
  assert.throws(() => nextDailyRun("*/5 * * * *", "UTC"), /daily form/);
  assert.throws(() => nextDailyRun("99 23 * * *", "UTC"), /invalid/);
});
