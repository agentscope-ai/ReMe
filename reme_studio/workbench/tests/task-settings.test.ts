import assert from "node:assert/strict";
import test from "node:test";
import { changedTaskSettings } from "../../client/src/tasks.ts";
import {
  changeSchedulePreset,
  cronAtTime,
  defaultTaskTrigger,
  parseScheduleTime,
  taskSettingsDraft,
  toRuntimeSettings,
} from "../src/tasks/settings-draft.ts";

test("creating a task sends only the selected mode's settings", () => {
  const draft = taskSettingsDraft();
  assert.equal(draft.trigger.type, "event");
  assert.deepEqual(
    toRuntimeSettings({
      ...draft,
      trigger: { type: "event", minIntervalSeconds: 12.5 },
    }),
    {
      enabled: true,
      cron: null,
      min_trigger_interval_seconds: 12.5,
      notify_window: null,
    },
  );
  const timed = toRuntimeSettings({
    ...draft,
    trigger: defaultTaskTrigger("time"),
  });
  assert.equal(timed.cron, "0 8 * * *");
  assert.equal(timed.min_trigger_interval_seconds, 0);
});

test("a timed task round-trips an existing schedule and unrelated settings", () => {
  const baseline = {
    enabled: true,
    cron: "05  09 * * 1-5",
    min_trigger_interval_seconds: 30,
    notify_window: { start: "22:00", end: "06:00" },
  };
  const draft = taskSettingsDraft(baseline);
  assert.deepEqual(draft.trigger, {
    type: "time",
    schedule: { preset: "weekdays", cron: baseline.cron },
  });
  assert.deepEqual(parseScheduleTime(baseline.cron), {
    preset: "weekdays",
    time: "09:05",
  });
  assert.deepEqual(toRuntimeSettings(draft, baseline), baseline);
  const updated = toRuntimeSettings({ ...draft, enabled: false }, baseline);
  assert.deepEqual(changedTaskSettings(baseline, updated), { enabled: false });
});

test("custom cron text stays editable even when it matches a preset", () => {
  const hourly = { preset: "hourly", cron: "0 * * * *" } as const;
  assert.deepEqual(changeSchedulePreset(hourly, "custom"), {
    preset: "custom",
    cron: hourly.cron,
  });
  assert.deepEqual(changeSchedulePreset(hourly, "weekdays"), {
    preset: "weekdays",
    cron: "0 8 * * 1-5",
  });
  assert.equal(cronAtTime("weekdays", "09:05"), "5 9 * * 1-5");
  assert.equal(cronAtTime("daily", ""), "");
  assert.equal(parseScheduleTime("0 24 * * *"), undefined);
});
