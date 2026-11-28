import assert from "node:assert/strict";
import test from "node:test";
import { changedTaskSettings } from "../../client/src/tasks.ts";
import {
  changeSchedulePreset,
  scheduleAtTime,
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
      schedule: null,
      min_trigger_interval_seconds: 12.5,
      notify_windows: [
        { start: "09:00", end: "12:00" },
        { start: "14:00", end: "23:00" },
      ],
    },
  );
  const timed = toRuntimeSettings({
    ...draft,
    trigger: defaultTaskTrigger("time"),
  });
  assert.equal(timed.schedule, "0 8 * * *");
  assert.equal(timed.min_trigger_interval_seconds, 0);
});

test("a timed task round-trips an existing schedule and unrelated settings", () => {
  const baseline = {
    enabled: true,
    schedule: "05  09 * * 1-5",
    min_trigger_interval_seconds: 30,
    notify_windows: [{ start: "22:00", end: "06:00" }],
  };
  const draft = taskSettingsDraft(baseline, "time");
  assert.deepEqual(draft.trigger, {
    type: "time",
    schedule: { preset: "weekdays", schedule: baseline.schedule },
  });
  assert.deepEqual(parseScheduleTime(baseline.schedule), {
    preset: "weekdays",
    time: "09:05",
  });
  assert.deepEqual(toRuntimeSettings(draft, baseline), baseline);
  const updated = toRuntimeSettings({ ...draft, enabled: false }, baseline);
  assert.deepEqual(changedTaskSettings(baseline, updated), { enabled: false });
});

test("custom schedule text stays editable even when it matches a preset", () => {
  const hourly = { preset: "hourly", schedule: "0 * * * *" } as const;
  assert.deepEqual(changeSchedulePreset(hourly, "custom"), {
    preset: "custom",
    schedule: hourly.schedule,
  });
  assert.deepEqual(changeSchedulePreset(hourly, "weekdays"), {
    preset: "weekdays",
    schedule: "0 8 * * 1-5",
  });
  assert.equal(scheduleAtTime("weekdays", "09:05"), "5 9 * * 1-5");
  assert.equal(scheduleAtTime("daily", ""), "");
  assert.equal(parseScheduleTime("0 24 * * *"), undefined);
});

test("notification periods are isolated between drafts and edits only patch the list", () => {
  const first = taskSettingsDraft();
  const second = taskSettingsDraft();
  first.notify_windows[0].start = "10:00";
  assert.equal(second.notify_windows[0].start, "09:00");
  assert.equal(taskSettingsDraft().notify_windows[0].start, "09:00");
  const baseline = toRuntimeSettings(second);
  const periods = [...second.notify_windows, { start: "23:30", end: "01:00" }];
  assert.deepEqual(
    changedTaskSettings(
      baseline,
      toRuntimeSettings({ ...second, notify_windows: periods }),
    ),
    { notify_windows: periods },
  );
  assert.deepEqual(
    changedTaskSettings(
      baseline,
      toRuntimeSettings({ ...second, notify_windows: [] }),
    ),
    { notify_windows: [] },
  );
});
