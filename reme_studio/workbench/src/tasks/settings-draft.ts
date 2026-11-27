import type { ProactiveRuntimeSettings } from "../../../client/src/tasks.ts";

const schedulePresets = {
  daily: "0 8 * * *",
  weekdays: "0 8 * * 1-5",
  hourly: "0 * * * *",
};

type ClockPreset = "daily" | "weekdays";
export type SchedulePreset = keyof typeof schedulePresets | "custom";

export type ScheduleDraft = {
  preset: SchedulePreset;
  schedule: string;
};

export type TaskTrigger =
  | { type: "event"; minIntervalSeconds: number }
  | { type: "time"; schedule: ScheduleDraft };

export type TaskSettingsDraft = Pick<
  ProactiveRuntimeSettings,
  "enabled" | "notify_window"
> & { trigger: TaskTrigger };

const defaultSettings: ProactiveRuntimeSettings = {
  enabled: true,
  schedule: null,
  min_trigger_interval_seconds: 0,
  notify_window: null,
};

export function parseScheduleTime(
  schedule: string,
): { preset: ClockPreset; time: string } | undefined {
  const match = /^([0-5]?\d)\s+([01]?\d|2[0-3])\s+\*\s+\*\s+(\*|1-5)$/.exec(
    schedule.trim(),
  );
  if (!match) return;
  const [, minute, hour, days] = match;
  return {
    preset: days === "*" ? "daily" : "weekdays",
    time: `${hour.padStart(2, "0")}:${minute.padStart(2, "0")}`,
  };
}

export function changeSchedulePreset(
  schedule: ScheduleDraft,
  preset: SchedulePreset,
): ScheduleDraft {
  return {
    preset,
    schedule: preset === "custom" ? schedule.schedule : schedulePresets[preset],
  };
}

export function scheduleAtTime(preset: ClockPreset, time: string): string {
  if (!time) return "";
  const [hour, minute] = time.split(":");
  return `${Number(minute)} ${Number(hour)} * * ${preset === "daily" ? "*" : "1-5"}`;
}

export function defaultTaskTrigger(type: TaskTrigger["type"]): TaskTrigger {
  return type === "event"
    ? { type, minIntervalSeconds: defaultSettings.min_trigger_interval_seconds }
    : { type, schedule: { preset: "daily", schedule: schedulePresets.daily } };
}

export function taskSettingsDraft(
  settings: ProactiveRuntimeSettings = defaultSettings,
  agentType: "event" | "time" = "event",
): TaskSettingsDraft {
  const { schedule, min_trigger_interval_seconds, ...shared } = settings;
  if (agentType === "event")
    return {
      ...shared,
      trigger: {
        type: "event",
        minIntervalSeconds: min_trigger_interval_seconds,
      },
    };
  if (schedule === null)
    throw new Error("Time-triggered tasks require a schedule");
  const preset =
    schedule.trim() === schedulePresets.hourly
      ? "hourly"
      : (parseScheduleTime(schedule)?.preset ?? "custom");
  return {
    ...shared,
    trigger: { type: "time", schedule: { preset, schedule } },
  };
}

export function toRuntimeSettings(
  { trigger, ...shared }: TaskSettingsDraft,
  baseline: ProactiveRuntimeSettings = defaultSettings,
): ProactiveRuntimeSettings {
  return {
    ...shared,
    schedule: trigger.type === "time" ? trigger.schedule.schedule : null,
    min_trigger_interval_seconds:
      trigger.type === "event"
        ? trigger.minIntervalSeconds
        : baseline.min_trigger_interval_seconds,
  };
}
