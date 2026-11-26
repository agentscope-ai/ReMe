import type { ProactiveRuntimeSettings } from "../../../client/src/tasks.ts";

const cronPresets = {
  daily: "0 8 * * *",
  weekdays: "0 8 * * 1-5",
  hourly: "0 * * * *",
};

type ClockPreset = "daily" | "weekdays";
export type SchedulePreset = keyof typeof cronPresets | "custom";

export type ScheduleDraft = {
  preset: SchedulePreset;
  cron: string;
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
  cron: null,
  min_trigger_interval_seconds: 0,
  notify_window: null,
};

export function parseScheduleTime(
  cron: string,
): { preset: ClockPreset; time: string } | undefined {
  const match = /^([0-5]?\d)\s+([01]?\d|2[0-3])\s+\*\s+\*\s+(\*|1-5)$/.exec(
    cron.trim(),
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
    cron: preset === "custom" ? schedule.cron : cronPresets[preset],
  };
}

export function cronAtTime(preset: ClockPreset, time: string): string {
  if (!time) return "";
  const [hour, minute] = time.split(":");
  return `${Number(minute)} ${Number(hour)} * * ${preset === "daily" ? "*" : "1-5"}`;
}

export function defaultTaskTrigger(type: TaskTrigger["type"]): TaskTrigger {
  return type === "event"
    ? { type, minIntervalSeconds: defaultSettings.min_trigger_interval_seconds }
    : { type, schedule: { preset: "daily", cron: cronPresets.daily } };
}

export function taskSettingsDraft(
  settings: ProactiveRuntimeSettings = defaultSettings,
): TaskSettingsDraft {
  const { cron, min_trigger_interval_seconds, ...shared } = settings;
  if (cron === null)
    return {
      ...shared,
      trigger: {
        type: "event",
        minIntervalSeconds: min_trigger_interval_seconds,
      },
    };
  const preset =
    cron.trim() === cronPresets.hourly
      ? "hourly"
      : (parseScheduleTime(cron)?.preset ?? "custom");
  return { ...shared, trigger: { type: "time", schedule: { preset, cron } } };
}

export function toRuntimeSettings(
  { trigger, ...shared }: TaskSettingsDraft,
  baseline: ProactiveRuntimeSettings = defaultSettings,
): ProactiveRuntimeSettings {
  return {
    ...shared,
    cron: trigger.type === "time" ? trigger.schedule.cron : null,
    min_trigger_interval_seconds:
      trigger.type === "event"
        ? trigger.minIntervalSeconds
        : baseline.min_trigger_interval_seconds,
  };
}
