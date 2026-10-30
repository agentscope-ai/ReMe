import type { SettingsFieldSpec } from "@deepseek-ai/dsh-client-ui-primitives";

import { validEndpoint } from "../endpoint.js";
import { validTimezone } from "../scheduling.js";

// Re-exported for callers that imported it from this module before it moved.
export { validTimezone };

/**
 * How each memory-settings field converts between its stored value and the text
 * a control shows, in display and reset order.
 *
 * DSH's `SettingsFormModel` stages these drafts, reports which ones are invalid
 * and writes them in one revision-fenced mutation; the specs below are the only
 * per-field code this plugin owns.
 */
export const SETTING_FIELDS = [
  "endpoint",
  "requestTimeoutMs",
  "backgroundTimeoutMs",
  "shutdownTimeoutMs",
  "autoMemoryRetryMs",
  "autoMemoryRetryMaxMs",
  "autoMemoryEnabled",
  "autoMemoryInterval",
  "autoDreamEnabled",
  "dreamCron",
  "dreamHint",
  "rootAgentsOnly",
  "language",
  "searchLimit",
  "timezone",
] as const;

export type SettingField = (typeof SETTING_FIELDS)[number];

/** A whole number inside the range the Host accepts; empty clears the field. */
function boundedInteger(
  field: SettingField,
  minimum: number,
  maximum: number,
): SettingsFieldSpec {
  return {
    field,
    format: (value) => (typeof value === "number" ? String(value) : ""),
    parse: (text) => {
      const trimmed = text.trim();
      if (trimmed === "") return { kind: "clear" };
      const value = Number(trimmed);
      if (!Number.isInteger(value) || value < minimum || value > maximum)
        return undefined;
      return { kind: "set", value };
    },
  };
}

/** A checkbox: the control shows and writes "true" or "false". */
function flag(field: SettingField): SettingsFieldSpec {
  return {
    field,
    format: (value) => (value === true ? "true" : "false"),
    parse: (text) => ({ kind: "set", value: text === "true" }),
  };
}

/** "HH:MM" in the control, a five-field cron expression in the document. */
function timeOfDay(field: SettingField): SettingsFieldSpec {
  return {
    field,
    format: (value) => (typeof value === "string" ? timeFromCron(value) : ""),
    parse: (text) => {
      const match = /^(\d{1,2}):(\d{2})$/.exec(text.trim());
      if (match === null) return undefined;
      const hour = Number(match[1]);
      const minute = Number(match[2]);
      if (hour > 23 || minute > 59) return undefined;
      return { kind: "set", value: `${minute} ${hour} * * *` };
    },
  };
}

/** "HH:MM" for a daily `m h * * *` schedule; anything else shows nothing. */
export function timeFromCron(cron: string): string {
  const match = /^(\d{1,2})\s+(\d{1,2})\s+\*\s+\*\s+\*$/.exec(cron.trim());
  if (match === null) return "";
  return `${String(Number(match[2])).padStart(2, "0")}:${String(
    Number(match[1]),
  ).padStart(2, "0")}`;
}

export const SETTINGS_SPECS: SettingsFieldSpec[] = [
  {
    // The endpoint loses its trailing slashes on the way in, the way the
    // client appends job names to it.
    field: "endpoint",
    format: (value) => (typeof value === "string" ? value : ""),
    parse: (text) => {
      const trimmed = text.trim().replace(/\/+$/, "");
      return validEndpoint(trimmed)
        ? { kind: "set", value: trimmed }
        : undefined;
    },
  },
  boundedInteger("requestTimeoutMs", 1000, 120000),
  boundedInteger("backgroundTimeoutMs", 1000, 3600000),
  boundedInteger("shutdownTimeoutMs", 100, 60000),
  boundedInteger("autoMemoryRetryMs", 1000, 60000),
  boundedInteger("autoMemoryRetryMaxMs", 1000, 3600000),
  flag("autoMemoryEnabled"),
  boundedInteger("autoMemoryInterval", 1, 1000),
  flag("autoDreamEnabled"),
  timeOfDay("dreamCron"),
  {
    field: "dreamHint",
    format: (value) => (typeof value === "string" ? value : ""),
    parse: (text) => ({ kind: "set", value: text }),
  },
  flag("rootAgentsOnly"),
  {
    field: "language",
    format: (value) => (value === "en" ? "en" : "zh"),
    parse: (text) =>
      text === "en" || text === "zh" ? { kind: "set", value: text } : undefined,
  },
  boundedInteger("searchLimit", 1, 50),
  {
    field: "timezone",
    format: (value) => (typeof value === "string" ? value : ""),
    parse: (text) =>
      validTimezone(text.trim())
        ? { kind: "set", value: text.trim() }
        : undefined,
  },
];
