import z from "@deepseek-ai/schemastery";
import { settingsNamespace } from "@deepseek-ai/dsh-settings";

import { nextDailyRun } from "./scheduler.js";
import type { ReMeConfig, ReMeConfigInput, ReMeSettings } from "./types.js";

/** Durable DSH settings section owned by the ReMe integration. */
export const REME_SETTINGS_NAMESPACE = settingsNamespace("reme-memory");

export const Config = z.object({
  endpoint: z.string().description("ReMe HTTP service URL"),
  requestTimeoutMs: z.natural().min(1000).max(120000).default(10000),
  backgroundTimeoutMs: z.natural().min(1000).max(3600000).default(3600000),
  shutdownTimeoutMs: z.natural().min(100).max(60000).default(5000),
  autoMemoryEnabled: z.boolean().default(true),
  autoMemoryInterval: z.natural().min(1).max(1000).default(5),
  autoDreamEnabled: z.boolean().default(true),
  dreamCron: z.string().description("Daily cron in the workspace timezone"),
  dreamHint: z.string().default(""),
  dreamIntervalMs: z.natural().max(2147483647).default(0),
  rootAgentsOnly: z.boolean().default(true),
  language: z.union(["en", "zh"]).default("en"),
  searchLimit: z.natural().min(1).max(50).default(5),
  timezone: z
    .string()
    .default("Asia/Shanghai")
    .description("IANA timezone matching the ReMe workspace"),
});

/** User-editable subset of the DSH integration configuration. */
export const SettingsConfig: z<ReMeSettings> = z.object({
  endpoint: z.string().required().description("ReMe HTTP service URL"),
  requestTimeoutMs: z.natural().min(1000).max(120000).default(10000),
  backgroundTimeoutMs: z.natural().min(1000).max(3600000).default(3600000),
  shutdownTimeoutMs: z.natural().min(100).max(60000).default(5000),
  autoMemoryEnabled: z.boolean().default(true),
  autoMemoryInterval: z.natural().min(1).max(1000).default(5),
  autoDreamEnabled: z.boolean().default(true),
  dreamCron: z
    .string()
    .required()
    .description("Daily cron in the workspace timezone"),
  dreamHint: z.string().default(""),
  rootAgentsOnly: z.boolean().default(true),
  language: z.union(["en", "zh"]).default("en"),
  searchLimit: z.natural().min(1).max(50).default(5),
  timezone: z
    .string()
    .default("Asia/Shanghai")
    .description("IANA timezone matching the ReMe workspace"),
});

const DEFAULT_CONFIG: Readonly<ReMeConfig> = Object.freeze({
  endpoint: "http://127.0.0.1:2333",
  requestTimeoutMs: 10000,
  backgroundTimeoutMs: 3600000,
  shutdownTimeoutMs: 5000,
  autoMemoryEnabled: true,
  autoMemoryInterval: 5,
  autoDreamEnabled: true,
  dreamCron: "0 23 * * *",
  dreamHint: "",
  dreamIntervalMs: 0,
  rootAgentsOnly: true,
  language: "en",
  searchLimit: 5,
  timezone: "Asia/Shanghai",
});

export function resolveConfig(
  input: ReMeConfigInput = {},
  env: Record<string, string | undefined> = process.env,
): ReMeConfig {
  const unknownKeys = Object.keys(input).filter(
    (key) => !(key in DEFAULT_CONFIG),
  );
  if (unknownKeys.length)
    throw new TypeError(
      `Unknown ReMe config option: ${unknownKeys.join(", ")}`,
    );
  const host = env.REME_HOST || "127.0.0.1";
  const port = env.REME_PORT || "2333";
  const config: ReMeConfig = {
    ...DEFAULT_CONFIG,
    ...input,
    endpoint: input.endpoint || env.REME_URL || `http://${host}:${port}`,
    dreamCron:
      input.dreamCron || env.REME_DSH_DREAM_CRON || DEFAULT_CONFIG.dreamCron,
  };

  config.endpoint = stripTrailingSlashes(String(config.endpoint));
  assertEndpoint(config.endpoint);
  config.requestTimeoutMs = integer(
    config.requestTimeoutMs,
    1000,
    120000,
    DEFAULT_CONFIG.requestTimeoutMs,
  );
  config.backgroundTimeoutMs = integer(
    config.backgroundTimeoutMs,
    1000,
    3600000,
    DEFAULT_CONFIG.backgroundTimeoutMs,
  );
  config.shutdownTimeoutMs = integer(
    config.shutdownTimeoutMs,
    100,
    60000,
    DEFAULT_CONFIG.shutdownTimeoutMs,
  );
  config.autoMemoryInterval = integer(
    config.autoMemoryInterval,
    1,
    1000,
    DEFAULT_CONFIG.autoMemoryInterval,
  );
  config.dreamIntervalMs = integer(config.dreamIntervalMs, 0, 2147483647, 0);
  config.searchLimit = integer(
    config.searchLimit,
    1,
    50,
    DEFAULT_CONFIG.searchLimit,
  );
  config.autoMemoryEnabled = config.autoMemoryEnabled !== false;
  config.autoDreamEnabled = config.autoDreamEnabled !== false;
  config.rootAgentsOnly = config.rootAgentsOnly !== false;
  config.language = config.language === "zh" ? "zh" : "en";
  if (!validTimezone(config.timezone))
    throw new TypeError(`Invalid ReMe timezone: ${String(config.timezone)}`);
  nextDailyRun(config.dreamCron, config.timezone);
  return config;
}

/** Project the full plugin configuration into its user-editable settings section. */
export function settingsFrom(config: ReMeConfig): ReMeSettings {
  const { dreamIntervalMs: _dreamIntervalMs, ...settings } = config;
  return settings;
}

/** Layer current DSH user settings over fixed deployment-only values. */
export function mergeSettings(
  base: ReMeConfig,
  settings: ReMeSettings,
): ReMeConfig {
  return { ...base, ...settings };
}

/** Reject a settings section the integration cannot use. */
export function validateSettings(settings: ReMeSettings): void {
  assertEndpoint(settings.endpoint);
  if (!validTimezone(settings.timezone)) {
    throw new TypeError(`Invalid ReMe timezone: ${String(settings.timezone)}`);
  }
  nextDailyRun(settings.dreamCron, settings.timezone);
}

function assertEndpoint(value: string): void {
  let endpoint: URL;
  try {
    endpoint = new URL(value);
  } catch {
    throw new TypeError("ReMe endpoint must be an absolute http(s) URL");
  }
  if (endpoint.protocol !== "http:" && endpoint.protocol !== "https:") {
    throw new TypeError("ReMe endpoint must be an absolute http(s) URL");
  }
}

function stripTrailingSlashes(value: string): string {
  let end = value.length;
  while (end > 0 && value.charCodeAt(end - 1) === 47) end -= 1;
  return value.slice(0, end);
}

function integer(
  value: unknown,
  minimum: number,
  maximum: number,
  fallback: number,
): number {
  const number = Math.round(Number(value));
  if (!Number.isFinite(number)) return fallback;
  return Math.max(minimum, Math.min(maximum, number));
}

function validTimezone(value: unknown): value is string {
  if (typeof value !== "string" || !value.trim()) return false;
  try {
    new Intl.DateTimeFormat("en", { timeZone: value }).format(0);
    return true;
  } catch {
    return false;
  }
}
