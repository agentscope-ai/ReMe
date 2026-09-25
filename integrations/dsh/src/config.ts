import z from "@deepseek-ai/schemastery";

import { nextDailyRun, validTimezone } from "./scheduling.js";
import type { ReMeConfig, ReMeConfigInput } from "./types.js";

export const Config = z.object({
  endpoint: z
    .string()
    .description("ReMe HTTP service URL")
    .default(
      process.env.REME_URL ||
        `http://${process.env.REME_HOST || "127.0.0.1"}:${
          process.env.REME_PORT || "2333"
        }`,
    )
    .volatile(),
  requestTimeoutMs: z.natural().min(1000).max(120000).default(10000).volatile(),
  backgroundTimeoutMs: z
    .natural()
    .min(1000)
    .max(3600000)
    .default(3600000)
    .volatile(),
  shutdownTimeoutMs: z.natural().min(100).max(60000).default(5000).volatile(),
  autoMemoryEnabled: z.boolean().default(true).volatile(),
  autoMemoryInterval: z.natural().min(1).max(1000).default(5).volatile(),
  autoDreamEnabled: z.boolean().default(true).volatile(),
  dreamCron: z
    .string()
    .description("Daily cron in the workspace timezone")
    .default(process.env.REME_DSH_DREAM_CRON || "0 23 * * *")
    .volatile(),
  dreamHint: z.string().default("").volatile(),
  dreamIntervalMs: z.natural().max(2147483647).default(0),
  rootAgentsOnly: z.boolean().default(true).volatile(),
  language: z.union(["en", "zh"]).default("en").volatile(),
  searchLimit: z.natural().min(1).max(50).default(5).volatile(),
  timezone: z
    .string()
    .default("Asia/Shanghai")
    .description("IANA timezone matching the ReMe workspace")
    .volatile(),
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
  input = Object.fromEntries(
    Object.entries(input).map(([key, value]) => [
      key,
      value !== null &&
      typeof value === "object" &&
      "get" in value &&
      typeof value.get === "function"
        ? value.get()
        : value,
    ]),
  ) as ReMeConfigInput;
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
