import type { ReMeConfig, ReMeConfigInput } from "./types.js";

const DEFAULT_CONFIG: Readonly<ReMeConfig> = Object.freeze({
  endpoint: "http://127.0.0.1:2333",
  apiKey: "",
  requestTimeoutMs: 10000,
  backgroundTimeoutMs: 3600000,
  autoMemoryEnabled: true,
  autoMemoryInterval: 5,
  autoDreamEnabled: true,
  dreamCron: "0 23 * * *",
  dreamHint: "",
  dreamIntervalMs: 0,
  rootAgentsOnly: true,
  language: "en",
  searchLimit: 5,
});

export function resolveConfig(
  input: ReMeConfigInput = {},
  env: Record<string, string | undefined> = process.env,
): ReMeConfig {
  const host = env.REME_HOST || "127.0.0.1";
  const port = env.REME_PORT || "2333";
  const config: ReMeConfig = {
    ...DEFAULT_CONFIG,
    ...input,
    endpoint: input.endpoint || env.REME_URL || `http://${host}:${port}`,
    apiKey: input.apiKey || env.REME_API_KEY || "",
    dreamCron: input.dreamCron || env.REME_DSH_DREAM_CRON || DEFAULT_CONFIG.dreamCron,
  };

  config.endpoint = String(config.endpoint).replace(/\/+$/, "");
  config.requestTimeoutMs = integer(config.requestTimeoutMs, 1000, 120000, DEFAULT_CONFIG.requestTimeoutMs);
  config.backgroundTimeoutMs = integer(
    config.backgroundTimeoutMs,
    1000,
    3600000,
    DEFAULT_CONFIG.backgroundTimeoutMs,
  );
  config.autoMemoryInterval = integer(config.autoMemoryInterval, 1, 1000, DEFAULT_CONFIG.autoMemoryInterval);
  config.dreamIntervalMs = integer(config.dreamIntervalMs, 0, 2147483647, 0);
  config.searchLimit = integer(config.searchLimit, 1, 50, DEFAULT_CONFIG.searchLimit);
  config.autoMemoryEnabled = config.autoMemoryEnabled !== false;
  config.autoDreamEnabled = config.autoDreamEnabled !== false;
  config.rootAgentsOnly = config.rootAgentsOnly !== false;
  config.language = config.language === "zh" ? "zh" : "en";
  return config;
}

function integer(value: unknown, minimum: number, maximum: number, fallback: number): number {
  const number = Math.round(Number(value));
  if (!Number.isFinite(number)) return fallback;
  return Math.max(minimum, Math.min(maximum, number));
}
