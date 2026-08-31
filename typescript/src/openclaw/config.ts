import { nextDailyRun, validTimezone } from "../core/scheduling.js";
import type { ReMeClientConfig } from "../core/types.js";

/** OpenClaw-owned controls layered over the shared ReMe HTTP client. */
export interface OpenClawReMeConfig extends ReMeClientConfig {
  shutdownTimeoutMs: number;
  autoMemoryEnabled: boolean;
  autoMemoryInterval: number;
  autoDreamEnabled: boolean;
  dreamCron: string;
  dreamHint: string;
  rootAgentsOnly: boolean;
  language: "en" | "zh";
  autoRecall: boolean;
  searchLimit: number;
  recallMinScore: number;
  timezone: string;
}

const DEFAULT_CONFIG: Readonly<OpenClawReMeConfig> = Object.freeze({
  endpoint: "http://127.0.0.1:2333",
  requestTimeoutMs: 10000,
  backgroundTimeoutMs: 3600000,
  shutdownTimeoutMs: 5000,
  autoMemoryEnabled: true,
  autoMemoryInterval: 5,
  autoDreamEnabled: true,
  dreamCron: "0 23 * * *",
  dreamHint: "",
  rootAgentsOnly: true,
  language: "en",
  autoRecall: true,
  searchLimit: 5,
  recallMinScore: 0,
  timezone: "Asia/Shanghai",
});

/** JSON Schema mirrored in openclaw.plugin.json for metadata-only discovery. */
export const OPENCLAW_CONFIG_SCHEMA = {
  type: "object",
  additionalProperties: false,
  properties: {
    endpoint: { type: "string" },
    requestTimeoutMs: { type: "integer", minimum: 1000, maximum: 120000 },
    backgroundTimeoutMs: { type: "integer", minimum: 1000, maximum: 3600000 },
    shutdownTimeoutMs: { type: "integer", minimum: 100, maximum: 60000 },
    autoMemoryEnabled: { type: "boolean" },
    autoMemoryInterval: { type: "integer", minimum: 1, maximum: 1000 },
    autoDreamEnabled: { type: "boolean" },
    dreamCron: { type: "string" },
    dreamHint: { type: "string" },
    rootAgentsOnly: { type: "boolean" },
    language: { type: "string", enum: ["en", "zh"] },
    autoRecall: { type: "boolean" },
    searchLimit: { type: "integer", minimum: 1, maximum: 50 },
    recallMinScore: { type: "number", minimum: 0 },
    timezone: { type: "string" },
  },
} as const;

/** Labels shared by the runtime schema and the static OpenClaw manifest. */
export const OPENCLAW_CONFIG_UI_HINTS = {
  endpoint: {
    label: "ReMe endpoint",
    placeholder: "http://127.0.0.1:2333",
  },
  autoMemoryEnabled: { label: "Automatic memory capture" },
  autoMemoryInterval: {
    label: "Capture batch size",
    advanced: true,
  },
  autoDreamEnabled: { label: "Daily memory consolidation" },
  dreamCron: {
    label: "Auto Dream schedule",
    placeholder: "0 23 * * *",
    advanced: true,
  },
  dreamHint: { label: "Auto Dream hint", advanced: true },
  timezone: {
    label: "Workspace timezone",
    placeholder: "Asia/Shanghai",
  },
  rootAgentsOnly: { label: "Root agents only", advanced: true },
  language: { label: "Memory guidance language" },
  autoRecall: { label: "Automatic recall" },
  searchLimit: { label: "Search result limit", advanced: true },
  recallMinScore: { label: "Minimum recall score", advanced: true },
} as const;

/** Resolve strict host configuration without accepting superseded option names. */
export function resolveOpenClawConfig(
  input: Record<string, unknown> = {},
  env: Record<string, string | undefined> = process.env,
): OpenClawReMeConfig {
  const unknownKeys = Object.keys(input).filter(
    (key) => !(key in DEFAULT_CONFIG),
  );
  if (unknownKeys.length)
    throw new TypeError(
      `Unknown ReMe config option: ${unknownKeys.join(", ")}`,
    );
  const endpoint =
    stringValue(input.endpoint) ||
    env.REME_URL ||
    `http://${env.REME_HOST || "127.0.0.1"}:${env.REME_PORT || "2333"}`;
  const normalizedEndpoint = stripTrailingSlashes(endpoint);
  assertEndpoint(normalizedEndpoint);
  const timezone = stringValue(input.timezone) || DEFAULT_CONFIG.timezone;
  if (!validTimezone(timezone))
    throw new TypeError(`Invalid ReMe timezone: ${String(timezone)}`);
  const dreamCron = stringValue(input.dreamCron) || DEFAULT_CONFIG.dreamCron;
  nextDailyRun(dreamCron, timezone);
  return {
    endpoint: normalizedEndpoint,
    requestTimeoutMs: integer(
      input.requestTimeoutMs,
      1000,
      120000,
      DEFAULT_CONFIG.requestTimeoutMs,
    ),
    backgroundTimeoutMs: integer(
      input.backgroundTimeoutMs,
      1000,
      3600000,
      DEFAULT_CONFIG.backgroundTimeoutMs,
    ),
    shutdownTimeoutMs: integer(
      input.shutdownTimeoutMs,
      100,
      60000,
      DEFAULT_CONFIG.shutdownTimeoutMs,
    ),
    autoMemoryEnabled: bool(
      input.autoMemoryEnabled,
      DEFAULT_CONFIG.autoMemoryEnabled,
    ),
    autoMemoryInterval: integer(
      input.autoMemoryInterval,
      1,
      1000,
      DEFAULT_CONFIG.autoMemoryInterval,
    ),
    autoDreamEnabled: bool(
      input.autoDreamEnabled,
      DEFAULT_CONFIG.autoDreamEnabled,
    ),
    dreamCron,
    dreamHint:
      typeof input.dreamHint === "string"
        ? input.dreamHint.trim()
        : DEFAULT_CONFIG.dreamHint,
    rootAgentsOnly: bool(input.rootAgentsOnly, DEFAULT_CONFIG.rootAgentsOnly),
    language: input.language === "zh" ? "zh" : "en",
    autoRecall: bool(input.autoRecall, DEFAULT_CONFIG.autoRecall),
    searchLimit: integer(input.searchLimit, 1, 50, DEFAULT_CONFIG.searchLimit),
    recallMinScore: Math.max(
      0,
      finite(input.recallMinScore, DEFAULT_CONFIG.recallMinScore),
    ),
    timezone,
  };
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

function bool(value: unknown, fallback: boolean): boolean {
  return value === undefined ? fallback : value !== false;
}

function integer(
  value: unknown,
  minimum: number,
  maximum: number,
  fallback: number,
): number {
  return Math.max(
    minimum,
    Math.min(maximum, Math.round(finite(value, fallback))),
  );
}

function stripTrailingSlashes(value: string): string {
  let end = value.length;
  while (end > 0 && value.charCodeAt(end - 1) === 47) end -= 1;
  return value.slice(0, end);
}

function finite(value: unknown, fallback: number): number {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function stringValue(value: unknown): string {
  return typeof value === "string" ? value.trim() : "";
}
