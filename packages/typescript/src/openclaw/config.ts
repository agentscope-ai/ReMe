import type { ReMeClientConfig } from "../core/types.js";

export interface OpenClawReMeConfig extends ReMeClientConfig {
  autoCapture: boolean;
  autoRecall: boolean;
  recallLimit: number;
  recallMinScore: number;
  shutdownTimeoutMs: number;
}

const DEFAULT_CONFIG: Readonly<OpenClawReMeConfig> = Object.freeze({
  endpoint: "http://127.0.0.1:2333",
  requestTimeoutMs: 5000,
  backgroundTimeoutMs: 3600000,
  shutdownTimeoutMs: 5000,
  autoCapture: true,
  autoRecall: true,
  recallLimit: 5,
  recallMinScore: 0,
});

/** JSON Schema mirrored in openclaw.plugin.json for runtime use and tests. */
export const OPENCLAW_CONFIG_SCHEMA = {
  type: "object",
  additionalProperties: false,
  properties: {
    endpoint: { type: "string" },
    requestTimeoutMs: { type: "integer", minimum: 1000, maximum: 120000 },
    backgroundTimeoutMs: { type: "integer", minimum: 1000, maximum: 3600000 },
    shutdownTimeoutMs: { type: "integer", minimum: 100, maximum: 60000 },
    autoCapture: { type: "boolean" },
    autoRecall: { type: "boolean" },
    recallLimit: { type: "integer", minimum: 1, maximum: 50 },
    recallMinScore: { type: "number", minimum: 0 },
  },
} as const;

/** Resolve and validate OpenClaw's host-specific ReMe configuration. */
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
  const url = new URL(normalizedEndpoint);
  if (url.protocol !== "http:" && url.protocol !== "https:") {
    throw new TypeError("ReMe endpoint must be an absolute http(s) URL");
  }
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
    autoCapture:
      input.autoCapture === undefined
        ? DEFAULT_CONFIG.autoCapture
        : input.autoCapture !== false,
    autoRecall:
      input.autoRecall === undefined
        ? DEFAULT_CONFIG.autoRecall
        : input.autoRecall !== false,
    recallLimit: integer(input.recallLimit, 1, 50, DEFAULT_CONFIG.recallLimit),
    recallMinScore: Math.max(
      0,
      finite(input.recallMinScore, DEFAULT_CONFIG.recallMinScore),
    ),
  };
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
