import type { ReMeClientConfig } from "../core/types.js";

export interface ReMeConfigInput {
  endpoint?: string;
  requestTimeoutMs?: number;
  backgroundTimeoutMs?: number;
  shutdownTimeoutMs?: number;
  autoMemoryEnabled?: boolean;
  autoMemoryInterval?: number;
  autoDreamEnabled?: boolean;
  dreamCron?: string;
  dreamHint?: string;
  dreamIntervalMs?: number;
  rootAgentsOnly?: boolean;
  language?: "en" | "zh";
  searchLimit?: number;
  timezone?: string;
}

export interface ReMeConfig extends ReMeClientConfig {
  shutdownTimeoutMs: number;
  autoMemoryEnabled: boolean;
  autoMemoryInterval: number;
  autoDreamEnabled: boolean;
  dreamCron: string;
  dreamHint: string;
  dreamIntervalMs: number;
  rootAgentsOnly: boolean;
  language: "en" | "zh";
  searchLimit: number;
  timezone: string;
}

/** ReMe integration fields owned by the DSH user-settings document. */
export type ReMeSettings = Omit<ReMeConfig, "dreamIntervalMs">;

export interface SessionEvent {
  type: string;
  seq?: number;
  time?: number;
  data?: unknown;
}

export interface DshSession {
  id: string;
  header?: { origin?: string };
  events?: readonly SessionEvent[];
}
