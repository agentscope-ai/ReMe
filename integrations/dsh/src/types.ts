export interface ReMeConfigInput {
  endpoint?: string;
  apiKey?: string;
  requestTimeoutMs?: number;
  backgroundTimeoutMs?: number;
  autoMemoryEnabled?: boolean;
  autoMemoryInterval?: number;
  autoDreamEnabled?: boolean;
  dreamCron?: string;
  dreamHint?: string;
  dreamIntervalMs?: number;
  rootAgentsOnly?: boolean;
  language?: "en" | "zh";
  searchLimit?: number;
}

export interface ReMeConfig {
  endpoint: string;
  apiKey: string;
  requestTimeoutMs: number;
  backgroundTimeoutMs: number;
  autoMemoryEnabled: boolean;
  autoMemoryInterval: number;
  autoDreamEnabled: boolean;
  dreamCron: string;
  dreamHint: string;
  dreamIntervalMs: number;
  rootAgentsOnly: boolean;
  language: "en" | "zh";
  searchLimit: number;
}

export interface ReMeResult {
  ok: boolean;
  status?: number;
  answer?: unknown;
  metadata?: Record<string, unknown>;
  error?: string;
}

export interface ReMeMessage {
  id: string;
  name: "user" | "assistant";
  role: "user" | "assistant";
  content: Array<{ type: "text"; text: string }>;
  created_at?: string;
}

export interface SessionEvent {
  type: string;
  seq?: number;
  time?: number;
  data?: Record<string, unknown>;
}

export interface DshSession {
  id: string;
  header?: { origin?: string };
  events?: SessionEvent[];
}

export interface ReMeClientLike {
  search(query: string, options?: SearchOptions): Promise<ReMeResult>;
  autoMemory(messages: ReMeMessage[], sessionId: string, memoryHint?: string): Promise<ReMeResult>;
  autoDream(options?: DreamOptions): Promise<ReMeResult>;
}

export interface SearchOptions {
  limit?: number;
  minScore?: number;
}

export interface DreamOptions {
  date?: string;
  hint?: string;
}

export interface LoggerLike {
  debug?(message: string, data?: unknown): void;
  warn?(message: string, data?: unknown): void;
  log?(message: string, data?: unknown): void;
}
