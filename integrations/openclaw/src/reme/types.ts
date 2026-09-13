export interface ReMeClientConfig {
  endpoint: string;
  requestTimeoutMs: number;
  backgroundTimeoutMs: number;
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

export interface SearchOptions {
  limit?: number;
  minScore?: number;
  signal?: AbortSignal;
}

export interface AutoMemoryOptions {
  date?: string;
  memoryHint?: string;
  signal?: AbortSignal;
}

export interface DreamOptions {
  date?: string;
  hint?: string;
  signal?: AbortSignal;
}

export interface ReMeClientLike {
  search(query: string, options?: SearchOptions): Promise<ReMeResult>;
  autoMemory(
    messages: ReMeMessage[],
    sessionId: string,
    options?: AutoMemoryOptions,
  ): Promise<ReMeResult>;
  autoDream(options?: DreamOptions): Promise<ReMeResult>;
}

export interface LoggerLike {
  debug?(message: string, data?: unknown): void;
  info?(message: string, data?: unknown): void;
  warn?(message: string, data?: unknown): void;
  error?(message: string, data?: unknown): void;
  log?(message: string, data?: unknown): void;
}
