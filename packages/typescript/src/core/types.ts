/** Connection settings shared by every TypeScript host adapter. */
export interface ReMeClientConfig {
  endpoint: string;
  apiKey: string;
  requestTimeoutMs: number;
  backgroundTimeoutMs: number;
}

/** Normalized response returned by the ReMe HTTP client. */
export interface ReMeResult {
  ok: boolean;
  status?: number;
  answer?: unknown;
  metadata?: Record<string, unknown>;
  error?: string;
}

/** Component health facts returned by ReMe's health-check job. */
export interface ReMeComponentHealth {
  is_started?: boolean;
  is_healthy?: boolean | null;
  model_name?: string | null;
  dimensions?: number | null;
  cache_size?: number;
  n_nodes?: number;
  n_edges?: number;
  n_virtual?: number;
  n_pending?: number;
  n_chunks?: number;
  n_chunks_with_embedding?: number;
  n_docs?: number | null;
  vocab_size?: number;
  memory?: string;
}

/** Structured health snapshot returned by the ReMe service. */
export interface ReMeHealth {
  version: string;
  healthy: boolean;
  components: Record<string, Record<string, ReMeComponentHealth>>;
}

/** One component's estimated owned memory. */
export interface ReMeComponentMemory {
  bytes: number;
  human: string;
}

/** Structured process and component memory status returned by ReMe. */
export interface ReMeMemoryStatus {
  components: Record<string, Record<string, ReMeComponentMemory>>;
  components_total_bytes: number;
  components_total: string;
  process_rss_bytes: number;
  process_rss: string;
}

/** Typed diagnostic result returned by the health-check job. */
export interface ReMeHealthResult extends ReMeResult {
  health?: ReMeHealth;
}

/** Typed diagnostic result returned by the status job. */
export interface ReMeStatusResult extends ReMeResult {
  memory?: ReMeMemoryStatus;
}

/** Workspace file listing returned by ReMe's read-only list job. */
export interface ReMeFileListingResult extends ReMeResult {
  files: string[];
  limited: boolean;
}

/** Complete text file returned by ReMe's read-only load job. */
export interface ReMeFileResult extends ReMeResult {
  content?: string;
  path?: string;
  mtime?: string;
}

/** Text message accepted by ReMe's automatic-memory job. */
export interface ReMeMessage {
  id: string;
  name: "user" | "assistant";
  role: "user" | "assistant";
  content: Array<{ type: "text"; text: string }>;
  created_at?: string;
}

/** Search request controls supported by the shared client. */
export interface SearchOptions {
  limit?: number;
  minScore?: number;
  signal?: AbortSignal;
}

/** Automatic-memory request controls supported by the shared client. */
export interface AutoMemoryOptions {
  date?: string;
  memoryHint?: string;
  signal?: AbortSignal;
}

/** Automatic-dream request controls supported by the shared client. */
export interface DreamOptions {
  date?: string;
  hint?: string;
  signal?: AbortSignal;
}

/** ReMe operations used by host adapters. */
export interface ReMeClientLike {
  search(query: string, options?: SearchOptions): Promise<ReMeResult>;
  autoMemory(
    messages: ReMeMessage[],
    sessionId: string,
    options?: AutoMemoryOptions,
  ): Promise<ReMeResult>;
  autoDream(options?: DreamOptions): Promise<ReMeResult>;
  healthCheck(options?: { signal?: AbortSignal }): Promise<ReMeHealthResult>;
  status(options?: { signal?: AbortSignal }): Promise<ReMeStatusResult>;
  appConfig(options?: { signal?: AbortSignal }): Promise<ReMeResult>;
  listFiles(
    path: string,
    options?: { limit?: number; signal?: AbortSignal },
  ): Promise<ReMeFileListingResult>;
  loadFile(
    path: string,
    options?: { signal?: AbortSignal },
  ): Promise<ReMeFileResult>;
}

/** Logger subset shared by the host runtimes. */
export interface LoggerLike {
  debug?(message: string, data?: unknown): void;
  info?(message: string, data?: unknown): void;
  warn?(message: string, data?: unknown): void;
  error?(message: string, data?: unknown): void;
  log?(message: string, data?: unknown): void;
}
