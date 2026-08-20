import type {
  AutoMemoryOptions,
  DreamOptions,
  ReMeClientConfig,
  ReMeHealth,
  ReMeHealthResult,
  ReMeFileListingResult,
  ReMeFileResult,
  ReMeMemoryStatus,
  ReMeMessage,
  ReMeResult,
  ReMeStatusResult,
  SearchOptions,
} from "./types.js";

interface ReMeResponseBody {
  success?: boolean;
  answer?: unknown;
  metadata?: Record<string, unknown>;
  detail?: unknown;
}

export class ReMeClient {
  private readonly configSource: () => ReMeClientConfig;

  constructor(config: ReMeClientConfig | (() => ReMeClientConfig)) {
    this.configSource = typeof config === "function" ? config : () => config;
  }

  async search(
    query: string,
    options: SearchOptions = {},
  ): Promise<ReMeResult> {
    const config = this.configSource();
    return this.request(
      "search",
      {
        query,
        limit: options.limit,
        min_score: options.minScore,
      },
      config,
      config.requestTimeoutMs,
      options.signal,
    );
  }

  async autoMemory(
    messages: ReMeMessage[],
    sessionId: string,
    options: AutoMemoryOptions = {},
  ): Promise<ReMeResult> {
    const config = this.configSource();
    return this.request(
      "auto_memory",
      {
        messages,
        session_id: sessionId,
        memory_hint: options.memoryHint || "",
        date: options.date || "",
      },
      config,
      config.backgroundTimeoutMs,
      options.signal,
    );
  }

  async autoDream(options: DreamOptions = {}): Promise<ReMeResult> {
    const config = this.configSource();
    return this.request(
      "auto_dream",
      {
        date: options.date || "",
        hint: options.hint || "",
      },
      config,
      config.backgroundTimeoutMs,
      options.signal,
    );
  }

  async healthCheck(
    options: { signal?: AbortSignal } = {},
  ): Promise<ReMeHealthResult> {
    const config = this.configSource();
    const result = await this.request(
      "health_check",
      {},
      config,
      config.requestTimeoutMs,
      options.signal,
    );
    return {
      ...result,
      health: healthFrom(result.metadata?.health),
    };
  }

  async status(
    options: { signal?: AbortSignal } = {},
  ): Promise<ReMeStatusResult> {
    const config = this.configSource();
    const result = await this.request(
      "status",
      {},
      config,
      config.requestTimeoutMs,
      options.signal,
    );
    return {
      ...result,
      memory: memoryFrom(result.metadata?.status),
    };
  }

  async appConfig(options: { signal?: AbortSignal } = {}): Promise<ReMeResult> {
    const config = this.configSource();
    return this.request(
      "app_config",
      {},
      config,
      config.requestTimeoutMs,
      options.signal,
    );
  }

  async listFiles(
    path: string,
    options: { limit?: number; signal?: AbortSignal } = {},
  ): Promise<ReMeFileListingResult> {
    const config = this.configSource();
    const limit = options.limit ?? 5000;
    const result = await this.request(
      "list",
      {
        path,
        recursive: true,
        sort_by: "mtime",
        extensions: ["md", "markdown", "txt", "yaml", "yml"],
        limit,
      },
      config,
      config.requestTimeoutMs,
      options.signal,
    );
    const items = result.metadata?.items;
    const files = Array.isArray(items)
      ? items.filter((item): item is string => typeof item === "string")
      : [];
    return { ...result, files, limited: files.length >= limit };
  }

  async loadFile(
    path: string,
    options: { signal?: AbortSignal } = {},
  ): Promise<ReMeFileResult> {
    const config = this.configSource();
    const result = await this.request(
      "load",
      { path },
      config,
      config.requestTimeoutMs,
      options.signal,
    );
    return {
      ...result,
      content: result.ok ? String(result.answer ?? "") : undefined,
      path:
        typeof result.metadata?.path === "string"
          ? result.metadata.path
          : undefined,
      mtime:
        typeof result.metadata?.mtime === "string"
          ? result.metadata.mtime
          : undefined,
    };
  }

  private async request(
    job: string,
    payload: Record<string, unknown>,
    config: ReMeClientConfig,
    timeoutMs: number,
    externalSignal?: AbortSignal,
  ): Promise<ReMeResult> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    const signal = externalSignal
      ? AbortSignal.any([externalSignal, controller.signal])
      : controller.signal;
    try {
      const response = await fetch(`${config.endpoint}/${job}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(config.apiKey
            ? { Authorization: `Bearer ${config.apiKey}` }
            : {}),
        },
        body: JSON.stringify(payload),
        signal,
      });
      const body = (await response
        .json()
        .catch(() => ({}))) as ReMeResponseBody;
      const ok = response.ok && body.success !== false;
      return {
        ok,
        status: response.status,
        answer: body.answer ?? "",
        metadata: body.metadata ?? {},
        error: ok
          ? ""
          : String(body.answer || body.detail || `HTTP ${response.status}`),
      };
    } catch (error) {
      return {
        ok: false,
        status: 0,
        answer: "",
        metadata: {},
        error: error instanceof Error ? error.message : String(error),
      };
    } finally {
      clearTimeout(timer);
    }
  }
}

function healthFrom(value: unknown): ReMeHealth | undefined {
  if (
    !isRecord(value) ||
    typeof value.version !== "string" ||
    typeof value.healthy !== "boolean"
  )
    return undefined;
  if (!isRecord(value.components)) return undefined;
  return value as unknown as ReMeHealth;
}

function memoryFrom(value: unknown): ReMeMemoryStatus | undefined {
  if (!isRecord(value) || !isRecord(value.memory)) return undefined;
  const memory = value.memory;
  if (
    typeof memory.process_rss_bytes !== "number" ||
    typeof memory.process_rss !== "string"
  )
    return undefined;
  return memory as unknown as ReMeMemoryStatus;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
