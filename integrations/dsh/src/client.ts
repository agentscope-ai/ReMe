import type {
  DreamOptions,
  ReMeConfig,
  ReMeMessage,
  ReMeResult,
  SearchOptions,
} from "./types.js";

interface ReMeResponseBody {
  success?: boolean;
  answer?: unknown;
  metadata?: Record<string, unknown>;
  detail?: unknown;
}

export class ReMeClient {
  constructor(readonly config: ReMeConfig) {}

  async search(query: string, options: SearchOptions = {}): Promise<ReMeResult> {
    return this.request("search", {
      query,
      limit: options.limit,
      min_score: options.minScore,
    }, this.config.requestTimeoutMs);
  }

  async autoMemory(messages: ReMeMessage[], sessionId: string, memoryHint = ""): Promise<ReMeResult> {
    return this.request("auto_memory", {
      messages,
      session_id: sessionId,
      memory_hint: memoryHint,
    }, this.config.backgroundTimeoutMs);
  }

  async autoDream(options: DreamOptions = {}): Promise<ReMeResult> {
    return this.request("auto_dream", {
      date: options.date || "",
      hint: options.hint || "",
    }, this.config.backgroundTimeoutMs);
  }

  private async request(job: string, payload: Record<string, unknown>, timeoutMs: number): Promise<ReMeResult> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const response = await fetch(`${this.config.endpoint}/${job}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(this.config.apiKey ? { Authorization: `Bearer ${this.config.apiKey}` } : {}),
        },
        body: JSON.stringify(payload),
        signal: controller.signal,
      });
      const body = await response.json().catch(() => ({})) as ReMeResponseBody;
      const ok = response.ok && body.success !== false;
      return {
        ok,
        status: response.status,
        answer: body.answer ?? "",
        metadata: body.metadata ?? {},
        error: ok ? "" : String(body.answer || body.detail || `HTTP ${response.status}`),
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
