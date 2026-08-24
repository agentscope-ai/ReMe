import type { LoggerLike, ReMeClientLike } from "../core/types.js";
import type { OpenClawReMeConfig } from "./config.js";
import { captureLastTurn, openClawSessionId } from "./messages.js";

const MAX_PENDING_PROMPTS = 256;

/** Host context supplied to OpenClaw agent lifecycle hooks. */
export interface OpenClawAgentContext {
  agentId?: string;
  sessionId?: string;
  sessionKey?: string;
  trigger?: string;
}

/** Background automatic-memory writer owned by one OpenClaw plugin instance. */
export class OpenClawReMeRuntime {
  private writes = Promise.resolve();
  private controller = new AbortController();
  private readonly prompts = new Map<string, string>();

  constructor(
    readonly client: ReMeClientLike,
    readonly config: OpenClawReMeConfig,
    readonly logger: LoggerLike,
  ) {}

  rememberPrompt(prompt: string, context: OpenClawAgentContext): void {
    if (!this.config.autoCapture || !capturesTrigger(context.trigger)) return;
    const key = promptKey(context);
    const text = prompt.trim();
    if (!key || !text) return;
    this.prompts.delete(key);
    this.prompts.set(key, text);
    while (this.prompts.size > MAX_PENDING_PROMPTS) {
      const oldest = this.prompts.keys().next().value;
      if (oldest === undefined) break;
      this.prompts.delete(oldest);
    }
  }

  takePrompt(context: OpenClawAgentContext): string | undefined {
    const key = promptKey(context);
    if (!key) return undefined;
    const prompt = this.prompts.get(key);
    this.prompts.delete(key);
    return prompt;
  }

  capture(
    messages: unknown[],
    context: OpenClawAgentContext,
    prompt?: string,
  ): void {
    if (!this.config.autoCapture || !capturesTrigger(context.trigger)) return;
    const nativeSessionId = context.sessionId || context.sessionKey;
    if (!nativeSessionId) {
      this.logger.warn?.("[reme] openclaw_capture_skipped", {
        reason: "missing session id",
      });
      return;
    }
    const sessionId = openClawSessionId(
      `${context.agentId || "default"}\n${nativeSessionId}`,
    );
    const captured = captureLastTurn(messages, sessionId, prompt);
    if (captured.length !== 2) return;
    this.writes = this.writes
      .then(async () => {
        const result = await this.client.autoMemory(captured, sessionId, {
          signal: this.controller.signal,
        });
        if (!result.ok) {
          this.logger.warn?.("[reme] openclaw_auto_memory_failed", {
            sessionId,
            error: result.error,
          });
        }
      })
      .catch((error: unknown) => {
        this.logger.warn?.("[reme] openclaw_auto_memory_failed", {
          sessionId,
          error: error instanceof Error ? error.message : String(error),
        });
      });
  }

  async dispose(): Promise<void> {
    this.prompts.clear();
    let timer: ReturnType<typeof setTimeout> | undefined;
    const timeout = new Promise<void>((resolve) => {
      timer = setTimeout(() => {
        this.controller.abort();
        resolve();
      }, this.config.shutdownTimeoutMs);
    });
    await Promise.race([this.writes, timeout]);
    if (timer) clearTimeout(timer);
  }
}

function capturesTrigger(trigger: string | undefined): boolean {
  return trigger === undefined || trigger === "user";
}

function promptKey(context: OpenClawAgentContext): string {
  const nativeSessionId = context.sessionId || context.sessionKey;
  return nativeSessionId
    ? `${context.agentId || "default"}\n${nativeSessionId}`
    : "";
}
