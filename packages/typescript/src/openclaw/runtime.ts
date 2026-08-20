import type { LoggerLike, ReMeClientLike } from "../core/types.js";
import type { OpenClawReMeConfig } from "./config.js";
import { captureLastTurn, openClawSessionId } from "./messages.js";

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

  constructor(
    readonly client: ReMeClientLike,
    readonly config: OpenClawReMeConfig,
    readonly logger: LoggerLike,
  ) {}

  capture(messages: unknown[], context: OpenClawAgentContext): void {
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
    const captured = captureLastTurn(messages, sessionId);
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
