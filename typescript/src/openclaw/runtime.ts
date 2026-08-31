import type { PluginHookAgentContext } from "openclaw/plugin-sdk/types";

import {
  dateInTimezone,
  messagesDay,
  nextDailyRun,
} from "../core/scheduling.js";
import type { LoggerLike, ReMeClientLike, ReMeMessage } from "../core/types.js";
import type { OpenClawReMeConfig } from "./config.js";
import { captureLastTurn, openClawSessionId } from "./messages.js";

const MAX_PENDING_PROMPTS = 256;

interface PendingTurn {
  messages: ReMeMessage[];
  day: string;
}

interface SessionState {
  sessionId: string;
  pendingTurns: PendingTurn[];
  unconfirmedTurns: number;
  writes: Promise<void>;
  controller: AbortController;
}

export interface OpenClawRuntimeSnapshot {
  phase: "stopped" | "running" | "stopping";
  autoMemory: {
    enabled: boolean;
    interval: number;
    activeSessions: number;
    queuedTurns: number;
  };
  autoDream: {
    enabled: boolean;
    cron: string;
    timezone: string;
    running: boolean;
    nextRunAt?: string;
    lastResult?: "completed" | "failed" | "cancelled";
    lastError?: string;
  };
}

/**
 * OpenClaw lifecycle adapter with the same reliability model as the DSH
 * integration: per-session serialized batches, retryable failures, bounded
 * shutdown flushing, and one application-owned Auto Dream schedule.
 */
export class OpenClawReMeRuntime {
  readonly states = new Map<string, SessionState>();
  private readonly prompts = new Map<string, string>();
  private dreamTimer: ReturnType<typeof setTimeout> | null = null;
  private dreamTask: Promise<void> | null = null;
  private dreamController: AbortController | null = null;
  private started = false;
  private stopping = false;
  private nextDreamAt: string | undefined;
  private dreamLastResult: "completed" | "failed" | "cancelled" | undefined;
  private dreamLastError: string | undefined;

  constructor(
    readonly client: ReMeClientLike,
    readonly config: OpenClawReMeConfig,
    readonly logger: LoggerLike,
  ) {}

  /** Restrict automatic behavior to conversational root-agent turns. */
  accepts(context: PluginHookAgentContext): boolean {
    if (
      this.config.rootAgentsOnly &&
      context.sessionKey?.includes(":subagent:")
    )
      return false;
    return (
      context.trigger === undefined ||
      context.trigger === "user" ||
      context.trigger === "manual"
    );
  }

  /** Retain the unmodified user prompt so injected context is never recaptured. */
  rememberPrompt(prompt: string, context: PluginHookAgentContext): void {
    if (!this.config.autoMemoryEnabled || !this.accepts(context)) return;
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

  takePrompt(context: PluginHookAgentContext): string | undefined {
    const key = promptKey(context);
    if (!key) return undefined;
    const prompt = this.prompts.get(key);
    this.prompts.delete(key);
    return prompt;
  }

  /** Queue one completed OpenClaw user/assistant pair for automatic memory. */
  capture(
    messages: unknown[],
    context: PluginHookAgentContext,
    prompt?: string,
  ): void {
    if (!this.config.autoMemoryEnabled || !this.accepts(context)) return;
    const key = sessionKey(context);
    if (!key) {
      this.logger.warn?.("[reme] openclaw_capture_skipped", {
        reason: "missing session id",
      });
      return;
    }
    const state = this.stateFor(key, context.agentId);
    const captured = captureLastTurn(messages, state.sessionId, prompt);
    if (captured.length !== 2) return;
    const day =
      messagesDay(captured, this.config.timezone) ||
      dateInTimezone(new Date(), this.config.timezone);
    const previousDay = state.pendingTurns.at(-1)?.day;
    if (previousDay && previousDay !== day)
      this.scheduleAutoMemory(state, true);
    state.pendingTurns.push({ messages: captured, day });
    this.scheduleAutoMemory(state);
  }

  /** Start the single Auto Dream timer when the Gateway starts the service. */
  start(): void {
    if (this.started || this.stopping) return;
    this.started = true;
    if (!this.config.autoDreamEnabled) return;
    this.scheduleDream();
  }

  /** Execute Auto Dream now; concurrent callers share the same task. */
  async runDream(): Promise<void> {
    if (this.dreamTask) return this.dreamTask;
    this.dreamController = new AbortController();
    this.dreamLastResult = undefined;
    this.dreamLastError = undefined;
    this.dreamTask = (async () => {
      try {
        const result = await this.client.autoDream({
          hint: this.config.dreamHint,
          signal: this.dreamController?.signal,
        });
        this.dreamLastResult = result.ok ? "completed" : "failed";
        this.dreamLastError = result.ok
          ? undefined
          : result.error || "ReMe rejected the Auto Dream request";
        this.logger[result.ok ? "debug" : "warn"]?.(
          result.ok
            ? "[reme] openclaw_auto_dream_complete"
            : "[reme] openclaw_auto_dream_failed",
          result.ok ? undefined : { error: this.dreamLastError },
        );
      } catch (error) {
        this.dreamLastResult = this.dreamController?.signal.aborted
          ? "cancelled"
          : "failed";
        this.dreamLastError = errorMessage(error);
        this.logger.warn?.("[reme] openclaw_auto_dream_failed", {
          error: this.dreamLastError,
        });
      }
    })().finally(() => {
      this.dreamTask = null;
      this.dreamController = null;
    });
    return this.dreamTask;
  }

  /** Flush one host session at an explicit OpenClaw session boundary. */
  async disposeSession(context: PluginHookAgentContext): Promise<void> {
    const key = sessionKey(context);
    if (!key) return;
    const state = this.states.get(key);
    if (!state) return;
    await this.flushState(state);
    if (!state.pendingTurns.length && state.unconfirmedTurns === 0)
      this.states.delete(key);
  }

  /** Bound all outstanding work to the configured Gateway shutdown budget. */
  async disposeAll(): Promise<void> {
    this.stopping = true;
    this.started = false;
    this.prompts.clear();
    if (this.dreamTimer) clearTimeout(this.dreamTimer);
    this.dreamTimer = null;
    this.nextDreamAt = undefined;
    this.dreamController?.abort();
    const shutdown = Promise.all([
      ...[...this.states.values()].map((state) => this.flushState(state)),
      ...(this.dreamTask ? [this.dreamTask] : []),
    ]).then(() => undefined);
    await this.withinShutdownBudget(shutdown, () => {
      this.dreamController?.abort();
      for (const state of this.states.values()) state.controller.abort();
    });
    for (const [key, state] of this.states) {
      if (!state.pendingTurns.length && state.unconfirmedTurns === 0)
        this.states.delete(key);
    }
  }

  /** Return content-free diagnostics suitable for tests and operator surfaces. */
  snapshot(): OpenClawRuntimeSnapshot {
    return {
      phase: this.stopping ? "stopping" : this.started ? "running" : "stopped",
      autoMemory: {
        enabled: this.config.autoMemoryEnabled,
        interval: this.config.autoMemoryInterval,
        activeSessions: this.states.size,
        queuedTurns: [...this.states.values()].reduce(
          (total, state) =>
            total + state.pendingTurns.length + state.unconfirmedTurns,
          0,
        ),
      },
      autoDream: {
        enabled: this.config.autoDreamEnabled,
        cron: this.config.dreamCron,
        timezone: this.config.timezone,
        running: this.dreamTask !== null,
        ...(this.nextDreamAt ? { nextRunAt: this.nextDreamAt } : {}),
        ...(this.dreamLastResult ? { lastResult: this.dreamLastResult } : {}),
        ...(this.dreamLastError ? { lastError: this.dreamLastError } : {}),
      },
    };
  }

  private stateFor(key: string, agentId?: string): SessionState {
    const existing = this.states.get(key);
    if (existing) {
      if (existing.controller.signal.aborted)
        existing.controller = new AbortController();
      return existing;
    }
    const state: SessionState = {
      sessionId: openClawSessionId(`${agentId || "default"}\n${key}`),
      pendingTurns: [],
      unconfirmedTurns: 0,
      writes: Promise.resolve(),
      controller: new AbortController(),
    };
    this.states.set(key, state);
    return state;
  }

  private scheduleAutoMemory(state: SessionState, force = false): void {
    const firstDay = state.pendingTurns[0]?.day;
    const dayCount = state.pendingTurns.findIndex((turn) =>
      Boolean(firstDay && turn.day && turn.day !== firstDay),
    );
    const available = dayCount === -1 ? state.pendingTurns.length : dayCount;
    const crossesDayBoundary = dayCount !== -1;
    if (
      !force &&
      !crossesDayBoundary &&
      available < this.config.autoMemoryInterval
    )
      return;
    const count =
      force || crossesDayBoundary ? available : this.config.autoMemoryInterval;
    if (count === 0) return;
    const turns = state.pendingTurns.splice(0, count);
    const messages = turns.flatMap((turn) => turn.messages);
    state.unconfirmedTurns += turns.length;
    state.writes = state.writes.then(async () => {
      try {
        const result = await this.client.autoMemory(messages, state.sessionId, {
          date: turns[0]?.day || "",
          signal: state.controller.signal,
        });
        if (result.ok) return;
        state.pendingTurns.unshift(...turns);
        this.logger.warn?.("[reme] openclaw_auto_memory_failed", {
          sessionId: state.sessionId,
          error: result.error,
        });
      } catch (error) {
        state.pendingTurns.unshift(...turns);
        this.logger.warn?.("[reme] openclaw_auto_memory_failed", {
          sessionId: state.sessionId,
          error: errorMessage(error),
        });
      } finally {
        state.unconfirmedTurns -= turns.length;
      }
    });
  }

  private async flushState(state: SessionState): Promise<void> {
    if (state.controller.signal.aborted)
      state.controller = new AbortController();
    const flush = (async () => {
      await state.writes;
      const retryCount = state.pendingTurns.length;
      let scheduled = 0;
      while (state.pendingTurns.length && scheduled < retryCount) {
        const before = state.pendingTurns.length;
        this.scheduleAutoMemory(state, true);
        scheduled += before - state.pendingTurns.length;
      }
      await state.writes;
    })();
    const completed = await this.withinShutdownBudget(flush, () =>
      state.controller.abort(),
    );
    const unsentTurns = state.pendingTurns.length + state.unconfirmedTurns;
    if (unsentTurns) {
      this.logger.warn?.(
        completed
          ? "[reme] openclaw_auto_memory_retained"
          : "[reme] openclaw_auto_memory_shutdown_timeout",
        { sessionId: state.sessionId, unsentTurns },
      );
    }
  }

  private scheduleDream(): void {
    if (this.stopping || !this.config.autoDreamEnabled) return;
    const delay =
      nextDailyRun(this.config.dreamCron, this.config.timezone).getTime() -
      Date.now();
    this.nextDreamAt = new Date(Date.now() + delay).toISOString();
    this.dreamTimer = setTimeout(() => {
      this.dreamTimer = null;
      this.nextDreamAt = undefined;
      void this.runDream().finally(() => this.scheduleDream());
    }, delay);
    this.dreamTimer.unref?.();
  }

  private async withinShutdownBudget(
    task: Promise<void>,
    abort: () => void,
  ): Promise<boolean> {
    let timer: ReturnType<typeof setTimeout> | undefined;
    const timeout = new Promise<boolean>((resolve) => {
      timer = setTimeout(() => {
        abort();
        resolve(false);
      }, this.config.shutdownTimeoutMs);
    });
    try {
      return await Promise.race([task.then(() => true), timeout]);
    } finally {
      if (timer) clearTimeout(timer);
    }
  }
}

function sessionKey(context: PluginHookAgentContext): string {
  return context.sessionId || context.sessionKey || "";
}

function promptKey(context: PluginHookAgentContext): string {
  if (context.runId) return `run:${context.runId}`;
  const key = sessionKey(context);
  return key ? `session:${context.agentId || "default"}\n${key}` : "";
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
