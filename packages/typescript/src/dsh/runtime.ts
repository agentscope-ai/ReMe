import { messagesDay, nextDailyRun } from "../core/scheduling.js";
import { captureMessage, remeSessionId } from "./messages.js";
import type { LoggerLike, ReMeClientLike, ReMeMessage } from "../core/types.js";
import type { DshSession, ReMeConfig, SessionEvent } from "./types.js";
import type { ReMeRuntimeSnapshot, ReMeRuntimeTask } from "./runtime-status.js";

interface PendingTurn {
  messages: ReMeMessage[];
  day: string;
}

interface SessionState {
  session: DshSession;
  sessionId: string;
  activeTurn: unknown;
  activeMessages: ReMeMessage[];
  pendingTurns: PendingTurn[];
  unconfirmedTurns: number;
  writes: Promise<void>;
  requestController: AbortController;
}

export class ReMeRuntime {
  readonly states = new Map<string, SessionState>();
  private readonly configSource: () => ReMeConfig;
  private dreamTimer: ReturnType<typeof setTimeout> | null = null;
  private dreamScheduleGeneration = 0;
  private dreamTask: Promise<void> | null = null;
  private dreamController: AbortController | null = null;
  private started = false;
  private stopping = false;
  private taskSequence = 0;
  private readonly recentTasks: ReMeRuntimeTask[] = [];
  private nextDreamAt: string | undefined;
  private dreamLastStartedAt: string | undefined;
  private dreamLastFinishedAt: string | undefined;
  private dreamLastResult: "completed" | "failed" | "cancelled" | undefined;
  private dreamLastError: string | undefined;

  constructor(
    readonly client: ReMeClientLike,
    config: ReMeConfig | (() => ReMeConfig),
    readonly logger: LoggerLike = console,
  ) {
    this.configSource = typeof config === "function" ? config : () => config;
  }

  stateFor(session: DshSession): SessionState {
    const existing = this.states.get(session.id);
    if (existing) {
      existing.session = session;
      if (existing.requestController.signal.aborted)
        existing.requestController = new AbortController();
      return existing;
    }
    const state: SessionState = {
      session,
      sessionId: remeSessionId(session.id),
      activeTurn: null,
      activeMessages: [],
      pendingTurns: [],
      unconfirmedTurns: 0,
      writes: Promise.resolve(),
      requestController: new AbortController(),
    };
    this.states.set(session.id, state);
    return state;
  }

  capture(session: DshSession, event: SessionEvent): void {
    const config = this.configSource();
    if (!config.autoMemoryEnabled) return;
    const state = this.stateFor(session);
    const data = isRecord(event.data) ? event.data : undefined;
    if (event.type === "turn/start") {
      state.activeTurn = data?.turn ?? null;
      state.activeMessages = [];
      return;
    }
    const message = captureMessage(event, session.id);
    if (message) state.activeMessages.push(message);
    if (event.type !== "turn/end") return;

    const reason = data?.reason;
    const reasonKind = isRecord(reason) ? reason.kind : undefined;
    const completed = reasonKind === "completed" || reasonKind === "max-tokens";
    const hasUser = state.activeMessages.some((item) => item.role === "user");
    const hasAssistant = state.activeMessages.some(
      (item) => item.role === "assistant",
    );
    if (completed && hasUser && hasAssistant) {
      const day = messagesDay(state.activeMessages, config.timezone);
      const previousDay = state.pendingTurns.at(-1)?.day;
      if (previousDay && day && previousDay !== day)
        this.scheduleAutoMemory(state, true);
      state.pendingTurns.push({ messages: state.activeMessages, day });
    }
    state.activeTurn = null;
    state.activeMessages = [];
    this.scheduleAutoMemory(state);
  }

  private scheduleAutoMemory(state: SessionState, force = false): void {
    const interval = this.configSource().autoMemoryInterval;
    const firstDay = state.pendingTurns[0]?.day;
    const dayCount = state.pendingTurns.findIndex((turn) =>
      Boolean(firstDay && turn.day && turn.day !== firstDay),
    );
    const available = dayCount === -1 ? state.pendingTurns.length : dayCount;
    const crossesDayBoundary = dayCount !== -1;
    if (!force && !crossesDayBoundary && available < interval) return;
    const count = force || crossesDayBoundary ? available : interval;
    if (count === 0) return;
    const turns = state.pendingTurns.splice(0, count);
    const messages = turns.flatMap((turn) => turn.messages);
    const date = turns[0]?.day || "";
    const task: ReMeRuntimeTask = {
      id: `auto-memory-${++this.taskSequence}`,
      phase: "queued",
      queuedAt: new Date().toISOString(),
      turns: turns.length,
      messages: messages.length,
    };
    this.recentTasks.unshift(task);
    this.recentTasks.length = Math.min(this.recentTasks.length, 20);
    state.unconfirmedTurns += turns.length;
    state.writes = state.writes.then(async () => {
      task.phase = "running";
      task.startedAt = new Date().toISOString();
      try {
        const result = await this.client.autoMemory(messages, state.sessionId, {
          date,
          signal: state.requestController.signal,
        });
        if (result.ok) {
          task.phase = "completed";
          task.result = resultSummary(result.answer, result.metadata);
          this.log("debug", "auto_memory_complete", {
            sessionId: state.sessionId,
            turns: turns.length,
          });
          return;
        }
        state.pendingTurns.unshift(...turns);
        task.phase = "failed";
        task.error =
          result.error || "ReMe rejected the automatic-memory request";
        this.log("warn", "auto_memory_failed", {
          sessionId: state.sessionId,
          error: result.error,
        });
      } catch (error) {
        state.pendingTurns.unshift(...turns);
        task.phase = state.requestController.signal.aborted
          ? "cancelled"
          : "failed";
        task.error = error instanceof Error ? error.message : String(error);
        this.log("warn", "auto_memory_failed", {
          sessionId: state.sessionId,
          error: error instanceof Error ? error.message : String(error),
        });
      } finally {
        task.finishedAt = new Date().toISOString();
        state.unconfirmedTurns -= turns.length;
      }
    });
  }

  start(): void {
    this.started = true;
    if (!this.configSource().autoDreamEnabled || this.stopping) return;
    this.scheduleDream(this.dreamScheduleGeneration);
  }

  /** Apply a changed settings snapshot to pending batching and dream scheduling. */
  reconfigure(): void {
    if (this.stopping) return;
    const config = this.configSource();
    if (config.autoMemoryEnabled) {
      for (const state of this.states.values()) this.scheduleAutoMemory(state);
    }
    if (!this.started) return;
    this.dreamScheduleGeneration += 1;
    if (this.dreamTimer) clearTimeout(this.dreamTimer);
    this.dreamTimer = null;
    this.nextDreamAt = undefined;
    if (config.autoDreamEnabled)
      this.scheduleDream(this.dreamScheduleGeneration);
  }

  private scheduleDream(generation: number): void {
    const config = this.configSource();
    if (
      generation !== this.dreamScheduleGeneration ||
      this.stopping ||
      !config.autoDreamEnabled
    )
      return;
    let delay: number;
    try {
      delay =
        config.dreamIntervalMs > 0
          ? config.dreamIntervalMs
          : nextDailyRun(config.dreamCron, config.timezone).getTime() -
            Date.now();
    } catch (error) {
      this.log("warn", "auto_dream_schedule_invalid", {
        error: error instanceof Error ? error.message : String(error),
      });
      return;
    }
    this.nextDreamAt = new Date(Date.now() + delay).toISOString();
    this.dreamTimer = setTimeout(() => {
      this.dreamTimer = null;
      this.nextDreamAt = undefined;
      void this.runDream().finally(() => this.scheduleDream(generation));
    }, delay);
    this.dreamTimer.unref?.();
  }

  async runDream(): Promise<void> {
    if (this.dreamTask) return this.dreamTask;
    const config = this.configSource();
    this.dreamController = new AbortController();
    this.dreamLastStartedAt = new Date().toISOString();
    this.dreamLastFinishedAt = undefined;
    this.dreamLastResult = undefined;
    this.dreamLastError = undefined;
    this.dreamTask = (async () => {
      try {
        const result = await this.client.autoDream({
          hint: config.dreamHint,
          signal: this.dreamController?.signal,
        });
        this.log(
          result.ok ? "debug" : "warn",
          result.ok ? "auto_dream_complete" : "auto_dream_failed",
          {
            error: result.ok ? undefined : result.error,
          },
        );
        this.dreamLastResult = result.ok ? "completed" : "failed";
        this.dreamLastError = result.ok
          ? undefined
          : result.error || "ReMe rejected the Auto Dream request";
      } catch (error) {
        this.dreamLastResult = this.dreamController?.signal.aborted
          ? "cancelled"
          : "failed";
        this.dreamLastError =
          error instanceof Error ? error.message : String(error);
        this.log("warn", "auto_dream_failed", {
          error: error instanceof Error ? error.message : String(error),
        });
      }
    })().finally(() => {
      this.dreamLastFinishedAt = new Date().toISOString();
      this.dreamTask = null;
      this.dreamController = null;
    });
    return this.dreamTask;
  }

  async dispose(session: DshSession): Promise<void> {
    const state = this.states.get(session.id);
    if (!state) return;
    if (state.requestController.signal.aborted)
      state.requestController = new AbortController();
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
      state.requestController.abort(),
    );
    const unsentTurns = state.pendingTurns.length + state.unconfirmedTurns;
    if (unsentTurns) {
      this.log(
        "warn",
        completed ? "auto_memory_retained" : "auto_memory_shutdown_timeout",
        {
          sessionId: state.sessionId,
          unsentTurns,
        },
      );
    } else {
      this.states.delete(session.id);
    }
  }

  async disposeAll(): Promise<void> {
    this.stopping = true;
    this.started = false;
    this.dreamScheduleGeneration += 1;
    if (this.dreamTimer) clearTimeout(this.dreamTimer);
    this.dreamTimer = null;
    this.nextDreamAt = undefined;
    this.dreamController?.abort();
    const shutdown = Promise.all([
      ...[...this.states.values()].map((state) => this.dispose(state.session)),
      ...(this.dreamTask ? [this.dreamTask] : []),
    ]).then(() => undefined);
    await this.withinShutdownBudget(shutdown, () => {
      this.dreamController?.abort();
      for (const state of this.states.values()) state.requestController.abort();
    });
  }

  /** Return a content-free snapshot for the local DSH status page. */
  snapshot(): ReMeRuntimeSnapshot {
    const config = this.configSource();
    const tasksRunning = this.recentTasks.filter(
      (task) => task.phase === "running",
    ).length;
    const tasksQueued = this.recentTasks.filter(
      (task) => task.phase === "queued",
    ).length;
    const lastError = this.recentTasks.find(
      (task) => task.phase === "failed" || task.phase === "cancelled",
    )?.error;
    return {
      phase: this.stopping ? "stopping" : this.started ? "running" : "stopped",
      autoMemory: {
        enabled: config.autoMemoryEnabled,
        interval: config.autoMemoryInterval,
        activeSessions: this.states.size,
        queuedTurns: [...this.states.values()].reduce(
          (total, state) =>
            total + state.pendingTurns.length + state.unconfirmedTurns,
          0,
        ),
        tasksRunning,
        tasksQueued,
        recentTasks: this.recentTasks.map((task) => ({ ...task })),
        ...(lastError ? { lastError } : {}),
      },
      autoDream: {
        enabled: config.autoDreamEnabled,
        cron: config.dreamCron,
        timezone: config.timezone,
        running: this.dreamTask !== null,
        ...(this.nextDreamAt ? { nextRunAt: this.nextDreamAt } : {}),
        ...(this.dreamLastStartedAt
          ? { lastStartedAt: this.dreamLastStartedAt }
          : {}),
        ...(this.dreamLastFinishedAt
          ? { lastFinishedAt: this.dreamLastFinishedAt }
          : {}),
        ...(this.dreamLastResult ? { lastResult: this.dreamLastResult } : {}),
        ...(this.dreamLastError ? { lastError: this.dreamLastError } : {}),
      },
    };
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
      }, this.configSource().shutdownTimeoutMs);
    });
    try {
      return await Promise.race([task.then(() => true), timeout]);
    } finally {
      if (timer) clearTimeout(timer);
    }
  }

  private log(
    level: "debug" | "warn",
    event: string,
    data: Record<string, unknown>,
  ): void {
    const method = this.logger[level] ?? this.logger.log;
    method?.call(this.logger, `[reme-memory] ${event}`, data);
  }
}

function resultSummary(
  answer: unknown,
  metadata: Record<string, unknown> | undefined,
): string | undefined {
  const path = metadata?.path;
  if (typeof path === "string" && path.length > 0) return path;
  if (typeof answer !== "string") return undefined;
  const normalized = answer.trim();
  if (!normalized) return undefined;
  return normalized.length > 240 ? `${normalized.slice(0, 237)}…` : normalized;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}
