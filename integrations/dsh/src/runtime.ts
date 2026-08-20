import { captureMessage, messagesDay, remeSessionId } from "./messages.js";
import { nextDailyRun } from "./scheduler.js";
import type {
  DshSession,
  LoggerLike,
  ReMeClientLike,
  ReMeConfig,
  ReMeMessage,
  SessionEvent,
} from "./types.js";

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
  private dreamTimer: ReturnType<typeof setTimeout> | null = null;
  private dreamTask: Promise<void> | null = null;
  private dreamController: AbortController | null = null;
  private stopping = false;

  constructor(
    readonly client: ReMeClientLike,
    readonly config: ReMeConfig,
    readonly logger: LoggerLike = console,
  ) {}

  stateFor(session: DshSession): SessionState {
    const existing = this.states.get(session.id);
    if (existing) {
      existing.session = session;
      if (existing.requestController.signal.aborted) existing.requestController = new AbortController();
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
    if (!this.config.autoMemoryEnabled) return;
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
    const hasAssistant = state.activeMessages.some((item) => item.role === "assistant");
    if (completed && hasUser && hasAssistant) {
      const day = messagesDay(state.activeMessages, this.config.timezone);
      const previousDay = state.pendingTurns.at(-1)?.day;
      if (previousDay && day && previousDay !== day) this.scheduleAutoMemory(state, true);
      state.pendingTurns.push({ messages: state.activeMessages, day });
    }
    state.activeTurn = null;
    state.activeMessages = [];
    this.scheduleAutoMemory(state);
  }

  private scheduleAutoMemory(state: SessionState, force = false): void {
    const interval = this.config.autoMemoryInterval;
    if (!force && state.pendingTurns.length < interval) return;
    const firstDay = state.pendingTurns[0]?.day;
    const dayCount = state.pendingTurns.findIndex((turn) => Boolean(firstDay && turn.day && turn.day !== firstDay));
    const available = dayCount === -1 ? state.pendingTurns.length : dayCount;
    if (!force && available < interval) return;
    const count = force ? available : interval;
    if (count === 0) return;
    const turns = state.pendingTurns.splice(0, count);
    const messages = turns.flatMap((turn) => turn.messages);
    const date = turns[0]?.day || "";
    state.unconfirmedTurns += turns.length;
    state.writes = state.writes.then(async () => {
      try {
        const result = await this.client.autoMemory(messages, state.sessionId, {
          date,
          signal: state.requestController.signal,
        });
        if (result.ok) {
          this.log("debug", "auto_memory_complete", {
            sessionId: state.sessionId,
            turns: turns.length,
          });
          return;
        }
        state.pendingTurns.unshift(...turns);
        this.log("warn", "auto_memory_failed", {
          sessionId: state.sessionId,
          error: result.error,
        });
      } catch (error) {
        state.pendingTurns.unshift(...turns);
        this.log("warn", "auto_memory_failed", {
          sessionId: state.sessionId,
          error: error instanceof Error ? error.message : String(error),
        });
      } finally {
        state.unconfirmedTurns -= turns.length;
      }
    });
  }

  start(): void {
    if (!this.config.autoDreamEnabled || this.stopping) return;
    this.scheduleDream();
  }

  private scheduleDream(): void {
    if (this.stopping || !this.config.autoDreamEnabled) return;
    let delay: number;
    try {
      delay = this.config.dreamIntervalMs > 0
        ? this.config.dreamIntervalMs
        : nextDailyRun(this.config.dreamCron).getTime() - Date.now();
    } catch (error) {
      this.log("warn", "auto_dream_schedule_invalid", {
        error: error instanceof Error ? error.message : String(error),
      });
      return;
    }
    this.dreamTimer = setTimeout(() => {
      this.dreamTimer = null;
      void this.runDream().finally(() => this.scheduleDream());
    }, delay);
    this.dreamTimer.unref?.();
  }

  async runDream(): Promise<void> {
    if (this.dreamTask) return this.dreamTask;
    this.dreamController = new AbortController();
    this.dreamTask = (async () => {
      try {
        const result = await this.client.autoDream({
          hint: this.config.dreamHint,
          signal: this.dreamController?.signal,
        });
        this.log(result.ok ? "debug" : "warn", result.ok ? "auto_dream_complete" : "auto_dream_failed", {
          error: result.ok ? undefined : result.error,
        });
      } catch (error) {
        this.log("warn", "auto_dream_failed", {
          error: error instanceof Error ? error.message : String(error),
        });
      }
    })().finally(() => {
      this.dreamTask = null;
      this.dreamController = null;
    });
    return this.dreamTask;
  }

  async dispose(session: DshSession): Promise<void> {
    const state = this.states.get(session.id);
    if (!state) return;
    if (state.requestController.signal.aborted) state.requestController = new AbortController();
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
    const completed = await this.withinShutdownBudget(flush, () => state.requestController.abort());
    const unsentTurns = state.pendingTurns.length + state.unconfirmedTurns;
    if (unsentTurns) {
      this.log("warn", completed ? "auto_memory_retained" : "auto_memory_shutdown_timeout", {
        sessionId: state.sessionId,
        unsentTurns,
      });
    } else {
      this.states.delete(session.id);
    }
  }

  async disposeAll(): Promise<void> {
    this.stopping = true;
    if (this.dreamTimer) clearTimeout(this.dreamTimer);
    this.dreamTimer = null;
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

  private async withinShutdownBudget(task: Promise<void>, abort: () => void): Promise<boolean> {
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

  private log(level: "debug" | "warn", event: string, data: Record<string, unknown>): void {
    const method = this.logger[level] ?? this.logger.log;
    method?.call(this.logger, `[reme-memory] ${event}`, data);
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}
