import { captureMessage, remeSessionId } from "./messages.js";
import { nextDailyRun } from "./scheduler.js";
import type {
  DshSession,
  LoggerLike,
  ReMeClientLike,
  ReMeConfig,
  ReMeMessage,
  SessionEvent,
} from "./types.js";

interface SessionState {
  session: DshSession;
  sessionId: string;
  activeTurn: unknown;
  activeMessages: ReMeMessage[];
  pendingTurns: ReMeMessage[][];
  writes: Promise<void>;
}

export class ReMeRuntime {
  readonly states = new Map<string, SessionState>();
  private dreamTimer: ReturnType<typeof setTimeout> | null = null;
  private dreamTask: Promise<void> | null = null;
  private stopping = false;

  constructor(
    readonly client: ReMeClientLike,
    readonly config: ReMeConfig,
    readonly logger: LoggerLike = console,
  ) {}

  stateFor(session: DshSession): SessionState {
    const existing = this.states.get(session.id);
    if (existing) return existing;
    const state: SessionState = {
      session,
      sessionId: remeSessionId(session.id),
      activeTurn: null,
      activeMessages: [],
      pendingTurns: [],
      writes: Promise.resolve(),
    };
    this.states.set(session.id, state);
    return state;
  }

  capture(session: DshSession, event: SessionEvent): void {
    if (!this.config.autoMemoryEnabled) return;
    const state = this.stateFor(session);
    if (event.type === "turn/start") {
      state.activeTurn = event.data?.turn ?? null;
      state.activeMessages = [];
      return;
    }
    const message = captureMessage(event, session.id);
    if (message) state.activeMessages.push(message);
    if (event.type !== "turn/end") return;

    const reason = event.data?.reason;
    const reasonKind = isRecord(reason) ? reason.kind : undefined;
    const completed = reasonKind === "completed" || reasonKind === "max-tokens";
    const hasUser = state.activeMessages.some((item) => item.role === "user");
    const hasAssistant = state.activeMessages.some((item) => item.role === "assistant");
    if (completed && hasUser && hasAssistant) {
      state.pendingTurns.push(state.activeMessages);
    }
    state.activeTurn = null;
    state.activeMessages = [];
    this.scheduleAutoMemory(state);
  }

  private scheduleAutoMemory(state: SessionState, force = false): void {
    const interval = this.config.autoMemoryInterval;
    if (!force && state.pendingTurns.length < interval) return;
    const count = force ? state.pendingTurns.length : interval;
    if (count === 0) return;
    const turns = state.pendingTurns.splice(0, count);
    const messages = turns.flat();
    state.writes = state.writes.then(async () => {
      const result = await this.client.autoMemory(messages, state.sessionId);
      if (!result.ok) {
        state.pendingTurns.unshift(...turns);
        this.log("warn", "auto_memory_failed", {
          sessionId: state.sessionId,
          error: result.error,
        });
        return;
      }
      this.log("debug", "auto_memory_complete", {
        sessionId: state.sessionId,
        turns: turns.length,
      });
    }).catch((error: unknown) => {
      state.pendingTurns.unshift(...turns);
      this.log("warn", "auto_memory_failed", {
        sessionId: state.sessionId,
        error: error instanceof Error ? error.message : String(error),
      });
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
    this.dreamTask = (async () => {
      try {
        const result = await this.client.autoDream({ hint: this.config.dreamHint });
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
    });
    return this.dreamTask;
  }

  async dispose(session: DshSession): Promise<void> {
    const state = this.states.get(session.id);
    if (!state) return;
    await state.writes;
    this.scheduleAutoMemory(state, true);
    await state.writes;
    this.states.delete(session.id);
  }

  async disposeAll(): Promise<void> {
    this.stopping = true;
    if (this.dreamTimer) clearTimeout(this.dreamTimer);
    this.dreamTimer = null;
    await Promise.all([...this.states.values()].map((state) => this.dispose(state.session)));
    if (this.dreamTask) await this.dreamTask;
  }

  private log(level: "debug" | "warn", event: string, data: Record<string, unknown>): void {
    const method = this.logger[level] ?? this.logger.log;
    method?.call(this.logger, `[reme-memory] ${event}`, data);
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}
