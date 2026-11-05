import type { LoggerLike, ReMeClientLike, ReMeMessage } from "./reme/types.js";
import { messagesDay, nextDailyRun } from "./scheduling.js";
import { captureMessage, remeSessionId } from "./messages.js";
import { MemoryOutbox, type MemoryBatch } from "./outbox.js";
import { safeError } from "./endpoint.js";
import type { DshSession, ReMeConfig, SessionEvent } from "./types.js";
import type { ReMeRuntimeSnapshot, ReMeRuntimeTask } from "./runtime-status.js";

interface MemoryWorker {
  controller: AbortController;
  task: Promise<void>;
  active?: ReMeRuntimeTask;
}

export class ReMeRuntime {
  private readonly states = new Map<string, ReMeMessage[]>();
  private readonly configSource: () => ReMeConfig;
  private readonly outbox: MemoryOutbox;
  private memoryTask: Promise<void> | null = null;
  private readonly memoryWorkers = new Map<string, MemoryWorker>();
  private readonly retryTimers = new Map<
    string,
    ReturnType<typeof setTimeout>
  >();
  private readonly failures = new Map<string, number>();
  private memoryError: string | undefined;
  private captureError: string | undefined;
  private dreamTimer: ReturnType<typeof setTimeout> | null = null;
  private dreamScheduleGeneration = 0;
  private dreamTask: Promise<void> | null = null;
  private dreamController: AbortController | null = null;
  private started = false;
  private stopping = false;
  private closed = false;
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
    this.outbox = new MemoryOutbox(this.configSource().autoMemoryQueueDir);
  }

  capture(session: DshSession, event: SessionEvent): void {
    const config = this.configSource();
    if (this.stopping || !config.autoMemoryEnabled) return;
    if (event.type === "turn/start") {
      this.states.set(session.id, []);
      return;
    }
    const messages = this.states.get(session.id);
    if (!messages) return;
    const message = captureMessage(event, session.id);
    if (message) messages.push(message);
    if (event.type !== "turn/end") return;
    const data = isRecord(event.data) ? event.data : undefined;
    const reason = isRecord(data?.reason) ? data.reason.kind : undefined;
    if (
      (reason === "completed" || reason === "max-tokens") &&
      messages.some((item) => item.role === "user") &&
      messages.some((item) => item.role === "assistant")
    ) {
      try {
        this.outbox.append(
          config.endpoint,
          remeSessionId(session.id),
          messagesDay(messages, config.timezone),
          messages,
          config.autoMemoryInterval,
        );
      } catch (error) {
        this.captureError = safeError(error);
        this.recordMemoryError(error);
      }
    }
    this.states.set(session.id, []);
    void this.drain();
  }

  /** Drain each endpoint independently; failed batches stay on disk for retry. */
  drain(): Promise<void> {
    if (this.closed) return Promise.resolve();
    if (this.retryTimers.has("*") || !this.configSource().autoMemoryEnabled)
      return this.memoryTask ?? Promise.resolve();
    try {
      for (;;) {
        const next = this.outbox.next(
          (batch) =>
            !this.memoryWorkers.has(batch.endpoint) && !this.blocked(batch),
        );
        if (!next) break;
        const { endpoint } = next.batch;
        const worker: MemoryWorker = {
          controller: new AbortController(),
          task: Promise.resolve(),
        };
        this.memoryWorkers.set(endpoint, worker);
        worker.task = Promise.resolve()
          .then(() => this.drainEndpoint(endpoint, worker))
          .catch((error: unknown) => {
            this.recordMemoryError(error);
            this.scheduleRetry();
          })
          .finally(() => {
            this.memoryWorkers.delete(endpoint);
            // A capture may have arrived after this worker observed an empty queue.
            if (!this.stopping) void this.drain();
          });
      }
    } catch (error) {
      this.recordMemoryError(error);
      this.scheduleRetry();
    }
    if (this.memoryTask) return this.memoryTask;
    if (!this.memoryWorkers.size) return Promise.resolve();
    // Share one completion promise instead of accumulating waiters on stalled requests.
    this.memoryTask = Promise.resolve()
      .then(async () => {
        while (this.memoryWorkers.size)
          await Promise.all(
            [...this.memoryWorkers.values()].map((worker) => worker.task),
          );
      })
      .finally(() => {
        this.memoryTask = null;
        if (!this.stopping) return this.drain();
      });
    return this.memoryTask;
  }

  private blocked(batch: MemoryBatch): boolean {
    return (
      this.retryTimers.has(endpointKey(batch.endpoint)) ||
      this.retryTimers.has(sessionKey(batch))
    );
  }

  private async drainEndpoint(
    endpoint: string,
    worker: MemoryWorker,
  ): Promise<void> {
    const { signal } = worker.controller;
    while (
      !signal.aborted &&
      !this.closed &&
      this.configSource().autoMemoryEnabled &&
      !this.retryTimers.has("*")
    ) {
      const next = this.outbox.next(
        (batch) => batch.endpoint === endpoint && !this.blocked(batch),
      );
      if (!next) return;
      const { name, batch } = next;
      const messages = batch.turns.flat();
      const task: ReMeRuntimeTask = {
        id: `${batch.id}-${++this.taskSequence}`,
        phase: "running",
        queuedAt: new Date(batch.createdAt).toISOString(),
        startedAt: new Date().toISOString(),
        turns: batch.turns.length,
        messages: messages.length,
      };
      worker.active = task;
      this.recentTasks.unshift(task);
      this.recentTasks.length = Math.min(this.recentTasks.length, 20);
      let retryKey = endpointKey(endpoint);
      try {
        const result = await this.client.autoMemory(messages, batch.sessionId, {
          date: batch.date,
          endpoint,
          requestId: batch.id,
          signal,
        });
        if (signal.aborted) {
          task.phase = "cancelled";
          return;
        }
        if (!result.ok) {
          // Payload/job errors affect this session; network, auth and service errors affect the endpoint.
          if (requestFailure(result.status)) retryKey = sessionKey(batch);
          throw new Error(
            `ReMe automatic-memory request failed (HTTP ${result.status ?? 0})`,
          );
        }
        this.outbox.acknowledge(name);
        task.phase = "completed";
        // Reply text and file names can contain personal information.
        this.failures.delete(endpointKey(endpoint));
        this.failures.delete(sessionKey(batch));
        if (!this.retryTimers.size) this.memoryError = undefined;
      } catch (error) {
        task.phase = signal.aborted ? "cancelled" : "failed";
        task.error = safeError(error);
        this.recordMemoryError(error);
        if (this.stopping || signal.aborted) return;
        this.scheduleRetry(retryKey);
      } finally {
        task.finishedAt = new Date().toISOString();
        worker.active = undefined;
      }
    }
  }

  private recordMemoryError(error: unknown): void {
    this.memoryError = safeError(error);
    this.log("warn", "auto_memory_failed", { error: this.memoryError });
  }

  private scheduleRetry(key = "*"): void {
    if (
      this.stopping ||
      this.retryTimers.has(key) ||
      !this.configSource().autoMemoryEnabled
    )
      return;
    const config = this.configSource();
    const failures = this.failures.get(key) ?? 0;
    this.failures.set(key, failures + 1);
    const delay = Math.min(
      config.autoMemoryRetryMaxMs,
      config.autoMemoryRetryMs * 2 ** Math.min(failures, 20),
    );
    const timer = setTimeout(() => {
      this.retryTimers.delete(key);
      void this.drain();
    }, delay);
    timer.unref?.();
    this.retryTimers.set(key, timer);
  }

  private clearRetries(): void {
    for (const timer of this.retryTimers.values()) clearTimeout(timer);
    this.retryTimers.clear();
    this.failures.clear();
  }

  start(): void {
    if (this.started || this.stopping) return;
    this.started = true;
    // A previous Host may have stopped before the batching interval was reached.
    try {
      this.outbox.flush();
    } catch (error) {
      this.outbox.close();
      throw error;
    }
    void this.drain();
    if (this.configSource().autoDreamEnabled)
      this.scheduleDream(this.dreamScheduleGeneration);
  }

  /** Apply settings without discarding captured work or replacing the runtime. */
  reconfigure(): void {
    if (this.stopping) return;
    this.clearRetries();
    if (!this.configSource().autoMemoryEnabled) this.states.clear();
    if (this.configSource().autoMemoryEnabled) {
      this.outbox.flush();
      void this.drain();
    }
    if (!this.started) return;
    this.dreamScheduleGeneration += 1;
    if (this.dreamTimer) clearTimeout(this.dreamTimer);
    this.dreamTimer = null;
    this.nextDreamAt = undefined;
    if (this.configSource().autoDreamEnabled)
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
        error: safeError(error),
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
    if (this.stopping) return;
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
            error: result.ok ? undefined : safeError(result.error),
          },
        );
        this.dreamLastResult = result.ok ? "completed" : "failed";
        this.dreamLastError = result.ok
          ? undefined
          : safeError(result.error || "ReMe rejected the Auto Dream request");
      } catch (error) {
        this.dreamLastResult = this.dreamController?.signal.aborted
          ? "cancelled"
          : "failed";
        this.dreamLastError = safeError(error);
        this.log("warn", "auto_dream_failed", {
          error: safeError(error),
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
    this.states.delete(session.id);
    this.outbox.flush(remeSessionId(session.id));
    void this.drain();
  }

  async disposeAll(): Promise<void> {
    if (this.stopping) return;
    this.stopping = true;
    this.started = false;
    this.states.clear();
    this.dreamScheduleGeneration += 1;
    if (this.dreamTimer) clearTimeout(this.dreamTimer);
    this.clearRetries();
    this.dreamTimer = null;
    this.nextDreamAt = undefined;
    this.dreamController?.abort();
    try {
      this.outbox.flush();
      await this.withinShutdownBudget(
        Promise.all([this.drain(), this.dreamTask]).then(() => undefined),
        () => {
          for (const worker of this.memoryWorkers.values())
            worker.controller.abort();
          this.dreamController?.abort();
        },
      );
    } finally {
      for (const worker of this.memoryWorkers.values())
        worker.controller.abort();
      this.outbox.close();
      this.closed = true;
    }
  }

  /** Return queue counts and bounded diagnostics without reply content. */
  snapshot(): ReMeRuntimeSnapshot {
    const config = this.configSource();
    const queued = this.outbox.stats();
    const running = [...this.memoryWorkers.values()].filter(
      (worker) => worker.active !== undefined,
    ).length;
    return {
      phase: this.closed
        ? "stopped"
        : this.stopping
        ? "stopping"
        : this.started
        ? "running"
        : "stopped",
      autoMemory: {
        enabled: config.autoMemoryEnabled,
        interval: config.autoMemoryInterval,
        activeSessions: this.states.size,
        queuedTurns: queued.turns,
        tasksRunning: running,
        tasksQueued: Math.max(0, queued.batches - running),
        recentTasks: this.recentTasks.map((task) => ({ ...task })),
        ...(this.captureError || this.memoryError
          ? { lastError: this.captureError || this.memoryError }
          : {}),
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
  ): Promise<void> {
    let timer: ReturnType<typeof setTimeout> | undefined;
    try {
      await Promise.race([
        task,
        new Promise<void>((resolve) => {
          timer = setTimeout(() => {
            abort();
            resolve();
          }, this.configSource().shutdownTimeoutMs);
        }),
      ]);
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

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function endpointKey(endpoint: string): string {
  return `endpoint:${endpoint}`;
}

function sessionKey(batch: MemoryBatch): string {
  return `session:${JSON.stringify([batch.endpoint, batch.sessionId])}`;
}

function requestFailure(status = 0): boolean {
  return (
    (status >= 200 && status < 300) ||
    (status >= 400 &&
      status < 500 &&
      ![401, 403, 404, 405, 408, 429].includes(status))
  );
}
