/** Lifecycle state of one DSH automatic-memory submission. */
export type ReMeRuntimeTaskPhase =
  | "queued"
  | "running"
  | "completed"
  | "failed"
  | "cancelled";

/** Bounded, content-free history entry for one automatic-memory submission. */
export interface ReMeRuntimeTask {
  id: string;
  phase: ReMeRuntimeTaskPhase;
  queuedAt: string;
  startedAt?: string;
  finishedAt?: string;
  turns: number;
  messages: number;
  result?: string;
  error?: string;
}

/** Browser-safe snapshot of the DSH-side ReMe integration runtime. */
export interface ReMeRuntimeSnapshot {
  phase: "running" | "stopping" | "stopped";
  autoMemory: {
    enabled: boolean;
    interval: number;
    activeSessions: number;
    queuedTurns: number;
    tasksRunning: number;
    tasksQueued: number;
    recentTasks: ReMeRuntimeTask[];
    lastError?: string;
  };
  autoDream: {
    enabled: boolean;
    cron: string;
    timezone: string;
    running: boolean;
    nextRunAt?: string;
    lastStartedAt?: string;
    lastFinishedAt?: string;
    lastResult?: "completed" | "failed" | "cancelled";
    lastError?: string;
  };
}
