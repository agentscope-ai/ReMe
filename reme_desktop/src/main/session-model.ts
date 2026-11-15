import { UiError } from "../shared/i18n.ts";
import type {
  AgentEvent,
  SessionMessage,
  SessionSnapshot,
  SessionStatus,
  Activities,
  TaskActivities,
  AgentActivity,
} from "../shared/protocol.ts";

const eventTypes = new Set([
  "snapshot",
  "session_changed",
  "message_upsert",
  "status_changed",
  "heartbeat",
]);
const statuses = new Set(["active", "idle", "not_loaded", "error"]);
const phases = new Set([
  "thinking",
  "working",
  "remembering",
  "answering",
  "waiting",
  "completed",
  "failed",
  "interrupted",
]);
const validActivity = (activity: AgentActivity) =>
  activity &&
  typeof activity.session_id === "string" &&
  typeof activity.turn_id === "string" &&
  phases.has(activity.phase);

function message(value: SessionMessage): SessionMessage {
  if (
    !value ||
    typeof value.id !== "string" ||
    !["user", "assistant"].includes(value.role) ||
    typeof value.content !== "string" ||
    !["chat", "proactive"].includes(value.source) ||
    typeof value.completed !== "boolean" ||
    (value.notification_eligible !== undefined &&
      typeof value.notification_eligible !== "boolean") ||
    !["idle", "immediate"].includes(value.delivery) ||
    (value.delivery === "immediate" &&
      (value.source !== "proactive" ||
        value.role !== "assistant" ||
        typeof value.origin_session_id !== "string" ||
        !value.origin_session_id))
  )
    throw new UiError("invalidMessage");
  // Preserve text-message boundaries without forwarding provider tool/event traces.
  return {
    id: value.id,
    role: value.role,
    content: value.content,
    source: value.source,
    created_at: value.created_at,
    expires_at: value.expires_at,
    turn_id: value.turn_id,
    completed: value.completed,
    delivery: value.delivery,
    origin_session_id: value.origin_session_id,
    notification_eligible: value.notification_eligible ?? false,
    chunks: (value.chunks ?? [])
      .filter(
        (chunk) =>
          chunk.chunk_type === "content" && typeof chunk.chunk === "string",
      )
      .map(({ chunk_type, chunk, block_id, event }) => ({
        chunk_type,
        chunk,
        block_id,
        event,
      })),
  };
}

/** The subscription owns history ordering; upserts are full replacements, not deltas. */
export class SessionModel {
  sessionId: string | null = null;
  status: SessionStatus | null = null;
  messages: SessionMessage[] = [];
  activities: Activities = {};
  taskActivities: TaskActivities = {};

  canNotify(message: SessionMessage, now: number): boolean {
    if (!message.notification_eligible) return false;
    const expires = message.expires_at;
    if (expires != null) {
      const deadline = Date.parse(expires);
      if (!Number.isFinite(deadline) || now >= deadline) return false;
    }
    return true;
  }

  clear(): void {
    this.sessionId = null;
    this.status = null;
    this.messages = [];
    this.activities = {};
    this.taskActivities = {};
  }

  select(sessionId: string | null): void {
    if (sessionId !== this.sessionId) {
      this.sessionId = sessionId;
      this.messages = [];
      this.status = null;
      this.activities = {};
    }
  }

  snapshot(value: SessionSnapshot): void {
    if (
      !value ||
      typeof value.session_id !== "string" ||
      !Array.isArray(value.messages) ||
      value.state?.session_id !== value.session_id ||
      !statuses.has(value.state?.status)
    )
      throw new UiError("invalidSnapshot");
    const messages = value.messages.map(message);
    if (new Set(messages.map((entry) => entry.id)).size !== messages.length)
      throw new UiError("duplicateMessages");
    this.select(value.session_id);
    this.messages = messages;
    this.status = value.state.status;
  }

  apply(value: unknown): AgentEvent {
    if (
      !value ||
      typeof value !== "object" ||
      !eventTypes.has((value as AgentEvent).type)
    )
      throw new UiError("invalidEvent");
    const event = value as AgentEvent;
    if (event.status != null && !statuses.has(event.status))
      throw new UiError("invalidStatus");
    if (["snapshot", "status_changed"].includes(event.type)) {
      if (
        !event.activities ||
        typeof event.activities !== "object" ||
        Array.isArray(event.activities)
      )
        throw new UiError("missingActivities");
      for (const [source, activity] of Object.entries(event.activities)) {
        if (
          !validActivity(activity) ||
          activity.source !== source ||
          !["chat", "proactive"].includes(source) ||
          activity.session_id !== event.session_id
        )
          throw new UiError("invalidActivity");
      }
      if (event.task_activities !== undefined) {
        if (
          !event.task_activities ||
          typeof event.task_activities !== "object" ||
          Array.isArray(event.task_activities)
        )
          throw new UiError("invalidActivity");
        for (const [taskId, activity] of Object.entries(
          event.task_activities,
        )) {
          if (
            !validActivity(activity) ||
            activity.source !== "task" ||
            activity.task_id !== taskId ||
            !taskId ||
            typeof activity.started_at !== "string" ||
            !Number.isFinite(Date.parse(activity.started_at))
          )
            throw new UiError("invalidActivity");
        }
      }
    }
    if (event.type === "heartbeat") return event;
    if (event.type === "snapshot") {
      if (event.snapshot) this.snapshot(event.snapshot);
      else {
        this.select(event.session_id ?? null);
        this.messages = [];
      }
      this.status = event.status ?? event.snapshot?.state.status ?? null;
      this.activities = event.activities;
    } else {
      if (event.type === "session_changed")
        this.select(event.session_id ?? null);
      else if (event.session_id !== this.sessionId)
        throw new UiError("sessionMismatch");
      if (event.type === "status_changed") {
        this.status = event.status ?? null;
        this.activities = event.activities;
      }
      if (event.type === "message_upsert") {
        const replacement = message(event.message!);
        const index = this.messages.findIndex(
          (entry) => entry.id === replacement.id,
        );
        if (index < 0) this.messages = [...this.messages, replacement];
        else
          this.messages = this.messages.map((entry, position) =>
            position === index ? replacement : entry,
          );
      }
    }
    if (event.type === "snapshot" || event.type === "status_changed") {
      this.taskActivities = event.task_activities ?? {};
    }
    return event;
  }
}
