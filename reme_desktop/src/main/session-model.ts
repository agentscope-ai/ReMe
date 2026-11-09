import { UiError } from "../shared/i18n.ts";
import type {
  PaEvent,
  SessionMessage,
  SessionSnapshot,
  SessionStatus,
  Activities,
  TaskActivities,
  AgentActivity,
  NotificationWindow,
} from "../shared/protocol.ts";

const eventTypes = new Set([
  "snapshot",
  "session_changed",
  "message_upsert",
  "status_changed",
  "reset",
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
    typeof value.notification_eligible !== "boolean" ||
    !["idle", "immediate"].includes(value.delivery) ||
    (value.delivery === "immediate" &&
      (value.source !== "proactive" ||
        value.role !== "assistant" ||
        typeof value.origin_session_id !== "string" ||
        !value.origin_session_id))
  )
    throw new UiError("invalidMessage");
  // Desktop renders the public message text, not provider-native tool/event traces.
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
    notification_eligible: value.notification_eligible,
    chunks: [],
  };
}

/** The subscription owns history ordering; upserts are full replacements, not deltas. */
export class SessionModel {
  sessionId: string | null = null;
  status: SessionStatus | null = null;
  messages: SessionMessage[] = [];
  cursor: string | undefined;
  waitingForSnapshot = false;
  activities: Activities = {};
  taskActivities: TaskActivities = {};
  notifications: NotificationWindow | null = null;
  inboxMessages: SessionMessage[] = [];

  canNotify(message: SessionMessage, now: number): boolean {
    const inbox = this.inboxMessages.find((item) => item.id === message.id);
    if (
      !message.notification_eligible ||
      inbox?.notification_eligible === false
    )
      return false;
    const expires = message.expires_at ?? inbox?.expires_at;
    if (expires != null) {
      const deadline = Date.parse(expires);
      if (!Number.isFinite(deadline) || now >= deadline) return false;
    }
    if (message.delivery === "immediate") return true;
    const window = this.notifications;
    if (!window || !window.enabled) return false;
    const start = Date.parse(window.starts_at);
    const end = Date.parse(window.ends_at);
    const created = Date.parse(message.created_at ?? "");
    return (
      start <= now &&
      now < end &&
      start <= created &&
      created < end &&
      (!window.closing_message_id || window.closing_message_id === message.id)
    );
  }

  clear(): void {
    this.sessionId = null;
    this.status = null;
    this.messages = [];
    this.cursor = undefined;
    this.waitingForSnapshot = false;
    this.activities = {};
    this.taskActivities = {};
    this.notifications = null;
    this.inboxMessages = [];
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

  apply(value: unknown): PaEvent {
    if (
      !value ||
      typeof value !== "object" ||
      !eventTypes.has((value as PaEvent).type)
    )
      throw new UiError("invalidEvent");
    const event = value as PaEvent;
    let inboxMessages = this.inboxMessages;
    if (event.inbox_messages != null) {
      if (!Array.isArray(event.inbox_messages))
        throw new UiError("invalidEvent");
      const messages = new Map<string, SessionMessage>();
      for (const item of event.inbox_messages) {
        const incoming = message(item);
        if (incoming.source !== "proactive" || !incoming.origin_session_id)
          throw new UiError("invalidEvent");
        messages.set(incoming.id, incoming);
      }
      inboxMessages = [...messages.values()].slice(-100);
    }
    if (event.notifications != null) {
      const window = event.notifications;
      const start = Date.parse(window.starts_at);
      const end = Date.parse(window.ends_at);
      if (
        !Number.isFinite(start) ||
        !Number.isFinite(end) ||
        typeof window.enabled !== "boolean" ||
        start >= end ||
        (window.closing_message_id != null &&
          typeof window.closing_message_id !== "string")
      )
        throw new UiError("invalidEvent");
    }
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
        for (const [runId, activity] of Object.entries(event.task_activities)) {
          if (
            !validActivity(activity) ||
            activity.source !== "task" ||
            activity.run_id !== runId ||
            !runId ||
            typeof activity.task_id !== "string" ||
            !activity.task_id
          )
            throw new UiError("invalidActivity");
        }
      }
    }
    if (event.type === "reset") {
      this.cursor = undefined;
      this.waitingForSnapshot = true;
      return event;
    }
    if (event.type === "heartbeat") return event;
    if (event.type === "snapshot") {
      if (event.error) {
        this.waitingForSnapshot = true;
        return event;
      }
      if (event.snapshot) this.snapshot(event.snapshot);
      else {
        this.select(event.session_id ?? null);
        this.messages = [];
      }
      this.status = event.status ?? event.snapshot?.state.status ?? null;
      this.waitingForSnapshot = false;
      this.activities = event.activities;
    } else {
      if (this.waitingForSnapshot) return event;
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
        if (replacement.source === "proactive") {
          const inbox = inboxMessages.find(
            (item) => item.id === replacement.id,
          );
          replacement.origin_session_id ??= event.session_id;
          replacement.expires_at ??= inbox?.expires_at;
          replacement.notification_eligible =
            inbox?.notification_eligible ?? replacement.notification_eligible;
          inboxMessages = [
            ...inboxMessages.filter((item) => item.id !== replacement.id),
            replacement,
          ].slice(-100);
        }
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
      this.notifications = event.notifications ?? null;
      this.taskActivities = event.task_activities ?? {};
    }
    this.inboxMessages = inboxMessages;
    if (event.cursor) this.cursor = event.cursor;
    return event;
  }
}
