import type {
  PaEvent,
  SessionMessage,
  SessionSnapshot,
  SessionStatus,
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

function message(value: SessionMessage): SessionMessage {
  if (
    !value ||
    typeof value.id !== "string" ||
    !["user", "assistant"].includes(value.role) ||
    typeof value.content !== "string" ||
    !["chat", "proactive"].includes(value.source) ||
    typeof value.completed !== "boolean"
  )
    throw new Error("ReMe 会话消息格式无效");
  // Desktop renders the public message text, not provider-native tool/event traces.
  return {
    id: value.id,
    role: value.role,
    content: value.content,
    source: value.source,
    created_at: value.created_at,
    turn_id: value.turn_id,
    completed: value.completed,
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

  clear(): void {
    this.sessionId = null;
    this.status = null;
    this.messages = [];
    this.cursor = undefined;
    this.waitingForSnapshot = false;
  }

  select(sessionId: string | null): void {
    if (sessionId !== this.sessionId) {
      this.sessionId = sessionId;
      this.messages = [];
      this.status = null;
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
      throw new Error("ReMe 会话快照格式无效");
    const messages = value.messages.map(message);
    if (new Set(messages.map((entry) => entry.id)).size !== messages.length)
      throw new Error("ReMe 会话快照包含重复消息");
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
      throw new Error("ReMe 主动消息事件格式无效");
    const event = value as PaEvent;
    if (event.status != null && !statuses.has(event.status))
      throw new Error("ReMe 会话状态无效");
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
    } else {
      if (this.waitingForSnapshot) return event;
      if (event.type === "session_changed")
        this.select(event.session_id ?? null);
      else if (event.session_id !== this.sessionId)
        throw new Error("ReMe 事件与当前会话不一致，需要重新同步");
      if (event.type === "status_changed") this.status = event.status ?? null;
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
    if (event.cursor) this.cursor = event.cursor;
    return event;
  }
}
