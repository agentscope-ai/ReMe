import { UiError } from "../shared/i18n.ts";
import { readMessage } from "../shared/transcript.ts";
import { clientEnums } from "../shared/wire.ts";
import type {
  AgentEvent,
  AgentMessage,
  AgentView,
  Activities,
} from "../shared/protocol.ts";

/** One authority for ordered messages and Agent state; updates are idempotent replacements. */
export class SessionModel {
  agent: AgentView | null = null;
  messages: AgentMessage[] = [];
  activities: Activities = {};

  get sessionId(): string | null {
    return this.agent?.session_id ?? null;
  }
  get status() {
    return this.agent?.status ?? null;
  }

  clear(): void {
    this.agent = null;
    this.messages = [];
    this.activities = {};
  }

  apply(value: unknown): AgentEvent {
    const event = value as AgentEvent;
    if (!event || !["snapshot", "update", "heartbeat"].includes(event.type))
      throw new UiError("invalidEvent");
    if (event.type === "heartbeat") return event;
    if (
      !event.agent ||
      typeof event.agent.agent_name !== "string" ||
      !clientEnums.sessionStatus.includes(event.agent.status) ||
      !event.activities ||
      Array.isArray(event.activities) ||
      !Array.isArray(event.messages)
    )
      throw new UiError("invalidSnapshot");
    for (const message of event.messages)
      if (readMessage(message).session_id !== event.agent.session_id)
        throw new UiError("invalidMessage");
    for (const activity of Object.values(event.activities)) {
      if (
        !activity ||
        typeof activity.session_id !== "string" ||
        typeof activity.turn_id !== "string" ||
        !clientEnums.activityPhase.includes(activity.phase)
      )
        throw new UiError("invalidActivity");
    }
    if (
      new Set(event.messages.map((message) => message.id)).size !==
      event.messages.length
    )
      throw new UiError("duplicateMessages");
    if (event.type === "snapshot") this.messages = event.messages;
    else {
      if (
        event.agent.session_id !== this.sessionId ||
        event.agent.agent_name !== this.agent?.agent_name
      )
        throw new UiError("sessionMismatch");
      const replacements = new Map(
        event.messages.map((message) => [message.id, message]),
      );
      const existing = new Set(this.messages.map((message) => message.id));
      this.messages = [
        ...this.messages.map(
          (message) => replacements.get(message.id) ?? message,
        ),
        ...event.messages.filter((message) => !existing.has(message.id)),
      ];
    }
    this.agent = event.agent;
    this.activities = event.activities;
    return event;
  }
}
