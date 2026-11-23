import type {
  AgentMessage,
  ItemStatus,
  StreamChunk,
  TranscriptItem,
} from "./protocol.ts";
import { UiError } from "./i18n.ts";
import { clientEnums } from "./wire.ts";

export type ReplyProgress = Pick<
  AgentMessage,
  "content" | "items" | "session_id" | "turn_id" | "outcome" | "error"
>;

export const emptyReply = (): ReplyProgress => ({
  content: "",
  items: [],
  session_id: "",
  turn_id: null,
  outcome: null,
  error: null,
});

/** Reject malformed network data before it can replace a valid transcript. */
export function readMessage(value: unknown): AgentMessage {
  const message = value as AgentMessage;
  if (
    !message ||
    typeof message.id !== "string" ||
    typeof message.session_id !== "string" ||
    typeof message.content !== "string" ||
    !clientEnums.messageRole.includes(message.role) ||
    !clientEnums.messageSource.includes(message.source) ||
    typeof message.completed !== "boolean" ||
    typeof message.notification_eligible !== "boolean" ||
    !clientEnums.messageDelivery.includes(message.delivery) ||
    ![
      message.turn_id,
      message.origin_session_id,
      message.created_at,
      message.error,
    ].every((field) => field === null || typeof field === "string") ||
    !Array.isArray(message.items) ||
    (message.delivery === "immediate" &&
      (message.source !== "proactive" ||
        message.role !== "assistant" ||
        typeof message.origin_session_id !== "string" ||
        !message.origin_session_id))
  )
    throw new UiError("invalidMessage");
  for (const item of message.items) {
    if (
      !item ||
      typeof item.id !== "string" ||
      !clientEnums.itemStatus.includes(item.status)
    )
      throw new UiError("invalidMessage");
    switch (item.type) {
      case "text":
      case "summary":
        if (typeof item.text !== "string") throw new UiError("invalidMessage");
        break;
      case "tool":
        if (
          typeof item.name !== "string" ||
          typeof item.input !== "string" ||
          typeof item.output !== "string" ||
          !Array.isArray(item.actions) ||
          item.actions.some(
            (action) =>
              !action ||
              !clientEnums.toolAction.includes(action.kind) ||
              ![action.target, action.query].every(
                (field) => field === null || typeof field === "string",
              ),
          )
        )
          throw new UiError("invalidMessage");
        break;
      case "artifact":
        if (
          typeof item.url !== "string" ||
          typeof item.label !== "string" ||
          typeof item.media_type !== "string"
        )
          throw new UiError("invalidMessage");
        break;
      default:
        throw new UiError("invalidMessage");
    }
  }
  return message;
}

/** Reply streams carry the same complete messages as session subscriptions. */
export function reduceReply(
  previous: ReplyProgress,
  chunk: StreamChunk,
): ReplyProgress {
  if (chunk.chunk_type === "error") throw new Error(String(chunk.chunk));
  if (chunk.chunk_type !== "data") return previous;
  return readMessage(chunk.chunk);
}

/** A disconnected local question must not leave its tools visibly running forever. */
export function finishItems(
  items: TranscriptItem[],
  outcome: ItemStatus,
): TranscriptItem[] {
  return items.map((item) =>
    ["running", "waiting"].includes(item.status)
      ? {
          ...item,
          status:
            outcome === "completed" && item.type === "tool"
              ? "unknown"
              : outcome,
        }
      : item,
  );
}
